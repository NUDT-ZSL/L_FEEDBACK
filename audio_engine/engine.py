"""End-to-end pipeline: load -> quantize -> loop -> layer-mix -> duck ->
schedule SFX -> soft-limit -> atomic write of WAV + JSON report."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

from . import ENGINE_SAMPLE_RATE
from .ducking import build_duck_envelope, envelope_max_slope_db_per_s
from .errors import BusinessError, ERR_INVALID_BPM
from .layers import LAYER_NAMES, layer_gain_envelopes
from .loader import load_wav, resample_to_engine
from .looping import loop_extend
from .quantize import quantize_time
from .sfx import schedule_sfx, soft_limit

PEAK_CEILING_DBFS = -1.0
LOOP_CROSSFADE_S = 0.010
TAIL_AFTER_LAST_TRIGGER_S = 2.0


def _resolve(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 2:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    raise ValueError(f"unsupported channel count: {data.shape[1]}")


def _atomic_write_wav(path: str, data: np.ndarray, sample_rate: int) -> None:
    tmp = path + ".tmp"
    try:
        sf.write(tmp, data, sample_rate, subtype="PCM_16", format="WAV")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run(config: dict, output_path: str, report_path: str) -> dict:
    """Run the full pipeline and return the quality report dict."""
    sr = ENGINE_SAMPLE_RATE
    config_dir = config.get("_config_dir", ".")

    # --- input validation (fail before touching any output) -------------
    bpm = float(config.get("bpm", 0.0))
    if bpm <= 0.0:
        raise BusinessError(ERR_INVALID_BPM, f"bpm must be > 0, got {bpm}")
    first_beat_offset = float(config.get("first_beat_offset", 0.0))
    duck_gain_db = float(config.get("duck_gain_db", -6.0))

    layer_paths = {name: _resolve(config_dir, config["layers"][name]) for name in LAYER_NAMES}
    sfx_paths = {k: _resolve(config_dir, v) for k, v in config["sfx"].items()}

    layers: dict[str, np.ndarray] = {}
    for name in LAYER_NAMES:
        data, src_sr = load_wav(layer_paths[name])
        layers[name] = resample_to_engine(data, src_sr, sr)

    sfx_bank: dict[str, np.ndarray] = {}
    for name, path in sfx_paths.items():
        data, src_sr = load_wav(path)
        sfx_bank[name] = resample_to_engine(data, src_sr, sr)

    # --- BPM grid quantization ------------------------------------------
    events = []
    for i, ev in enumerate(config["events"]):
        grid = ev.get("grid", "1/16")
        quantized, err_ms = quantize_time(float(ev["time"]), first_beat_offset, bpm, grid)
        sfx_name = ev["sfx"]
        if sfx_name not in sfx_bank:
            raise BusinessError("ERR_MISSING_FILE", f"unknown sfx id {sfx_name!r} (no WAV mapped)")
        events.append(
            {
                "index": i,
                "sfx": sfx_name,
                "grid": grid,
                "time": float(ev["time"]),
                "quantized": quantized,
                "error_ms": err_ms,
                "gain": 10.0 ** (float(ev.get("gain_db", 0.0)) / 20.0),
                "pan": float(ev.get("pan", 0.0)),
            }
        )
    if not events:
        raise BusinessError(ERR_INVALID_BPM, "event timeline is empty")

    # --- total duration: last quantized trigger + 2 s (plus SFX tail) ---
    last_q = max(ev["quantized"] for ev in events)
    total_s = last_q + TAIL_AFTER_LAST_TRIGGER_S
    for ev in events:
        sfx_dur = sfx_bank[ev["sfx"]].shape[0] / sr
        total_s = max(total_s, ev["quantized"] + sfx_dur)
    n_samples = int(round(total_s * sr))
    total_s = n_samples / sr

    # --- whole-bar loop extension of BGM layers --------------------------
    bar_len = int(round(4 * 60.0 / bpm * sr))
    xf_len = int(round(LOOP_CROSSFADE_S * sr))
    loop_counts: dict[str, int] = {}
    extended: dict[str, np.ndarray] = {}
    for name in LAYER_NAMES:
        extended[name], loop_counts[name] = loop_extend(
            layers[name], n_samples, bar_len, xf_len
        )

    # --- combo-state layered mix -----------------------------------------
    fps = float(config.get("combo_fps", 30.0))
    envelopes = layer_gain_envelopes(
        config.get("combo_states", []), fps, n_samples, sr
    )
    bgm = np.zeros((n_samples, 2), dtype=np.float64)
    for name in LAYER_NAMES:
        bgm += _to_stereo(extended[name]) * envelopes[name][:, None]

    # --- BGM ducking under each SFX interval ------------------------------
    intervals = []
    for ev in events:
        sfx_dur = sfx_bank[ev["sfx"]].shape[0] / sr
        intervals.append((ev["quantized"], ev["quantized"] + sfx_dur))
    duck_env = build_duck_envelope(n_samples, sr, intervals, duck_gain_db)
    bgm *= duck_env[:, None]

    # --- SFX scheduling with gain/pan -------------------------------------
    bus = np.zeros((n_samples, 2), dtype=np.float64)
    for ev in events:
        start = int(round(ev["quantized"] * sr))
        schedule_sfx(bus, sfx_bank[ev["sfx"]], start, ev["gain"], ev["pan"])

    # --- final mix + soft limiting to the -1 dBFS ceiling -----------------
    ceiling = 10.0 ** (PEAK_CEILING_DBFS / 20.0)
    mix = soft_limit(bgm + bus, ceiling)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    peak_dbfs = 20.0 * np.log10(peak) if peak > 0.0 else -120.0

    # --- quality report ----------------------------------------------------
    report = {
        "bpm": round(bpm, 2),
        "sample_rate": sr,
        "first_beat_offset_s": round(first_beat_offset, 2),
        "quantization": {
            "max_error_ms": round(max(abs(ev["error_ms"]) for ev in events), 2),
            "events": [
                {
                    "index": ev["index"],
                    "sfx": ev["sfx"],
                    "grid": ev["grid"],
                    "time_s": round(ev["time"], 2),
                    "quantized_s": round(ev["quantized"], 2),
                    "error_ms": round(ev["error_ms"], 2),
                }
                for ev in events
            ],
        },
        "ducking": {
            "duck_gain_db": round(duck_gain_db, 2),
            "max_slope_db_per_s": round(envelope_max_slope_db_per_s(duck_env, sr), 2),
        },
        "mix": {"peak_dbfs": round(float(peak_dbfs), 2)},
        "loop": {
            "count": max(loop_counts.values()),
            "per_layer": {k: int(v) for k, v in sorted(loop_counts.items())},
            "bar_beats": 4,
        },
        "output": {
            "duration_s": round(total_s, 2),
            "channels": 2,
            "sample_rate": sr,
        },
    }

    # --- atomic output (no partial files on failure) -----------------------
    for path in (output_path, report_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic_write_wav(output_path, mix, sr)
    _atomic_write_json(report_path, report)
    return report
