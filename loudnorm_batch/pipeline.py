"""Per-file processing pipeline."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import bs1770, decode, dsp, probe, truepeak

OUT_RATE = 48_000
OUT_BIT_DEPTH = 24
OUT_CHANNELS = 2
TIMESTAMP_TOLERANCE_MS = 5.0


@dataclass
class Params:
    target_lufs: float = -23.0
    ceiling_dbtp: float = -1.0
    gain_min_db: float = -24.0
    gain_max_db: float = 24.0
    total_ms: float = 0.0
    out_dir: str = "out"
    timeout_s: float = 30.0


@dataclass
class FileResult:
    index: int
    path: str
    offset_ms: float
    status: str = "ok"
    flags: list[str] = field(default_factory=list)
    metadata: dict | None = None
    measured: dict = field(default_factory=dict)
    processing: dict = field(default_factory=dict)
    output: dict | None = None
    error: dict | None = None


def sha256_16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stderr_excerpt(text: str, limit: int = 400) -> str:
    # normalize ASLR addresses so reports are byte-reproducible
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    return text.strip()[-limit:]


def probe_file(index: int, entry: dict, params: Params) -> FileResult:
    """Phase 1: hashing + ffprobe metadata only."""
    path = entry["path"]
    result = FileResult(index=index, path=path, offset_ms=float(entry.get("offset_ms", 0.0)))
    try:
        result.processing["sha256_16"] = sha256_16(path)
    except OSError as exc:
        result.status = "unreadable"
        result.error = {"stage": "hash", "stderr_excerpt": str(exc)}
        return result
    try:
        result.metadata = probe.probe(path, timeout=params.timeout_s)
    except probe.ProbeError as exc:
        result.status = "unreadable"
        result.error = {"stage": "probe", "stderr_excerpt": _stderr_excerpt(exc.stderr or str(exc))}
    return result


def process_file(result: FileResult, params: Params) -> FileResult:
    """Phase 2: decode, measure, normalize, limit, align, write."""
    if result.status != "ok":
        return result
    path = result.path
    meta = result.metadata
    channels = meta["channels"]
    src_rate = meta["sample_rate"]
    layout = meta["channel_layout"]

    # --- measurement pass (native rate) -------------------------------
    try:
        pcm_native, _, stderr_native, rc_native = decode.decode(
            path, channels, sample_rate=None, timeout=params.timeout_s
        )
    except decode.DecodeError as exc:
        result.status = "decode_error"
        result.error = {"stage": "decode", "stderr_excerpt": _stderr_excerpt(exc.stderr)}
        return result
    decoded_s = pcm_native.shape[0] / src_rate if src_rate else 0.0
    probed_s = meta.get("duration_s") or 0.0
    shortfall = probed_s - decoded_s
    stderr_l = stderr_native.lower()
    if (
        rc_native != 0
        or "error" in stderr_l
        or "invalid" in stderr_l
        or shortfall > 0.1
    ):
        result.flags.append("partial_decode")
        result.processing["decode_stderr_excerpt"] = _stderr_excerpt(stderr_native)
        result.processing["decoded_duration_s"] = round(decoded_s, 6)

    names = dsp.layout_channels(layout, channels)
    weights = [bs1770.channel_weight(n) for n in names]
    loudness = bs1770.integrated_loudness(pcm_native, src_rate, weights)
    in_tp = truepeak.true_peak_dbfs(pcm_native, src_rate)
    result.measured = {
        "integrated_lufs": _r2(loudness),
        "true_peak_dbtp": _r2(in_tp),
    }

    # --- gain staging ---------------------------------------------------
    gain_db = params.target_lufs - loudness if np.isfinite(loudness) else float("inf")
    result.processing.update(
        {
            "target_lufs": params.target_lufs,
            "computed_gain_db": _r2(gain_db),
            "alignment_offset_ms": result.offset_ms,
            "file_start_time_ms": meta["start_time_ms"],
        }
    )
    if abs(meta["start_time_ms"] - result.offset_ms) > TIMESTAMP_TOLERANCE_MS:
        result.flags.append("timestamp_mismatch")

    if not np.isfinite(gain_db) or gain_db < params.gain_min_db or gain_db > params.gain_max_db:
        result.status = "loudness_out_of_range"
        result.error = {
            "stage": "normalize",
            "reason": (
                f"required gain {_r2(gain_db)} dB outside "
                f"[{params.gain_min_db}, {params.gain_max_db}] dB"
            ),
        }
        return result

    # --- output-rate decode + downmix -----------------------------------
    try:
        pcm48, _, _, _ = decode.decode(
            path, channels, sample_rate=OUT_RATE, timeout=params.timeout_s
        )
    except decode.DecodeError as exc:
        result.status = "decode_error"
        result.error = {"stage": "decode", "stderr_excerpt": _stderr_excerpt(exc.stderr)}
        return result
    stereo = dsp.downmix_stereo(pcm48, layout)

    gained = dsp.apply_gain(stereo, gain_db)
    limited, reduction_db = dsp.limit(gained, OUT_RATE, params.ceiling_dbtp)
    out_tp = truepeak.true_peak_dbfs(limited, OUT_RATE)
    # safety trim so the reported true peak never exceeds the ceiling
    if out_tp > params.ceiling_dbtp:
        trim_db = params.ceiling_dbtp - out_tp
        limited = dsp.apply_gain(limited, trim_db)
        reduction_db += -trim_db
        out_tp = truepeak.true_peak_dbfs(limited, OUT_RATE)
    result.processing["applied_gain_db"] = _r2(gain_db)
    result.processing["limiter_reduction_db"] = _r2(reduction_db)

    # --- alignment + length check ---------------------------------------
    aligned = dsp.align(limited, OUT_RATE, result.offset_ms, params.total_ms)
    expected = int(round(params.total_ms * OUT_RATE / 1000.0))
    if aligned.shape[0] != expected:
        result.status = "length_mismatch"
        result.error = {
            "stage": "align",
            "reason": f"expected {expected} samples, got {aligned.shape[0]}",
        }
        return result

    os.makedirs(params.out_dir, exist_ok=True)
    out_name = f"{result.index:03d}_{os.path.splitext(os.path.basename(path))[0]}.wav"
    out_path = os.path.join(params.out_dir, out_name)
    sf.write(out_path, aligned, OUT_RATE, subtype="PCM_24")

    result.output = {
        "path": out_path,
        "format": "WAV/PCM",
        "sample_rate": OUT_RATE,
        "bit_depth": OUT_BIT_DEPTH,
        "channels": OUT_CHANNELS,
        "samples": int(aligned.shape[0]),
        "integrated_lufs": _r2(
            bs1770.integrated_loudness(aligned, OUT_RATE, [1.0, 1.0])
        ),
        "true_peak_dbtp": _r2(out_tp),
    }
    return result


def _r2(value: float):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 2)
