"""Pipeline stages: synth -> align -> mix -> loudness normalize."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import dsp, synth, wavfile
from .project import Project

TIMELINE_RATE = 48000


class StageTimer:
    def __init__(self):
        self.timings: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                timer.timings[name] = round(time.perf_counter() - self.t0, 6)
                return False

        return _Ctx()


def align_events(project: Project, timer: StageTimer, out_dir):
    """Synthesize each event, resample to 48 kHz and place it on the
    unified per-track timeline. Returns ({track: mono array}, alignment)."""
    total_samples = int(math.ceil(project.timeline_duration * TIMELINE_RATE))
    tracks = {name: np.zeros(total_samples, dtype=np.float64) for name in project.tracks}
    alignment = []
    with timer("align"):
        for ev in project.events:
            raw = synth.synth_event(ev.event_id, ev.source, ev.duration)
            resampled = dsp.resample_rational(raw, 160, 147)  # 44.1k -> 48k
            offset = int(round(ev.start * TIMELINE_RATE))
            end = min(offset + resampled.size, total_samples)
            seg = resampled[:end - offset]
            tracks[ev.track][offset:end] += seg
            alignment.append({
                "event_id": ev.event_id,
                "track": ev.track,
                "start_s": ev.start,
                "sample_offset": offset,
                "offset_error_ms": (offset / TIMELINE_RATE - ev.start) * 1000.0,
                "samples": int(seg.size),
            })
    if out_dir is not None:
        align_dir = Path(out_dir) / "aligned"
        align_dir.mkdir(parents=True, exist_ok=True)
        for name, data in tracks.items():
            wavfile.write_wav24(align_dir / f"track_{name}.wav", data, TIMELINE_RATE)
    return tracks, alignment


def mix_tracks(project: Project, tracks: dict, timer: StageTimer, out_dir):
    """Apply per-track gain (dB) and sum to stereo float (no integer clipping)."""
    with timer("mix"):
        total_samples = int(math.ceil(project.timeline_duration * TIMELINE_RATE))
        mix = np.zeros((total_samples, 2), dtype=np.float64)
        track_peaks = {}
        for name, data in tracks.items():
            gain_db = float(project.tracks[name]["gain_db"])
            scaled = data * dsp.db_to_linear(gain_db)
            peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
            track_peaks[name] = {
                "gain_db": gain_db,
                "peak_linear": peak,
                "peak_dbfs": dsp.linear_to_db(peak),
            }
            mix[:, 0] += scaled
            mix[:, 1] += scaled
        mix_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if out_dir is not None:
        wavfile.write_wav24(Path(out_dir) / "mix_48k_24bit_stereo.wav", mix, TIMELINE_RATE)
    return mix, {"tracks": track_peaks,
                 "mix_peak_linear": mix_peak,
                 "mix_peak_dbfs": dsp.linear_to_db(mix_peak)}


def normalize_loudness(mix: np.ndarray, target_lufs: float, tp_ceiling_dbtp: float,
                       timer: StageTimer):
    """EBU R128 normalize to target_lufs with true-peak limiting."""
    with timer("loudness"):
        before_lufs = dsp.integrated_loudness(mix, TIMELINE_RATE)
        before_tp = dsp.true_peak_dbtp(mix)
        if math.isinf(before_lufs):
            return mix, {
                "before": {"integrated_lufs": None, "true_peak_dbtp": None},
                "after": {"integrated_lufs": None, "true_peak_dbtp": None},
                "target_lufs": target_lufs,
                "true_peak_ceiling_dbtp": tp_ceiling_dbtp,
                "gain_db": 0.0, "limiter_engaged": False,
                "note": "silence: loudness undefined, gain skipped",
            }
        ceiling = dsp.db_to_linear(tp_ceiling_dbtp)
        out = mix * dsp.db_to_linear(target_lufs - before_lufs)
        limiter_engaged = False
        lufs = None
        for _ in range(4):
            sample_peak = float(np.max(np.abs(out))) if out.size else 0.0
            if sample_peak > ceiling * 0.98 and dsp.true_peak_dbtp(out) > tp_ceiling_dbtp:
                out = dsp.peak_limit(out, ceiling * 0.999, TIMELINE_RATE)
                limiter_engaged = True
                lufs = None
            lufs = dsp.integrated_loudness(out, TIMELINE_RATE)
            trim = target_lufs - lufs
            if abs(trim) <= 0.05:
                break
            out = out * dsp.db_to_linear(trim)
            lufs = None
        after_lufs = lufs if lufs is not None else dsp.integrated_loudness(out, TIMELINE_RATE)
        after_tp = dsp.true_peak_dbtp(out)
        if after_tp > tp_ceiling_dbtp:  # final safety trim
            out = out * dsp.db_to_linear(tp_ceiling_dbtp - after_tp)
            after_lufs = dsp.integrated_loudness(out, TIMELINE_RATE)
            after_tp = dsp.true_peak_dbtp(out)
        report = {
            "before": {"integrated_lufs": round(before_lufs, 3),
                       "true_peak_dbtp": round(before_tp, 3)},
            "after": {"integrated_lufs": round(after_lufs, 3),
                      "true_peak_dbtp": round(after_tp, 3)},
            "target_lufs": target_lufs,
            "true_peak_ceiling_dbtp": tp_ceiling_dbtp,
            "gain_db": round(target_lufs - before_lufs, 3),
            "limiter_engaged": limiter_engaged,
        }
    return out, report
