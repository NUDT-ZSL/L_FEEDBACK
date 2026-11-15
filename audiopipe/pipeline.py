"""Concurrent segment processing, ordered concatenation, resume support."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

from . import channelmap, loudness, resample, wavio
from .errors import Code, PipelineError
from .manifest import Manifest, SegmentSpec

OUTPUT_SAMPLE_RATE = 48000
OUTPUT_BITS = 24
OUTPUT_CHANNELS = 2
PEAK_LIMIT_DB = -1.0          # post-gain sample-peak ceiling
REPORT_NAME = "report.json"
OUTPUT_NAME = "output.wav"
STATE_DIRNAME = ".state"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"


def _state_path(state_dir: str, index: int) -> str:
    return os.path.join(state_dir, f"seg_{index:04d}.npy")


def _atomic_json(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _apply_fades(x: np.ndarray, fade_in_ms: float, fade_out_ms: float,
                 warnings: List[str]) -> np.ndarray:
    n = x.shape[0]
    for ms, is_in in ((fade_in_ms, True), (fade_out_ms, False)):
        if ms <= 0:
            continue
        length = int(round(ms * OUTPUT_SAMPLE_RATE / 1000.0))
        if length > n // 2:
            length = n // 2
            warnings.append("FADE_TRUNCATED")
        if length <= 0:
            continue
        t = (np.arange(length, dtype=np.float64) + 0.5) / length
        ramp = 0.5 - 0.5 * np.cos(np.pi * t)  # raised cosine 0 -> 1
        if is_in:
            x[:length] *= ramp[:, None]
        else:
            x[n - length:] *= ramp[::-1][:, None]
    return x


def process_segment(spec: SegmentSpec, state_dir: str) -> dict:
    """Decode, normalise and render one segment to 48 kHz stereo float64.

    Never raises for business errors; returns a report entry dict.
    """
    entry = {
        "index": spec.index,
        "source": spec.source,
        "start_sec": spec.start_sec,
        "end_sec": spec.end_sec,
        "target_loudness_lkfs": spec.target_lkfs,
        "integrated_loudness_lkfs": None,
        "applied_gain_db": None,
        "fade_in_ms": spec.fade_in_ms,
        "fade_out_ms": spec.fade_out_ms,
        "xfade_in_ms": None,
        "xfade_out_ms": None,
        "offset_sec": None,
        "duration_sec": None,
        "source_sample_rate": None,
        "source_channels": None,
        "status": STATUS_FAILED,
        "error_code": Code.OK,
        "error_message": "",
        "warnings": [],
    }
    if spec.error_code != Code.OK:
        entry["error_code"] = spec.error_code
        entry["error_message"] = spec.error_message
        return entry
    try:
        samples, rate, channels = wavio.read_wav(spec.source)
        entry["source_sample_rate"] = rate
        entry["source_channels"] = channels
        duration = samples.shape[0] / rate
        if spec.end_sec > duration + 1e-6:
            raise PipelineError(
                Code.E_INVALID_RANGE,
                f"end_sec {spec.end_sec} exceeds file duration {duration:.3f}s")
        i0 = int(round(spec.start_sec * rate))
        i1 = int(round(spec.end_sec * rate))
        seg = samples[i0:i1]
        if seg.shape[0] == 0:
            raise PipelineError(Code.E_INVALID_RANGE, "empty effective range")

        seg = resample.resample(seg, rate, OUTPUT_SAMPLE_RATE)

        weights = loudness.CHANNEL_WEIGHTS.get(channels)
        if weights is None:
            raise PipelineError(Code.E_UNSUPPORTED_LAYOUT,
                                f"channel count {channels} not supported (1/2/6)")
        loud = loudness.integrated_loudness(seg, weights)
        if loud == float("-inf"):
            gain_db = 0.0
            entry["warnings"].append("SILENT_SEGMENT")
        else:
            gain_db = spec.target_lkfs - loud
            entry["integrated_loudness_lkfs"] = round(loud, 3)
        seg = seg * (10.0 ** (gain_db / 20.0))

        stereo = channelmap.to_stereo(seg)

        peak = float(np.max(np.abs(stereo))) if stereo.shape[0] else 0.0
        limit = 10.0 ** (PEAK_LIMIT_DB / 20.0)
        if peak > limit:
            scale = limit / peak
            stereo = stereo * scale
            gain_db += 20.0 * np.log10(scale)
            entry["warnings"].append("PEAK_LIMITED")
        entry["applied_gain_db"] = round(float(gain_db), 3)

        stereo = _apply_fades(stereo, spec.fade_in_ms, spec.fade_out_ms,
                              entry["warnings"])

        os.makedirs(state_dir, exist_ok=True)
        np.save(_state_path(state_dir, spec.index), stereo)
        entry["duration_sec"] = round(stereo.shape[0] / OUTPUT_SAMPLE_RATE, 6)
        entry["status"] = STATUS_SUCCESS
    except PipelineError as exc:
        entry["error_code"] = exc.code
        entry["error_message"] = exc.message
    except Exception as exc:  # keep other segments alive
        entry["error_code"] = Code.E_INTERNAL
        entry["error_message"] = f"{type(exc).__name__}: {exc}"
    return entry


def _plan_crossfades(lengths: List[int], xfade_samples: int,
                     entries: List[dict]) -> List[int]:
    """Conservative per-junction crossfade lengths (samples)."""
    count = len(lengths) - 1
    if count <= 0:
        return []
    ds = [min(xfade_samples, lengths[i], lengths[i + 1]) for i in range(count)]
    truncated = any(d < xfade_samples for d in ds)
    for k in range(1, len(lengths) - 1):
        total = ds[k - 1] + ds[k]
        if total > lengths[k]:
            scale = lengths[k] / total
            ds[k - 1] = int(ds[k - 1] * scale)
            ds[k] = int(ds[k] * scale)
            truncated = True
    if truncated:
        for e in entries:
            e["warnings"].append("XFADE_TRUNCATED")
    return ds


def _concatenate(pieces: List[np.ndarray], ds: List[int]) -> np.ndarray:
    out = pieces[0].copy()
    for i, piece in enumerate(pieces[1:]):
        d = ds[i]
        if d > 0:
            t = (np.arange(d, dtype=np.float64) + 0.5) / d
            w_out = np.cos(0.5 * np.pi * t)[:, None]   # equal-power
            w_in = np.sin(0.5 * np.pi * t)[:, None]
            out[-d:] = out[-d:] * w_out + piece[:d] * w_in
            out = np.vstack([out, piece[d:]])
        else:
            out = np.vstack([out, piece])
    return out


def _load_previous(report_path: str, manifest: Manifest) -> Dict[int, dict]:
    """Reusable successful entries from a previous in-progress run."""
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        return {}
    if old.get("manifest_sha256") != manifest.sha256:
        return {}
    if old.get("status") not in (STATUS_IN_PROGRESS, STATUS_COMPLETED,
                                 STATUS_COMPLETED_WITH_ERRORS):
        return {}
    return {e["index"]: e for e in old.get("segments", [])
            if e.get("status") == STATUS_SUCCESS}


def run(manifest: Manifest, output_dir: str, jobs: int = 4,
        crossfade_ms: Optional[float] = None) -> int:
    """Execute the pipeline. Returns process exit code 0/1/2."""
    started = time.monotonic()
    os.makedirs(output_dir, exist_ok=True)
    state_dir = os.path.join(output_dir, STATE_DIRNAME)
    report_path = os.path.join(output_dir, REPORT_NAME)
    output_path = os.path.join(output_dir, OUTPUT_NAME)
    xfade_ms = manifest.crossfade_ms if crossfade_ms is None else crossfade_ms
    xfade_samples = int(round(xfade_ms * OUTPUT_SAMPLE_RATE / 1000.0))

    reusable = _load_previous(report_path, manifest)

    entries: List[Optional[dict]] = [None] * len(manifest.segments)
    pending: List[SegmentSpec] = []
    reused = 0
    for spec in manifest.segments:
        prev = reusable.get(spec.index)
        state_file = _state_path(state_dir, spec.index)
        if prev is not None and os.path.isfile(state_file):
            entries[spec.index] = dict(prev)
            entries[spec.index]["warnings"] = list(prev.get("warnings", []))
            reused += 1
        else:
            pending.append(spec)

    report = {
        "version": 1,
        "status": STATUS_IN_PROGRESS,
        "manifest_path": manifest.path,
        "manifest_sha256": manifest.sha256,
        "params": {"jobs": jobs, "crossfade_ms": xfade_ms,
                   "peak_limit_db": PEAK_LIMIT_DB,
                   "loudness_standard": "ITU-R BS.1770-4",
                   "downmix_standard": "ITU-R BS.775"},
        "output": {"path": os.path.abspath(output_path),
                   "sample_rate": OUTPUT_SAMPLE_RATE,
                   "bits": OUTPUT_BITS, "channels": OUTPUT_CHANNELS},
        "segments": [],
        "summary": {},
    }

    def snapshot():
        report["segments"] = sorted(
            (e for e in entries if e is not None), key=lambda e: e["index"])
        _atomic_json(report_path, report)

    if pending:
        workers = max(1, int(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_segment, spec, state_dir): spec
                       for spec in pending}
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    entry = fut.result()
                except Exception as exc:  # defensive; process_segment catches
                    entry = {"index": spec.index, "source": spec.source,
                             "status": STATUS_FAILED,
                             "error_code": Code.E_INTERNAL,
                             "error_message": f"{type(exc).__name__}: {exc}",
                             "warnings": []}
                entries[spec.index] = entry
                snapshot()

    ok = [e for e in entries if e is not None and e["status"] == STATUS_SUCCESS]
    failed = [e for e in entries
              if e is not None and e["status"] == STATUS_FAILED]

    if not ok:
        report["status"] = STATUS_COMPLETED_WITH_ERRORS
        report["summary"] = {
            "total": len(entries), "succeeded": 0, "failed": len(failed),
            "reused": reused,
            "elapsed_sec": round(time.monotonic() - started, 3),
        }
        snapshot()
        print(f"汇总: 成功 0 / 失败 {len(failed)} / 无有效输出")
        return 2

    # Ordered concatenation of the successful segments only.
    ok.sort(key=lambda e: e["index"])
    pieces = [np.load(_state_path(state_dir, e["index"])) for e in ok]
    lengths = [p.shape[0] for p in pieces]
    ds = _plan_crossfades(lengths, xfade_samples, ok)

    offset = 0
    for i, e in enumerate(ok):
        e["offset_sec"] = round(offset / OUTPUT_SAMPLE_RATE, 6)
        e["xfade_in_ms"] = round(ds[i - 1] / OUTPUT_SAMPLE_RATE * 1000.0, 3) if i > 0 else 0.0
        e["xfade_out_ms"] = round(ds[i] / OUTPUT_SAMPLE_RATE * 1000.0, 3) if i < len(ds) else 0.0
        offset += lengths[i] - (ds[i] if i < len(ds) else 0)

    mixed = _concatenate(pieces, ds)
    wavio.write_wav(output_path, mixed, OUTPUT_SAMPLE_RATE, OUTPUT_BITS)

    elapsed = round(time.monotonic() - started, 3)
    report["status"] = (STATUS_COMPLETED if not failed
                        else STATUS_COMPLETED_WITH_ERRORS)
    report["summary"] = {
        "total": len(entries),
        "succeeded": len(ok),
        "failed": len(failed),
        "reused": reused,
        "elapsed_sec": elapsed,
        "output_duration_sec": round(mixed.shape[0] / OUTPUT_SAMPLE_RATE, 6),
    }
    snapshot()

    print(f"汇总: 成功 {len(ok)} / 失败 {len(failed)} / 复用 {reused} / "
          f"耗时 {elapsed}s")
    print(f"输出: {os.path.abspath(output_path)}")
    print(f"报告: {os.path.abspath(report_path)}")
    return 0 if not failed else 1
