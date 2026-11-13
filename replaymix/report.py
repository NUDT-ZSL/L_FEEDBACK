"""JSON replay-report construction and writing."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .mix import MixResult
from .timeline import INTERNAL_SAMPLE_RATE, EventTable
from .track import TrackResult

REPORT_VERSION = "1.0"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _aggregate_errors(results: Iterable[TrackResult]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for result in results:
        for code in result.error_codes:
            entry = aggregate.setdefault(
                code, {"code": code, "track_ids": [], "count": 0}
            )
            entry["track_ids"].append(result.track_id)
            entry["count"] += 1
    return [aggregate[key] for key in sorted(aggregate)]


def _aggregate_zero_fill(results: Iterable[TrackResult]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for result in results:
        for key, value in result.counts.items():
            if key.endswith("_frames"):
                totals[key] = totals.get(key, 0) + int(value)
    return {key: totals[key] for key in sorted(totals)}


def build_report(
    *,
    table: EventTable,
    results: list[TrackResult],
    mix: MixResult,
    config: Any,
    checkpoint_summary: dict[str, Any],
    wav_path: Path,
    report_path: Path,
    wav_sha256: str,
    wav_sample_rate_hz: int,
    wav_channels: int,
    effective_concurrency: int,
    max_observed_concurrency: int,
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item.spec.sort_key)
    report = {
        "report_version": REPORT_VERSION,
        "session": {
            "session_id": table.session_id,
            "event_table": str(table.path),
            "session_ms": table.session_ms,
            "internal_sample_rate_hz": INTERNAL_SAMPLE_RATE,
            "mix_frames": table.total_frames,
            "frame_formula": "round(session_ms * 48)",
        },
        "settings": {
            "target_peak_dbfs": config.target_peak_dbfs,
            "max_gain_db": config.max_gain_db,
            "requested_concurrency": config.concurrency,
            "effective_concurrency": effective_concurrency,
            "max_observed_concurrency": max_observed_concurrency,
            "checkpoint_version": config.checkpoint_version,
        },
        "mix": {
            "total_frames": mix.total_frames,
            "expected_frames": table.total_frames,
            "last_active_end_frame": mix.last_active_end_frame,
            "tail_padding_frames": mix.tail_padding_frames,
            "peak_before_clip_dbfs": (
                None
                if mix.peak_before_clip <= 0
                else 20.0 * math.log10(mix.peak_before_clip)
            ),
            "peak_after_clip_dbfs": (
                None
                if mix.peak_after_clip <= 0
                else 20.0 * math.log10(mix.peak_after_clip)
            ),
            "clipped_sample_count": mix.clipped_sample_count,
            "out_of_bounds_frames_discarded": mix.out_of_bounds_frames_discarded,
            "clip_range": [-1.0, 1.0],
        },
        "output_wav": {
            "path": str(wav_path),
            "sample_rate_hz": wav_sample_rate_hz,
            "channels": wav_channels,
            "sample_width_bytes": 2,
            "encoding": "PCM16",
            "frames": mix.total_frames,
            "sha256": wav_sha256,
        },
        "report_path": str(report_path),
        "checkpoint": checkpoint_summary,
        "checkpoint_reused": bool(checkpoint_summary.get("reused", False)),
        "tracks": [result.record() for result in ordered],
        "zero_fill": _aggregate_zero_fill(ordered),
        "events": table.events,
        "errors": _aggregate_errors(ordered),
    }
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    text = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    _atomic_write_text(path, text + "\n")
