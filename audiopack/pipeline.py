"""End-to-end orchestration of the offline audio packaging pipeline."""

from __future__ import annotations

import os

import numpy as np

from .audioio import write_wav
from .errors import (
    E_CORRUPT_CLIP,
    E_MEMORY_LIMIT,
    E_TIMELINE_FORMAT,
    PipelineError,
)
from .loudness import normalize
from .memoryutil import DEFAULT_MEMORY_LIMIT_BYTES, MemoryGuard
from .renderer import render_events
from .report import build_report, write_report
from .timeline import load_timeline


def run_pipeline(
    input_dir: str,
    output_wav: str,
    report_path: str | None = None,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> tuple[int, dict, list[dict]]:
    """Execute the full pipeline.

    Returns ``(exit_code, report, errors)``.  Fatal validation / memory errors
    are raised as :class:`PipelineError` by the caller-facing CLI; per-clip
    ``E_CORRUPT_CLIP`` failures are collected so other events still render.
    """
    if not os.path.isdir(input_dir):
        raise PipelineError(
            E_TIMELINE_FORMAT, {"input_dir": input_dir, "reason": "input directory not found"}
        )

    wav_files = sorted(
        name for name in os.listdir(input_dir) if name.lower().endswith(".wav")
    )
    if not wav_files:
        raise PipelineError(
            E_TIMELINE_FORMAT,
            {"input_dir": input_dir, "reason": "input directory contains no .wav files"},
        )

    events, sample_rate = load_timeline(input_dir)
    guard = MemoryGuard(memory_limit_bytes)

    mix, event_reports, overlap_records, warning_details, errors = render_events(
        events, input_dir, sample_rate, guard
    )

    normalized, loudness_metrics, loudness_warnings = normalize(mix, sample_rate)
    warning_codes = list(loudness_warnings)

    report = build_report(
        sample_rate=sample_rate,
        output_frames=int(normalized.shape[0]),
        event_reports=event_reports,
        overlap_records=overlap_records,
        loudness=loudness_metrics,
        warnings=warning_codes,
        warning_details=warning_details,
        errors=errors,
    )

    _write_outputs(output_wav, report_path, normalized, sample_rate, report)

    has_error = any(item.get("code") == E_CORRUPT_CLIP for item in errors)
    return (1 if has_error else 0), report, errors


def _write_outputs(
    output_wav: str,
    report_path: str | None,
    audio: np.ndarray,
    sample_rate: int,
    report: dict,
) -> None:
    """Write WAV then report; remove partial outputs on any failure."""
    if report_path is None:
        report_path = os.path.splitext(output_wav)[0] + "_report.json"

    created: list[str] = []
    try:
        write_wav(output_wav, audio, sample_rate)
        created.append(output_wav)
        write_report(report_path, report)
    except PipelineError:
        _cleanup(created)
        raise
    except (OSError, MemoryError, ValueError) as exc:
        _cleanup(created)
        raise PipelineError(
            E_MEMORY_LIMIT if isinstance(exc, MemoryError) else E_CORRUPT_CLIP,
            {"reason": f"failed to write outputs: {exc}"},
        ) from exc


def _cleanup(paths: list[str]) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
