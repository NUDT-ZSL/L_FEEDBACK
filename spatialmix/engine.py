"""End-to-end project processing for the spatial mixer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .audio import AUDIO_SAMPLE_RATE, fit_to_duration, read_wav_mono, resample_to_rate
from .config import ProjectConfig, load_project
from .errors import ToolError
from .spatial import compute_spatial_gains
from .timeline import (
    PreparedEvent,
    ScheduledEvent,
    samples_to_milliseconds,
    schedule_events,
    seconds_to_samples,
)

MAX_NORMALIZED_PEAK = 0.98
MAX_PCM_SAMPLE = int(math.floor(MAX_NORMALIZED_PEAK * 32767.0))


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _validate_output_path(path: Path, field: str) -> Path:
    candidate = Path(path).expanduser()
    if not str(candidate):
        raise ToolError("{}: path must not be empty".format(field))
    if candidate.exists() and candidate.is_dir():
        raise ToolError("{}: path is a directory: '{}'".format(field, candidate))

    parent = candidate.parent
    if not parent.exists():
        raise ToolError(
            "{}: output directory does not exist: '{}'".format(field, parent)
        )
    if not parent.is_dir():
        raise ToolError(
            "{}: output parent is not a directory: '{}'".format(field, parent)
        )
    return candidate


def validate_paths(
    source_directory: Path,
    events_path: Path,
    mix_output: Path,
    report_output: Path,
) -> Tuple[Path, Path, Path, Path]:
    """Validate top-level paths before doing expensive audio work."""

    source_dir = Path(source_directory).expanduser()
    if not source_dir.exists():
        raise ToolError(
            "sources_dir: directory does not exist: '{}'".format(source_directory)
        )
    if not source_dir.is_dir():
        raise ToolError(
            "sources_dir: path is not a directory: '{}'".format(source_directory)
        )

    events_file = Path(events_path).expanduser()
    if not events_file.exists():
        raise ToolError("events_file: file not found: '{}'".format(events_path))
    if not events_file.is_file():
        raise ToolError("events_file: path is not a file: '{}'".format(events_path))

    mix_file = _validate_output_path(Path(mix_output), "mix_output")
    report_file = _validate_output_path(Path(report_output), "report_output")

    if _normalized_path(mix_file) == _normalized_path(report_file):
        raise ToolError("mix_output: must differ from report_output")
    if _normalized_path(mix_file) == _normalized_path(events_file):
        raise ToolError("mix_output: must differ from events_file")
    if _normalized_path(report_file) == _normalized_path(events_file):
        raise ToolError("report_output: must differ from events_file")

    return source_dir, events_file, mix_file, report_file


def prepare_events(
    config: ProjectConfig,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> List[PreparedEvent]:
    prepared: List[PreparedEvent] = []
    for event in config.events:
        field = "events[{}].filename".format(event.index)
        clip = read_wav_mono(event.source_path, field)
        resampled = resample_to_rate(
            clip.samples,
            clip.source_sample_rate,
            sample_rate,
        )
        target_length = max(1, seconds_to_samples(event.duration, sample_rate))
        fitted, fit_kind = fit_to_duration(
            resampled,
            target_length,
            event.fit_strategy,
        )
        spatial = compute_spatial_gains(
            config.listener_x,
            config.listener_y,
            event.source_x,
            event.source_y,
            min_gain=config.min_gain,
        )
        prepared.append(
            PreparedEvent(
                spec=event,
                mono=fitted,
                spatial=spatial,
                source_sample_rate=clip.source_sample_rate,
                source_frames=clip.source_frames,
                source_channels=clip.source_channels,
                fit_kind=fit_kind,
                requested_start_sample=seconds_to_samples(
                    event.trigger_time,
                    sample_rate,
                ),
                requested_length=int(fitted.shape[0]),
            )
        )
    return prepared


def _event_contribution(entry: ScheduledEvent) -> np.ndarray:
    gains = np.asarray(
        (entry.prepared.spatial.left_gain, entry.prepared.spatial.right_gain),
        dtype=np.float64,
    )
    return entry.prepared.mono[:, np.newaxis] * gains[np.newaxis, :]


def mix_events(
    scheduled_events: List[ScheduledEvent],
) -> Tuple[np.ndarray, int, float, float, float]:
    """Mix events, then apply one master gain to keep the peak at or below 0.98.

    Returns ``(pcm, frames, raw_peak, master_gain, normalized_peak)``.
    """

    active = [
        entry
        for entry in scheduled_events
        if not entry.dropped and entry.start_sample is not None
    ]
    if active:
        frames = max(int(entry.end_sample) for entry in active)
    else:
        frames = 0

    raw_mix = np.zeros((frames, 2), dtype=np.float64)
    for entry in active:
        start = int(entry.start_sample)
        contribution = _event_contribution(entry)
        raw_mix[start : start + contribution.shape[0], :] += contribution

    raw_peak = float(np.max(np.abs(raw_mix))) if frames else 0.0
    if raw_peak > MAX_NORMALIZED_PEAK:
        master_gain = MAX_NORMALIZED_PEAK / raw_peak
    else:
        master_gain = 1.0

    normalized_mix = np.zeros_like(raw_mix)
    for entry in active:
        start = int(entry.start_sample)
        contribution = _event_contribution(entry)
        normalized_mix[start : start + contribution.shape[0], :] += (
            contribution * master_gain
        )

    normalized_peak = (
        float(np.max(np.abs(normalized_mix))) if normalized_mix.size else 0.0
    )
    if normalized_peak > MAX_NORMALIZED_PEAK:
        master_gain *= MAX_NORMALIZED_PEAK / normalized_peak
        normalized_mix = np.zeros_like(raw_mix)
        for entry in active:
            start = int(entry.start_sample)
            contribution = _event_contribution(entry)
            normalized_mix[start : start + contribution.shape[0], :] += (
                contribution * master_gain
            )
        normalized_peak = (
            float(np.max(np.abs(normalized_mix))) if normalized_mix.size else 0.0
        )

    if normalized_mix.size:
        scaled = np.rint(normalized_mix * 32767.0)
        scaled = np.clip(scaled, -MAX_PCM_SAMPLE, MAX_PCM_SAMPLE)
        pcm = scaled.astype("<i2")
        normalized_peak = float(np.max(np.abs(pcm.astype(np.int32)))) / 32767.0
    else:
        pcm = np.zeros((0, 2), dtype="<i2")
        normalized_peak = 0.0

    return pcm, frames, raw_peak, master_gain, normalized_peak


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    try:
        sf.write(
            str(path),
            pcm,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
    except Exception as exc:
        raise ToolError(
            "mix_output: cannot write WAV file '{}': {}".format(path, exc)
        ) from exc


def _round_float(value: float, digits: int = 12) -> float:
    rounded = round(float(value), digits)
    if rounded == 0.0:
        return 0.0
    return rounded


def _fit_reason(fit_kind: str) -> str:
    if fit_kind == "stretched":
        return "audio time-stretched with linear interpolation to match duration"
    if fit_kind == "cropped":
        return "audio cropped to match duration"
    if fit_kind == "padded":
        return "audio padded with silence to match duration"
    return ""


def _event_report(
    entry: ScheduledEvent,
    sample_rate: int,
) -> Dict[str, Any]:
    spec = entry.spec
    prepared = entry.prepared
    requested_start_ms = samples_to_milliseconds(
        prepared.requested_start_sample,
        sample_rate,
    )
    requested_end_ms = samples_to_milliseconds(
        prepared.requested_start_sample + prepared.requested_length,
        sample_rate,
    )

    adjusted = entry.dropped or (
        entry.start_sample != prepared.requested_start_sample
    ) or prepared.fit_kind != "none"

    reasons: List[str] = []
    fit_reason = _fit_reason(prepared.fit_kind)
    if fit_reason:
        reasons.append(fit_reason)
    if entry.dropped:
        reasons.append(entry.reason or "dropped by conflict policy")
    elif entry.start_sample != prepared.requested_start_sample:
        reasons.append(entry.reason or "trigger time adjusted for conflict")
    elif entry.reason and entry.reason != "no timeline adjustment":
        reasons.append(entry.reason)
    if not reasons:
        reasons.append("no adjustment")

    if entry.dropped:
        start_ms: Optional[int] = None
        end_ms: Optional[int] = None
        start_sample: Optional[int] = None
        end_sample: Optional[int] = None
        status = "dropped"
    else:
        start_sample = int(entry.start_sample)
        end_sample = int(entry.end_sample)
        start_ms = samples_to_milliseconds(start_sample, sample_rate)
        end_ms = samples_to_milliseconds(end_sample, sample_rate)
        status = "played"

    spatial = prepared.spatial
    return {
        "adjusted": adjusted,
        "audio_fit": prepared.fit_kind,
        "conflict_policy": spec.conflict_policy,
        "conflict_with": list(entry.conflict_indices),
        "delay_ms": samples_to_milliseconds(entry.delay_samples, sample_rate),
        "delay_samples": int(entry.delay_samples),
        "dropped": bool(entry.dropped),
        "end_sample": end_sample,
        "end_time_ms": end_ms,
        "filename": spec.filename,
        "fit_strategy": spec.fit_strategy,
        "index": spec.index,
        "priority": _round_float(spec.priority),
        "reason": "; ".join(reasons),
        "requested_end_time_ms": requested_end_ms,
        "requested_start_time_ms": requested_start_ms,
        "source_channels": prepared.source_channels,
        "source_sample_rate": prepared.source_sample_rate,
        "spatial": {
            "azimuth_degrees": _round_float(spatial.azimuth_degrees, 9),
            "distance": _round_float(spatial.distance),
            "distance_gain": _round_float(spatial.distance_gain),
            "left_gain": _round_float(spatial.left_gain),
            "right_gain": _round_float(spatial.right_gain),
        },
        "start_sample": start_sample,
        "start_time_ms": start_ms,
        "status": status,
    }


def build_report(
    config: ProjectConfig,
    scheduled_events: List[ScheduledEvent],
    mix_path: Path,
    sample_rate: int,
    frames: int,
    raw_peak: float,
    master_gain: float,
    normalized_peak: float,
) -> Dict[str, Any]:
    event_reports = [
        _event_report(entry, sample_rate)
        for entry in scheduled_events
    ]
    played = sum(1 for item in event_reports if item["status"] == "played")
    dropped = sum(1 for item in event_reports if item["status"] == "dropped")
    return {
        "events": event_reports,
        "listener": {
            "x": _round_float(config.listener_x),
            "y": _round_float(config.listener_y),
        },
        "min_gain": _round_float(config.min_gain),
        "output": {
            "channels": 2,
            "duration_ms": samples_to_milliseconds(frames, sample_rate),
            "filename": mix_path.name,
            "frames": int(frames),
            "normalization_gain": _round_float(master_gain),
            "normalization_required": bool(raw_peak > MAX_NORMALIZED_PEAK),
            "peak_after_normalization": _round_float(normalized_peak),
            "peak_before_normalization": _round_float(raw_peak),
            "sample_rate": int(sample_rate),
        },
        "summary": {
            "dropped_events": dropped,
            "played_events": played,
            "total_events": len(event_reports),
        },
    }


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except Exception as exc:
        raise ToolError(
            "report_output: cannot write JSON file '{}': {}".format(path, exc)
        ) from exc


def process_project(
    source_directory: Path,
    events_path: Path,
    mix_output: Path,
    report_output: Path,
    global_policy: Optional[str] = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Dict[str, Any]:
    """Run a complete project and return the report dictionary."""

    (
        source_dir,
        events_file,
        mix_file,
        report_file,
    ) = validate_paths(source_directory, events_path, mix_output, report_output)

    config = load_project(
        source_dir,
        events_file,
        global_policy=global_policy,
    )
    event_outputs = {_normalized_path(mix_file), _normalized_path(report_file)}
    for event in config.events:
        if _normalized_path(event.source_path) in event_outputs:
            raise ToolError(
                "{}: output path would overwrite source file '{}'".format(
                    "events[{}].filename".format(event.index),
                    event.filename,
                )
            )

    prepared = prepare_events(config, sample_rate=sample_rate)
    scheduled = schedule_events(prepared)
    pcm, frames, raw_peak, master_gain, normalized_peak = mix_events(scheduled)
    report = build_report(
        config,
        scheduled,
        mix_file,
        sample_rate,
        frames,
        raw_peak,
        master_gain,
        normalized_peak,
    )

    _write_wav(mix_file, pcm, sample_rate)
    _write_report(report_file, report)
    return report
