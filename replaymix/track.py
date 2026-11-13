"""Per-track validation, resampling, slicing, and per-segment gain."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (
    E_SAMPLE_RATE_UNSUPPORTED,
    E_TRACK_EXCEEDS_SESSION,
    E_TRACK_FILE_MISSING,
    E_TRACK_TIMELINE_MISMATCH,
    E_TRACK_WAV_INVALID,
    TrackError,
    add_error,
    error,
)
from .gain import GainResult, adaptive_gain
from .resample import (
    RESAMPLER_VERSION,
    fit_to_target,
    natural_output_frames,
    resample_linear_to_48k,
)
from .segments import SegmentSpec, build_segments
from .timeline import (
    INTERNAL_SAMPLE_RATE,
    EventTable,
    TrackSpec,
    ms_to_frames,
    supported_sample_rate,
)
from .wavio import WavInfo, WavReadError, inspect_wav, read_wav_mono_float64


@dataclass
class SegmentAudio:
    start_frame: int
    audio: np.ndarray


@dataclass
class TrackResult:
    track_id: Any
    spec: TrackSpec
    status: str
    errors: list[TrackError] = field(default_factory=list)
    actual: dict[str, Any] = field(default_factory=dict)
    resample: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    active_audio: list[SegmentAudio] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    input_signature: str = ""
    from_checkpoint: bool = False

    @property
    def error_codes(self) -> list[str]:
        return list(dict.fromkeys(item.code for item in self.errors))

    def record(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "player": self.spec.player,
            "source_file": self.spec.file_display,
            "status": self.status,
            "error_codes": self.error_codes,
            "errors": [item.as_dict() for item in self.errors],
            "declared": {
                "sample_rate_hz": self.spec.sample_rate_hz,
                "channels": self.spec.channels,
                "frames": self.spec.declared_frames,
                "write_start_ms": self.spec.write_start_ms,
                "write_end_ms": self.spec.write_end_ms,
                "silence_ms": [list(item) for item in self.spec.silence],
                "gaps_ms": [list(item) for item in self.spec.gaps],
            },
            "actual": self.actual,
            "resample": self.resample,
            "segments": self.segments,
            "zero_fill": self.counts,
            "event_markers": self.spec.markers,
            "from_checkpoint": self.from_checkpoint,
        }


def _marker_time(marker: dict[str, Any]) -> float | None:
    for key in ("at_ms", "time_ms", "t_ms", "time", "at"):
        value = marker.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _markers_in_segment(
    markers: list[dict[str, Any]], start_ms: float, end_ms: float
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for marker in markers:
        at = _marker_time(marker)
        if at is not None and start_ms <= at < end_ms:
            matched.append(marker)
    return matched


def _file_sha256_or_missing(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_track_signature(
    *,
    spec: TrackSpec,
    table: EventTable,
    target_peak_dbfs: float,
    max_gain_db: float,
    checkpoint_version: int,
) -> str:
    payload = {
        "checkpoint_version": checkpoint_version,
        "resampler": RESAMPLER_VERSION,
        "session_ms": table.session_ms,
        "session_frames": table.total_frames,
        "target_peak_dbfs": target_peak_dbfs,
        "max_gain_db": max_gain_db,
        "track": spec.canonical(),
        "file_sha256": _file_sha256_or_missing(spec.file_path),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalid_result(
    spec: TrackSpec,
    status: str,
    errors: list[TrackError],
    *,
    actual: dict[str, Any] | None = None,
    input_signature: str = "",
) -> TrackResult:
    return TrackResult(
        track_id=spec.track_id,
        spec=spec,
        status=status,
        errors=errors,
        actual=actual or {},
        counts={},
        input_signature=input_signature,
    )


def _validate_audio(
    spec: TrackSpec, table: EventTable
) -> tuple[WavInfo | None, list[TrackError], bool, bool]:
    errors: list[TrackError] = list(spec.spec_errors)
    if spec.file_path is None:
        add_error(errors, E_TRACK_FILE_MISSING, "track file path is missing")
        return None, errors, True, True
    if not spec.file_path.is_file():
        add_error(
            errors,
            E_TRACK_FILE_MISSING,
            f"track file does not exist: {spec.file_display}",
        )
        return None, errors, True, True

    try:
        info = inspect_wav(spec.file_path)
    except WavReadError as exc:
        add_error(errors, E_TRACK_WAV_INVALID, str(exc))
        return None, errors, True, bool(spec.spec_errors)

    fatal = bool(spec.spec_errors)
    if info.compression != "NONE":
        add_error(
            errors,
            E_TRACK_WAV_INVALID,
            f"unsupported WAV compression: {info.compression}",
        )
        fatal = True
    if info.sample_width_bytes not in (1, 2, 3, 4):
        add_error(
            errors,
            E_TRACK_WAV_INVALID,
            f"unsupported PCM sample width: {info.sample_width_bytes}",
        )
        fatal = True

    declared_rate = spec.sample_rate_hz
    declared_channels = spec.channels
    if not supported_sample_rate(declared_rate) or not supported_sample_rate(
        info.sample_rate_hz
    ):
        add_error(
            errors,
            E_SAMPLE_RATE_UNSUPPORTED,
            (
                f"sample rate declared={declared_rate}, actual={info.sample_rate_hz}; "
                "supported range is 8000..48000 Hz"
            ),
        )
        fatal = True
    if declared_rate is not None and info.sample_rate_hz != declared_rate:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            (
                f"sample rate mismatch: event table declares {declared_rate} Hz, "
                f"WAV contains {info.sample_rate_hz} Hz"
            ),
        )
        fatal = True
    if declared_channels != 1 or info.channels != 1:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            (
                f"channel mismatch: event table declares {declared_channels}, "
                f"WAV contains {info.channels}; mono is required"
            ),
        )
        fatal = True
    if spec.write_start_ms < 0:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            f"negative write_start_ms: {spec.write_start_ms}",
        )
        fatal = True
    if spec.write_end_ms <= spec.write_start_ms:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            "write_end_ms must be greater than write_start_ms",
        )
        fatal = True
    if spec.write_start_ms >= table.session_ms:
        add_error(
            errors,
            E_TRACK_EXCEEDS_SESSION,
            "track starts at or after the end of the session",
        )
        fatal = True
    if spec.write_end_ms > table.session_ms:
        add_error(
            errors,
            E_TRACK_EXCEEDS_SESSION,
            (
                f"track write_end_ms={spec.write_end_ms} exceeds "
                f"session_ms={table.session_ms}; out-of-session samples are clipped"
            ),
        )

    if spec.declared_frames is not None and info.frames != spec.declared_frames:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            (
                f"frame-count mismatch: event table declares {spec.declared_frames}, "
                f"WAV contains {info.frames}"
            ),
        )
        fatal = True

    if declared_rate is not None and spec.write_end_ms > spec.write_start_ms:
        expected_frames = int(
            round((spec.write_end_ms - spec.write_start_ms) * declared_rate / 1000.0)
        )
        if abs(info.frames - expected_frames) > 1:
            add_error(
                errors,
                E_TRACK_TIMELINE_MISMATCH,
                (
                    f"WAV duration does not match event interval: "
                    f"{info.frames} frames at {info.sample_rate_hz} Hz vs "
                    f"{expected_frames} expected frames"
                ),
            )
            fatal = True

    actual_duration_ms = info.frames * 1000.0 / info.sample_rate_hz
    if actual_duration_ms > table.session_ms + (1000.0 / info.sample_rate_hz):
        add_error(
            errors,
            E_TRACK_EXCEEDS_SESSION,
            (
                f"WAV duration {actual_duration_ms:.3f} ms exceeds "
                f"session {table.session_ms:.3f} ms"
            ),
        )
    return info, errors, fatal, False


def _extract_active_audio(
    resampled: np.ndarray, segment: SegmentSpec, write_start_frame: int
) -> np.ndarray:
    expected = segment.frames
    start_index = segment.start_frame - write_start_frame
    end_index = segment.end_frame - write_start_frame
    if start_index < 0 or end_index <= start_index:
        return np.zeros(max(0, expected), dtype=np.float64)
    available_start = min(max(start_index, 0), resampled.size)
    available_end = min(max(end_index, 0), resampled.size)
    chunk = resampled[available_start:available_end].copy()
    if chunk.size < expected:
        chunk = np.concatenate(
            [chunk, np.zeros(expected - chunk.size, dtype=np.float64)]
        )
    elif chunk.size > expected:
        chunk = chunk[:expected]
    return chunk


def process_track(
    *,
    spec: TrackSpec,
    table: EventTable,
    target_peak_dbfs: float,
    max_gain_db: float,
    input_signature: str = "",
) -> TrackResult:
    info, errors, fatal, file_missing = _validate_audio(spec, table)
    actual: dict[str, Any] = {}
    if info is not None:
        actual = {
            "sample_rate_hz": info.sample_rate_hz,
            "channels": info.channels,
            "frames": info.frames,
            "duration_ms": info.frames * 1000.0 / info.sample_rate_hz,
            "sample_width_bytes": info.sample_width_bytes,
            "compression": info.compression,
        }
    if fatal:
        status = "skipped" if file_missing else "skipped"
        return _invalid_result(
            spec, status, errors, actual=actual, input_signature=input_signature
        )
    assert info is not None
    assert spec.sample_rate_hz is not None

    try:
        source, read_info = read_wav_mono_float64(spec.file_path)
    except WavReadError as exc:
        add_error(errors, E_TRACK_WAV_INVALID, str(exc))
        return _invalid_result(
            spec, "skipped", errors, actual=actual, input_signature=input_signature
        )
    if read_info.frames != info.frames:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            "WAV frame count changed while reading",
        )
        return _invalid_result(
            spec, "skipped", errors, actual=actual, input_signature=input_signature
        )

    write_start_frame = ms_to_frames(spec.write_start_ms)
    write_end_frame = ms_to_frames(spec.write_end_ms)
    target_frames = write_end_frame - write_start_frame
    natural_frames = natural_output_frames(source.size, spec.sample_rate_hz)
    frame_delta = natural_frames - target_frames
    if abs(frame_delta) > 1:
        add_error(
            errors,
            E_TRACK_TIMELINE_MISMATCH,
            (
                f"resampled frame delta {frame_delta} exceeds one frame "
                f"(natural={natural_frames}, target={target_frames})"
            ),
        )
        return _invalid_result(
            spec, "skipped", errors, actual=actual, input_signature=input_signature
        )

    resampled_natural = resample_linear_to_48k(source, spec.sample_rate_hz)
    resampled, fit = fit_to_target(resampled_natural, target_frames)
    resample_record = {
        "algorithm": RESAMPLER_VERSION,
        "source_sample_rate_hz": spec.sample_rate_hz,
        "target_sample_rate_hz": INTERNAL_SAMPLE_RATE,
        "source_frames": int(source.size),
        "natural_frames": fit.natural_frames,
        "expected_frames": fit.target_frames,
        "output_frames": int(resampled.size),
        "frame_delta": fit.frame_delta,
        "zero_padded_frames": fit.zero_padded_frames,
        "trimmed_frames": fit.trimmed_frames,
        "within_one_frame": abs(fit.frame_delta) <= 1,
    }

    segment_specs = build_segments(
        session_ms=table.session_ms,
        write_start_ms=spec.write_start_ms,
        write_end_ms=spec.write_end_ms,
        silence_ms=spec.silence,
        gaps_ms=spec.gaps,
    )
    segment_records: list[dict[str, Any]] = []
    active_audio: list[SegmentAudio] = []
    counts = {
        "active_frames": 0,
        "silence_fill_frames": 0,
        "gap_fill_frames": 0,
        "pre_roll_frames": 0,
        "post_roll_frames": 0,
        "resample_zero_pad_frames": fit.zero_padded_frames,
        "resample_trimmed_frames": fit.trimmed_frames,
        "total_zero_fill_frames": 0,
    }

    for index, segment in enumerate(segment_specs):
        if segment.kind == "active":
            raw = _extract_active_audio(resampled, segment, write_start_frame)
            audio, gain = adaptive_gain(
                raw,
                target_peak_dbfs=target_peak_dbfs,
                max_gain_db=max_gain_db,
            )
            active_audio.append(SegmentAudio(segment.start_frame, audio))
            segment_records.append(
                {
                    "index": index,
                    "kind": "active",
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "frames": segment.frames,
                    "gain_linear": gain.gain_linear,
                    "gain_db": gain.gain_db,
                    "input_peak_linear": gain.input_peak_linear,
                    "input_peak_dbfs": gain.input_peak_dbfs,
                    "output_peak_linear": gain.output_peak_linear,
                    "output_peak_dbfs": gain.output_peak_dbfs,
                    "input_peak_above_target": gain.input_peak_above_target,
                    "zero_fill_frames": 0,
                    "markers": _markers_in_segment(
                        spec.markers, segment.start_ms, segment.end_ms
                    ),
                }
            )
            counts["active_frames"] += segment.frames
        else:
            segment_records.append(
                {
                    "index": index,
                    "kind": segment.kind,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "frames": segment.frames,
                    "gain_linear": 1.0,
                    "gain_db": 0.0,
                    "input_peak_linear": 0.0,
                    "input_peak_dbfs": None,
                    "output_peak_linear": 0.0,
                    "output_peak_dbfs": None,
                    "input_peak_above_target": False,
                    "zero_fill_frames": segment.frames,
                    "markers": _markers_in_segment(
                        spec.markers, segment.start_ms, segment.end_ms
                    ),
                }
            )
            if segment.kind == "silence":
                counts["silence_fill_frames"] += segment.frames
            elif segment.kind == "gap":
                counts["gap_fill_frames"] += segment.frames
            elif segment.kind == "pre_roll":
                counts["pre_roll_frames"] += segment.frames
            elif segment.kind == "post_roll":
                counts["post_roll_frames"] += segment.frames

    counts["total_zero_fill_frames"] = (
        counts["silence_fill_frames"]
        + counts["gap_fill_frames"]
        + counts["pre_roll_frames"]
        + counts["post_roll_frames"]
        + counts["resample_zero_pad_frames"]
    )
    status = "processed_with_errors" if errors else "processed"
    return TrackResult(
        track_id=spec.track_id,
        spec=spec,
        status=status,
        errors=errors,
        actual=actual,
        resample=resample_record,
        segments=segment_records,
        active_audio=active_audio,
        counts=counts,
        input_signature=input_signature,
    )
