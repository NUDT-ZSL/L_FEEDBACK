"""Timeline scheduling, mixing, reports and atomic output publication."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import (
    equal_power_fades,
    linear_to_db,
    quantize_pcm16,
    trim_leading_trailing_silence,
)
from .errors import (
    EXIT_AUDIO,
    EXIT_OUTPUT,
    EXIT_TIMELINE,
    BusinessError,
    warning,
)
from .loudness import (
    SAMPLE_RATE,
    integrated_loudness,
    limit_peak,
    normalize_segment,
)
from .manifest import Manifest, TargetSegment, load_manifest
from .media import (
    ToolPaths,
    decode_audio_canonical,
    find_tools,
    probe_audio,
    seconds_to_samples,
    validate_source_format,
    validate_target_format,
)


@dataclass
class PreparedClip:
    segment: TargetSegment
    audio: np.ndarray
    mapped_start_sample: int = 0
    actual_start_sample: int = 0
    actual_end_sample: int = 0
    auto_shift_samples: int = 0
    overlap_samples: int = 0
    shifted: bool = False
    truncated: bool = False
    caused_shift: bool = False
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _timeline_error(
    code: str,
    message: str,
    *,
    file: str,
    details: dict[str, Any] | None = None,
) -> BusinessError:
    return BusinessError(
        code=code,
        message=message,
        exit_code=EXIT_TIMELINE,
        file=file,
        details=details or {},
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _prepare_clips(
    manifest: Manifest,
    tools: ToolPaths,
) -> list[PreparedClip]:
    clips: list[PreparedClip] = []
    for segment in manifest.segments:
        info = probe_audio(segment.audio_path, tools)
        validate_target_format(
            info,
            segment.audio_path,
            segment.expected_sample_rate,
            segment.expected_channels,
        )
        decoded = decode_audio_canonical(segment.audio_path, tools)
        trimmed = trim_leading_trailing_silence(decoded, SAMPLE_RATE)
        if trimmed.shape[0] == 0:
            raise BusinessError(
                code="EMPTY_AUDIO_SEGMENT",
                message=(
                    f"target dialogue contains no non-silent samples: "
                    f"{segment.audio_path}"
                ),
                exit_code=EXIT_AUDIO,
                file=segment.audio_path,
                details={
                    "segment_id": segment.segment_id,
                    "role": "target_dialogue",
                },
            )
        clips.append(PreparedClip(segment=segment, audio=trimmed))
    return clips


def _check_offset_bounds(
    manifest: Manifest,
    offset_ms: int,
) -> None:
    offset_sec = offset_ms / 1000.0
    for segment in manifest.segments:
        shifted_start = segment.target_start_sec + offset_sec
        if shifted_start < -1.0 / SAMPLE_RATE:
            raise _timeline_error(
                "OFFSET_OUT_OF_RANGE",
                (
                    f"global offset moves segment '{segment.segment_id}' "
                    "before time zero"
                ),
                file=manifest.path,
                details={
                    "segment_id": segment.segment_id,
                    "target_start_sec": segment.target_start_sec,
                    "offset_ms": offset_ms,
                },
            )


def _schedule_clips(
    clips: list[PreparedClip],
    *,
    video_samples: int,
    offset_ms: int,
    max_crossfade_samples: int,
) -> list[dict[str, Any]]:
    offset_sec = offset_ms / 1000.0
    cumulative_shift = 0
    previous_actual_end = 0
    global_warnings: list[dict[str, Any]] = []

    for clip in clips:
        segment = clip.segment
        base_start = seconds_to_samples(
            segment.target_start_sec + offset_sec
        )
        slot_samples = max(
            1,
            seconds_to_samples(segment.target_end_sec + offset_sec)
            - seconds_to_samples(segment.target_start_sec + offset_sec),
        )
        start = base_start + cumulative_shift

        clip_length = clip.audio.shape[0]
        if clip_length > slot_samples:
            overflow = clip_length - slot_samples
            cumulative_shift += overflow
            clip.caused_shift = True
            clip.warnings.append(
                warning(
                    "OVERFLOW_SHIFTED_FOLLOWING",
                    (
                        "localized clip is longer than its manifest slot; "
                        "following segments were shifted"
                    ),
                    segment_id=segment.segment_id,
                    overflow_ms=round(overflow * 1000.0 / SAMPLE_RATE, 3),
                )
            )

        if start < previous_actual_end - max_crossfade_samples:
            delta = previous_actual_end - max_crossfade_samples - start
            start += delta
            cumulative_shift += delta
            clip.warnings.append(
                warning(
                    "OVERLAP_REDUCED_TO_CROSSFADE",
                    (
                        "segment overlap exceeded the configured crossfade; "
                        "the segment was shifted right"
                    ),
                    segment_id=segment.segment_id,
                    shift_ms=round(delta * 1000.0 / SAMPLE_RATE, 3),
                    max_crossfade_ms=round(
                        max_crossfade_samples * 1000.0 / SAMPLE_RATE, 3
                    ),
                )
            )

        auto_shift = start - base_start
        clip.auto_shift_samples = auto_shift
        clip.shifted = auto_shift > 0
        if clip.shifted and not clip.warnings:
            clip.warnings.append(
                warning(
                    "SHIFTED_AFTER_PREVIOUS_OVERFLOW",
                    "segment was shifted to preserve preceding semantic audio",
                    segment_id=segment.segment_id,
                    shift_ms=round(auto_shift * 1000.0 / SAMPLE_RATE, 3),
                )
            )

        clip.mapped_start_sample = start
        clip.actual_start_sample = min(max(start, 0), video_samples)
        intended_end = start + clip_length
        clip.actual_end_sample = min(intended_end, video_samples)
        if clip.actual_start_sample > video_samples:
            clip.actual_start_sample = video_samples
        if clip.actual_end_sample < clip.actual_start_sample:
            clip.actual_end_sample = clip.actual_start_sample

        if intended_end > video_samples:
            clip.truncated = True
            clip.warnings.append(
                warning(
                    "TRUNCATED_AT_VIDEO_END",
                    "segment was truncated to the video duration",
                    segment_id=segment.segment_id,
                    truncated_ms=round(
                        (intended_end - video_samples) * 1000.0 / SAMPLE_RATE,
                        3,
                    ),
                )
            )

        available = clip.actual_end_sample - clip.actual_start_sample
        if available > 0:
            clip.overlap_samples = max(
                0,
                min(
                    max_crossfade_samples,
                    previous_actual_end - clip.actual_start_sample,
                    available,
                ),
            )
        else:
            clip.overlap_samples = 0

        previous_actual_end = min(
            clip.actual_start_sample + clip_length, video_samples
        )
        global_warnings.extend(clip.warnings)

    return global_warnings


def _normalize_scheduled_clips(
    clips: list[PreparedClip],
    *,
    target_lufs: float,
    peak_dbfs: float,
) -> None:
    for clip in clips:
        available = clip.actual_end_sample - clip.actual_start_sample
        if available <= 0:
            clip.audio = clip.audio[:0].copy()
            continue
        clip.audio = clip.audio[:available].copy()
        try:
            clip.audio, _ = normalize_segment(
                clip.audio,
                target_lufs=target_lufs,
                peak_dbfs=peak_dbfs,
            )
        except ValueError as exc:
            raise BusinessError(
                code="EMPTY_AUDIO_SEGMENT",
                message=(
                    f"segment '{clip.segment.segment_id}' has no measurable "
                    "loudness after timeline truncation"
                ),
                exit_code=EXIT_AUDIO,
                file=clip.segment.audio_path,
                details={"segment_id": clip.segment.segment_id},
            ) from exc


def _mix_clips(
    clips: list[PreparedClip],
    *,
    video_samples: int,
    peak_dbfs: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mix = np.zeros((video_samples, 2), dtype=np.float64)
    warnings: list[dict[str, Any]] = []
    previous_end = 0

    for clip in sorted(
        clips, key=lambda item: (item.actual_start_sample, item.segment.segment_id)
    ):
        if clip.audio.shape[0] == 0:
            continue
        start = clip.actual_start_sample
        end = min(start + clip.audio.shape[0], video_samples)
        if end <= start:
            continue
        audio = clip.audio[: end - start]
        overlap = min(
            clip.overlap_samples,
            max(0, previous_end - start),
            audio.shape[0],
        )
        if overlap > 0:
            fade_out, fade_in = equal_power_fades(overlap)
            mix[start : start + overlap] = (
                mix[start : start + overlap] * fade_out[:, np.newaxis]
                + audio[:overlap] * fade_in[:, np.newaxis]
            )
            mix[start + overlap : end] += audio[overlap:]
        else:
            mix[start:end] += audio
        previous_end = end

    mix_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    ceiling = 10.0 ** ((peak_dbfs - 0.01) / 20.0)
    if mix_peak > ceiling:
        mix = limit_peak(mix, ceiling)
        warnings.append(
            warning(
                "MIX_PEAK_LIMITED",
                "final mix was peak-limited to preserve the -1 dBFS ceiling",
                peak_before_dbfs=round(linear_to_db(mix_peak), 6),
                peak_after_dbfs=round(
                    linear_to_db(float(np.max(np.abs(mix)))), 6
                ),
            )
        )
    return quantize_pcm16(mix), warnings


def _build_report(
    *,
    manifest: Manifest,
    clips: list[PreparedClip],
    mixed_audio: np.ndarray,
    output_path: str,
    report_path: str,
    target_language: str,
    offset_ms: int,
    target_lufs: float,
    peak_dbfs: float,
    max_crossfade_ms: float,
    global_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    segment_reports: list[dict[str, Any]] = []
    for clip in clips:
        start = clip.actual_start_sample
        end = clip.actual_end_sample
        if end > start:
            measured_audio = mixed_audio[start:end]
            loudness = integrated_loudness(measured_audio)
            peak_linear = float(np.max(np.abs(measured_audio)))
            peak = linear_to_db(peak_linear)
        else:
            loudness = -math.inf
            peak = -math.inf
        segment_reports.append(
            {
                "id": clip.segment.segment_id,
                "target_path": clip.segment.audio_path,
                "subtitle": clip.segment.target_text,
                "source_start_sec": round(clip.segment.source_start_sec, 6),
                "source_end_sec": round(clip.segment.source_end_sec, 6),
                "manifest_target_start_sec": round(
                    clip.segment.target_start_sec, 6
                ),
                "manifest_target_end_sec": round(
                    clip.segment.target_end_sec, 6
                ),
                "mapped_start_sec": round(
                    clip.mapped_start_sample / SAMPLE_RATE, 6
                ),
                "actual_start_sec": round(start / SAMPLE_RATE, 6),
                "actual_end_sec": round(end / SAMPLE_RATE, 6),
                "duration_sec": round((end - start) / SAMPLE_RATE, 6),
                "integrated_loudness_lufs": _finite_or_none(loudness),
                "peak_dbfs": _finite_or_none(peak),
                "shifted": clip.shifted,
                "shift_ms": round(
                    clip.auto_shift_samples * 1000.0 / SAMPLE_RATE, 3
                ),
                "caused_shift": clip.caused_shift,
                "truncated": clip.truncated,
                "crossfade_ms": round(
                    clip.overlap_samples * 1000.0 / SAMPLE_RATE, 3
                ),
                "warnings": clip.warnings,
            }
        )

    return {
        "status": "ok",
        "code": "OK",
        "exit_code": 0,
        "schema_version": 1,
        "target_language": target_language,
        "global_offset_ms": offset_ms,
        "manifest": manifest.path,
        "source_dialogue": {
            "path": manifest.source_audio.path,
            "sample_rate": manifest.source_audio.sample_rate,
            "channels": manifest.source_audio.channels,
        },
        "video": {
            "path": manifest.video.path,
            "duration_sec": round(manifest.video.duration_sec, 6),
            "fps": manifest.video.fps,
            "width": manifest.video.width,
            "height": manifest.video.height,
        },
        "output": {
            "path": output_path,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "sample_format": "PCM_16",
            "samples": int(mixed_audio.shape[0]),
            "duration_sec": round(
                mixed_audio.shape[0] / SAMPLE_RATE, 6
            ),
        },
        "report_path": report_path,
        "processing": {
            "loudness_target_lufs": target_lufs,
            "loudness_tolerance_lu": 1.0,
            "peak_ceiling_dbfs": peak_dbfs,
            "max_crossfade_ms": max_crossfade_ms,
        },
        "segments": segment_reports,
        "warnings": global_warnings,
    }


def _write_json_temp(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def _publish_atomically(
    *,
    audio: np.ndarray,
    report: dict[str, Any],
    output_path: str,
    report_path: str,
) -> None:
    output = Path(output_path)
    report_file = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    wav_temp = output.parent / f".{output.name}.{token}.part"
    report_temp = report_file.parent / f".{report_file.name}.{token}.part"
    output_backup = output.parent / f".{output.name}.{token}.bak"
    report_backup = report_file.parent / f".{report_file.name}.{token}.bak"
    output_had_previous = output.exists()
    report_had_previous = report_file.exists()
    output_committed = False
    report_committed = False

    try:
        sf.write(
            str(wav_temp),
            audio,
            SAMPLE_RATE,
            subtype="PCM_16",
            format="WAV",
        )
        written_info = sf.info(str(wav_temp))
        if (
            written_info.samplerate != SAMPLE_RATE
            or written_info.channels != 2
            or written_info.subtype != "PCM_16"
            or written_info.frames != audio.shape[0]
        ):
            raise RuntimeError(
                "written WAV does not match the requested canonical format"
            )
        _write_json_temp(report_temp, report)

        if output_had_previous:
            os.replace(output, output_backup)
        if report_had_previous:
            os.replace(report_file, report_backup)
        os.replace(wav_temp, output)
        output_committed = True
        os.replace(report_temp, report_file)
        report_committed = True
    except Exception as exc:
        for path in (wav_temp, report_temp):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if report_committed:
            try:
                report_file.unlink(missing_ok=True)
            except OSError:
                pass
        if output_committed:
            try:
                output.unlink(missing_ok=True)
            except OSError:
                pass
        if output_had_previous and output_backup.exists():
            try:
                os.replace(output_backup, output)
            except OSError:
                pass
        if report_had_previous and report_backup.exists():
            try:
                os.replace(report_backup, report_file)
            except OSError:
                pass
        raise BusinessError(
            code="OUTPUT_WRITE_FAILED",
            message=f"could not publish output files: {exc}",
            exit_code=EXIT_OUTPUT,
            file=str(output),
            details={"report_path": str(report_file)},
        ) from exc
    else:
        for backup in (output_backup, report_backup):
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass


def run_localization(
    *,
    manifest_path: str,
    target_language: str,
    output_path: str,
    report_path: str,
    offset_ms: int = 0,
    target_lufs: float = -23.0,
    peak_dbfs: float = -1.0,
    crossfade_ms: float = 10.0,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    output_resolved = str(Path(output_path).resolve())
    report_resolved = str(Path(report_path).resolve())
    if output_resolved == report_resolved:
        raise BusinessError(
            code="INVALID_ARGUMENTS",
            message="output WAV and report JSON must use different paths",
            exit_code=2,
            file=output_resolved,
        )
    if not 0.0 <= crossfade_ms <= 10.0:
        raise BusinessError(
            code="INVALID_ARGUMENTS",
            message="crossfade must be between 0 and 10 milliseconds",
            exit_code=2,
            details={"crossfade_ms": crossfade_ms},
        )
    max_crossfade_samples = min(
        int(round(crossfade_ms * SAMPLE_RATE / 1000.0)),
        int(round(0.010 * SAMPLE_RATE)),
    )

    manifest = load_manifest(manifest_path, target_language)
    _check_offset_bounds(manifest, offset_ms)
    tools = find_tools(ffmpeg=ffmpeg, ffprobe=ffprobe)

    source_info = probe_audio(manifest.source_audio.path, tools)
    validate_source_format(
        source_info,
        manifest.source_audio.path,
        manifest.source_audio.sample_rate,
        manifest.source_audio.channels,
    )

    clips = _prepare_clips(manifest, tools)
    video_samples = seconds_to_samples(manifest.video.duration_sec)
    if video_samples <= 0:
        raise BusinessError(
            code="INVALID_TIME_RANGE",
            message="video duration must produce at least one audio sample",
            exit_code=2,
            file=manifest.path,
        )

    global_warnings = _schedule_clips(
        clips,
        video_samples=video_samples,
        offset_ms=offset_ms,
        max_crossfade_samples=max_crossfade_samples,
    )
    _normalize_scheduled_clips(
        clips,
        target_lufs=target_lufs,
        peak_dbfs=peak_dbfs,
    )
    mixed_audio, mix_warnings = _mix_clips(
        clips,
        video_samples=video_samples,
        peak_dbfs=peak_dbfs,
    )
    global_warnings.extend(mix_warnings)

    report = _build_report(
        manifest=manifest,
        clips=clips,
        mixed_audio=mixed_audio,
        output_path=output_resolved,
        report_path=report_resolved,
        target_language=target_language,
        offset_ms=offset_ms,
        target_lufs=target_lufs,
        peak_dbfs=peak_dbfs,
        max_crossfade_ms=crossfade_ms,
        global_warnings=global_warnings,
    )
    _publish_atomically(
        audio=mixed_audio,
        report=report,
        output_path=output_resolved,
        report_path=report_resolved,
    )
    return report

