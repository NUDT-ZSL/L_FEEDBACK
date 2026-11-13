"""Single-job rendering, warning detection, and playback timeline construction."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import ConfigError, TrackInputError, WavError
from .resample import SincConfig, render_track_chunk
from .resources import RenderLimits, ResourceGovernor
from .timeline import (
    ChannelSlot,
    Policy,
    ProjectSpec,
    SourceRoute,
    TrackSpec,
    alignment_to_dict,
    fit_alignment,
)
from .util import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    unique_temp_path,
)
from .version import PLAYBACK_SCHEMA, TOOL_VERSION
from .wav import (
    ExtensibleWavSink,
    WavInfo,
    metadata_comment,
    parse_wav,
)


@dataclass(frozen=True)
class EngineConfig:
    limits: RenderLimits = RenderLimits()
    sinc: SincConfig = SincConfig()
    on_track_error: str | None = None
    bit_depth: int | None = None
    target_sample_rate: int | None = None
    silence_threshold_dbfs: float | None = None
    min_silence_seconds: float | None = None


@dataclass
class TrackRuntime:
    spec: TrackSpec
    info: WavInfo | None
    routes: list[tuple[SourceRoute, ...]]
    temp_path: Path | None
    warnings: list[dict[str, Any]]
    peak: float = 0.0
    gain: float = 1.0
    status: str = "ok"
    rendered_frames: int = 0
    coverage_start_seconds: float = 0.0
    coverage_end_seconds: float = 0.0


@dataclass
class TaskResult:
    task_id: str
    wav_path: Path
    timeline_path: Path
    timeline: dict[str, Any]
    timeline_bytes: bytes
    warnings: list[dict[str, Any]]
    degraded: bool
    output_wav_sha256: str
    output_wav_size: int
    output_timeline_sha256: str
    output_timeline_size: int


def _warning(
    code: str,
    severity: str,
    track_id: str,
    start_frame: int,
    end_frame: int,
    sample_rate: int,
    total_frames: int,
    message: str,
    action: str,
) -> dict[str, Any]:
    start_frame = max(0, min(total_frames, int(start_frame)))
    end_frame = max(start_frame, min(total_frames, int(end_frame)))
    return {
        "code": code,
        "severity": severity,
        "recoverable": True,
        "track_id": track_id,
        "action": action,
        "range": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_seconds": start_frame / sample_rate,
            "end_seconds": end_frame / sample_rate,
        },
        "message": message,
    }


def _apply_overrides(project: ProjectSpec, config: EngineConfig) -> tuple[ProjectSpec, Policy]:
    program = project.program
    sample_rate = (
        program.sample_rate
        if config.target_sample_rate is None
        else config.target_sample_rate
    )
    if sample_rate < 8000 or sample_rate > 768000:
        raise ConfigError("target sample rate must be between 8000 and 768000")
    bit_depth = program.bit_depth if config.bit_depth is None else config.bit_depth
    if bit_depth not in (16, 24, 32):
        raise ConfigError("bit depth must be 16, 24, or 32")
    on_track_error = (
        project.policy.on_track_error
        if config.on_track_error is None
        else config.on_track_error
    )
    if on_track_error not in {"silence", "fail"}:
        raise ConfigError("on_track_error must be 'silence' or 'fail'")
    threshold = (
        project.policy.silence_threshold_dbfs
        if config.silence_threshold_dbfs is None
        else config.silence_threshold_dbfs
    )
    min_silence = (
        project.policy.min_silence_seconds
        if config.min_silence_seconds is None
        else config.min_silence_seconds
    )
    if threshold > 0:
        raise ConfigError("silence threshold must be <= 0 dBFS")
    if min_silence < 0:
        raise ConfigError("minimum silence duration cannot be negative")
    policy = Policy(
        on_track_error=on_track_error,
        silence_threshold_dbfs=threshold,
        min_silence_seconds=min_silence,
        max_anchor_error_ms=project.policy.max_anchor_error_ms,
    )
    total_frames = max(
        1, int(math.floor(program.duration_seconds * sample_rate + 0.5))
    )
    cue_frames = [
        min(
            total_frames,
            max(0, int(math.floor(cue.program_seconds * sample_rate + 0.5))),
        )
        for cue in project.cues
    ]
    cues = tuple(
        replace(
            cue,
            frame=frame,
            program_seconds=frame / sample_rate,
        )
        for cue, frame in zip(project.cues, cue_frames)
    )
    tracks: dict[str, TrackSpec] = {}
    for track in project.tracks.values():
        fit = (
            track.alignment
            if sample_rate == program.sample_rate
            else fit_alignment(track.anchors, sample_rate)
        )
        tracks[track.id] = replace(track, alignment=fit)
    return (
        replace(
            project,
            program=replace(
                program,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                total_frames=total_frames,
                duration_seconds=total_frames / sample_rate,
            ),
            tracks=tracks,
            cues=cues,
            policy=policy,
        ),
        policy,
    )


def _slot_ordinal(track: TrackSpec, slot_index: int) -> int:
    return track.slots.index(slot_index)


def _resolve_routes(
    track: TrackSpec,
    slots: tuple[ChannelSlot, ...],
    source_channels: int,
) -> list[tuple[SourceRoute, ...]]:
    resolved: list[tuple[SourceRoute, ...]] = []
    track_slot_indices = track.slots
    for slot_index in track_slot_indices:
        slot = slots[slot_index]
        if slot.sources is not None:
            routes = slot.sources
        else:
            ordinal = _slot_ordinal(track, slot_index)
            if len(track.slots) == 1 and source_channels > 1:
                routes = tuple(
                    SourceRoute(channel=channel, gain=1.0 / source_channels)
                    for channel in range(source_channels)
                )
            elif source_channels == 1:
                routes = (SourceRoute(channel=0, gain=1.0),)
            elif ordinal < source_channels:
                routes = (SourceRoute(channel=ordinal, gain=1.0),)
            else:
                raise ConfigError(
                    f"cannot infer source channel for output slot {slot.name!r} "
                    f"of track {track.id!r}; specify sources explicitly"
                )
        for route in routes:
            if route.channel >= source_channels:
                raise ConfigError(
                    f"channel_layout slot {slot.name!r} references source channel "
                    f"{route.channel}, but track {track.id!r} has {source_channels}"
                )
        resolved.append(routes)
    return resolved


def _merge_ranges(
    ranges: list[tuple[int, int]], min_frames: int, total_frames: int
) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[list[int]] = []
    for start, end in ranges:
        start = max(0, min(total_frames, start))
        end = max(start, min(total_frames, end))
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= min_frames]


class _SilenceDetector:
    def __init__(self, threshold_dbfs: float):
        self.threshold = 10.0 ** (threshold_dbfs / 20.0)
        self.ranges: list[tuple[int, int]] = []
        self._active_start: int | None = None
        self._active_end = 0

    def feed(self, chunk: np.ndarray, start_frame: int, frames: int) -> None:
        if frames <= 0:
            return
        if chunk.size == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        end_frame = start_frame + frames
        if rms <= self.threshold:
            if self._active_start is None:
                self._active_start = start_frame
            self._active_end = end_frame
        elif self._active_start is not None:
            self.ranges.append((self._active_start, self._active_end))
            self._active_start = None

    def finish(self, total_frames: int) -> list[tuple[int, int]]:
        if self._active_start is not None:
            self.ranges.append((self._active_start, self._active_end))
            self._active_start = None
        return self.ranges


def _append_float_temp(handle: Any, audio: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(audio, dtype="<f4")
    handle.write(contiguous.tobytes())


def _safe_peak(bits: int) -> float:
    scale = float(1 << (bits - 1))
    return 1.0 - 0.5 / scale - 1.0e-10


def _timeline_track_dict(
    runtime: TrackRuntime,
    target_rate: int,
    warnings_for_track: list[dict[str, Any]],
    channel_slots: tuple[ChannelSlot, ...],
) -> dict[str, Any]:
    info = runtime.info
    alignment = alignment_to_dict(runtime.spec.alignment)
    input_dict = (
        {
            "path": str(runtime.spec.path),
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "bit_depth": info.bits_per_sample,
            "format": info.audio_format,
            "frames": info.frames,
            "duration_seconds": info.frames / info.sample_rate,
            "channel_mask": info.channel_mask,
        }
        if info is not None
        else {
            "path": str(runtime.spec.path),
            "sample_rate": None,
            "channels": 0,
            "bit_depth": None,
            "format": "unreadable",
            "frames": 0,
            "duration_seconds": 0.0,
            "channel_mask": 0,
        }
    )
    return {
        "id": runtime.spec.id,
        "name": runtime.spec.name,
        "role": runtime.spec.role,
        "input": input_dict,
        "output_channels": [
            {
                "slot_index": slot_index,
                "name": channel_slots[slot_index].name,
                "speaker": channel_slots[slot_index].speaker,
            }
            for slot_index in runtime.spec.slots
        ],
        "alignment": alignment,
        "coverage": {
            "program_start_seconds": runtime.coverage_start_seconds,
            "program_end_seconds": runtime.coverage_end_seconds,
            "rendered_frames": runtime.rendered_frames,
        },
        "processing": {
            "resampler": "kaiser_sinc_fractional_delay",
            "target_sample_rate": target_rate,
            "peak_float": runtime.peak,
            "safety_gain": runtime.gain,
            "status": runtime.status,
            "warning_count": len(warnings_for_track),
        },
        "warning_codes": [warning["code"] for warning in warnings_for_track],
    }


def _build_playback_timeline(
    *,
    task_id: str,
    project: ProjectSpec,
    runtimes: Mapping[str, TrackRuntime],
    warnings: list[dict[str, Any]],
    resampler: SincConfig,
) -> dict[str, Any]:
    warnings_by_track: dict[str, list[dict[str, Any]]] = {}
    for warning in warnings:
        warnings_by_track.setdefault(warning["track_id"], []).append(warning)
    channel_layout = [
        {
            "index": slot.index,
            "name": slot.name,
            "speaker": slot.speaker,
            "mask_bit": slot.mask_bit,
            "track_id": slot.track_id,
        }
        for slot in project.channel_slots
    ]
    cues = [
        {
            "id": cue.id,
            "name": cue.name,
            "wav_cue_id": cue.wav_cue_id,
            "frame": cue.frame,
            "program_seconds": cue.frame / project.program.sample_rate,
            "wav_cue_position": cue.frame,
            "wav_data_chunk_id": "data",
        }
        for cue in project.cues
    ]
    return {
        "schema_version": PLAYBACK_SCHEMA,
        "tool_version": TOOL_VERSION,
        "task_id": task_id,
        "event": dict(project.event),
        "time_semantics": {
            "program_origin": "first output sample frame",
            "cue_frame": "absolute frame offset in the WAV data chunk",
            "cue_seconds": "cue_frame / program.sample_rate",
            "alignment_model": "program_seconds = offset_seconds + drift_factor * source_seconds",
            "drift_ppm": "(drift_factor - 1) * 1000000",
        },
        "program": {
            "sample_rate": project.program.sample_rate,
            "duration_seconds": project.program.total_frames / project.program.sample_rate,
            "total_frames": project.program.total_frames,
            "bit_depth": project.program.bit_depth,
            "channels": project.program.channels,
            "channel_mask": project.program.channel_mask,
            "channel_layout": channel_layout,
        },
        "tracks": [
            _timeline_track_dict(
                runtime,
                project.program.sample_rate,
                warnings_by_track.get(runtime.spec.id, []),
                project.channel_slots,
            )
            for runtime in runtimes.values()
        ],
        "cues": cues,
        "warnings": warnings,
        "resampler": {
            "algorithm": "kaiser_sinc_fractional_delay",
            "half_width": resampler.half_width,
            "max_half_width": resampler.max_half_width,
            "normalized_kernel_sum": True,
        },
        "integrity": None,
    }


def _attach_integrity(timeline: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    digest_payload = dict(timeline)
    digest_payload.pop("integrity", None)
    digest = sha256_bytes(canonical_json_bytes(digest_payload))
    timeline = dict(digest_payload)
    timeline["integrity"] = {
        "algorithm": "sha256",
        "digest": digest,
        "scope": "canonical timeline JSON with integrity removed",
    }
    return timeline, canonical_json_bytes(timeline)


def _mix_and_write_output(
    *,
    wav_path: Path,
    project: ProjectSpec,
    runtimes: Mapping[str, TrackRuntime],
    timeline: Mapping[str, Any],
    timeline_bytes: bytes,
    limits: RenderLimits,
    governor: ResourceGovernor,
) -> str:
    info_metadata = {
        "title": str(project.event.get("title", project.event.get("id", "EventMix program"))),
        "artist": str(project.event.get("broadcast", project.event.get("venue", ""))),
        "date": str(project.event.get("date", "")),
        "software": f"EventMix {TOOL_VERSION}",
        "comment": metadata_comment(
            project.event,
            timeline["integrity"]["digest"],
            len(project.cues),
        ),
    }
    temp_wav = unique_temp_path(wav_path.parent, wav_path.name, ".wav.tmp")
    sink = ExtensibleWavSink(
        temp_wav,
        sample_rate=project.program.sample_rate,
        bits=project.program.bit_depth,
        channels=project.program.channels,
        channel_mask=project.program.channel_mask,
        total_frames=project.program.total_frames,
        cues=timeline["cues"],
        timeline_bytes=timeline_bytes,
        info_metadata=info_metadata,
    )
    temp_handles: dict[str, Any] = {}
    try:
        for track_id, runtime in runtimes.items():
            if runtime.temp_path is not None:
                temp_handles[track_id] = open(runtime.temp_path, "rb")
        frame = 0
        chunk_frames = limits.chunk_frames
        while frame < project.program.total_frames:
            count = min(chunk_frames, project.program.total_frames - frame)
            with governor.memory(
                count * project.program.channels * 8
                + count * max(1, project.program.channels) * 8
                + 1024
            ):
                mixed = np.zeros((count, project.program.channels), dtype=np.float64)
                for runtime in runtimes.values():
                    handle = temp_handles.get(runtime.spec.id)
                    destination_count = len(runtime.spec.slots)
                    if handle is None or destination_count == 0:
                        continue
                    expected_bytes = count * destination_count * 4
                    raw = handle.read(expected_bytes)
                    if len(raw) != expected_bytes:
                        read_error_marked = any(
                            item.get("code") == "wav_read_error"
                            for item in runtime.warnings
                        )
                        if not read_error_marked:
                            raise WavError(
                                f"temporary track data for {runtime.spec.id!r} is truncated"
                            )
                        complete_frames = len(raw) // (destination_count * 4)
                        values = np.zeros(
                            (count, destination_count), dtype=np.float64
                        )
                        usable_bytes = complete_frames * destination_count * 4
                        if usable_bytes:
                            values[:complete_frames, :] = np.frombuffer(
                                raw[:usable_bytes], dtype="<f4"
                            ).reshape(complete_frames, destination_count)
                    else:
                        values = np.frombuffer(raw, dtype="<f4").reshape(
                            count, destination_count
                        ).astype(np.float64)
                    values = values.astype(np.float64) * runtime.gain
                    for local_index, slot_index in enumerate(runtime.spec.slots):
                        mixed[:, slot_index] = values[:, local_index]
                sink.write_float(mixed)
            frame += count
        if sink.clamp_count:
            raise WavError(
                f"quantizer required {sink.clamp_count} safety clamps; "
                "peak normalization invariant failed"
            )
        sink.close()
        os.replace(temp_wav, wav_path)
    except BaseException:
        try:
            sink.close()
        except BaseException:
            pass
        try:
            temp_wav.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        for handle in temp_handles.values():
            handle.close()
    return sha256_file(wav_path)


def build_task(
    *,
    task_id: str,
    project: ProjectSpec,
    wav_path: Path,
    timeline_path: Path,
    governor: ResourceGovernor,
    config: EngineConfig,
) -> TaskResult:
    project, policy = _apply_overrides(project, config)
    limits = config.limits
    runtimes: dict[str, TrackRuntime] = {}
    warnings: list[dict[str, Any]] = []
    temp_paths: list[Path] = []
    temp_dir = wav_path.parent / ".eventmix-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for track in project.tracks.values():
            runtime_warnings: list[dict[str, Any]] = []
            if track.alignment.max_abs_error_ms > policy.max_anchor_error_ms:
                anchor_frames = [
                    int(round(item.expected_program_seconds * project.program.sample_rate))
                    for item in track.alignment.anchor_errors
                ]
                warning = _warning(
                    "anchor_fit_residual_high",
                    "warning",
                    track.id,
                    min(anchor_frames),
                    max(anchor_frames),
                    project.program.sample_rate,
                    project.program.total_frames,
                    (
                        f"anchor fit maximum residual {track.alignment.max_abs_error_ms:.3f} ms "
                        f"exceeds policy {policy.max_anchor_error_ms:.3f} ms"
                    ),
                    "report_only",
                )
                runtime_warnings.append(warning)

            try:
                info = parse_wav(track.path)
            except WavError as exc:
                if policy.on_track_error == "fail":
                    raise TrackInputError(
                        f"track {track.id!r} ({track.path}) is unreadable: {exc}"
                    ) from exc
                warning = _warning(
                    "wav_header_corrupt",
                    "error",
                    track.id,
                    0,
                    project.program.total_frames,
                    project.program.sample_rate,
                    project.program.total_frames,
                    f"WAV could not be read ({exc}); track replaced with silence placeholder",
                    "silent_placeholder",
                )
                runtime_warnings.append(warning)
                runtime = TrackRuntime(
                    spec=track,
                    info=None,
                    routes=[],
                    temp_path=None,
                    warnings=runtime_warnings,
                    status="degraded",
                    coverage_start_seconds=track.alignment.offset_seconds,
                    coverage_end_seconds=track.alignment.offset_seconds,
                )
                runtimes[track.id] = runtime
                warnings.extend(runtime_warnings)
                continue

            routes = _resolve_routes(track, project.channel_slots, info.channels)
            coverage_start = track.alignment.source_to_program_seconds(0.0)
            coverage_end = track.alignment.source_to_program_seconds(
                info.frames / info.sample_rate
            )
            if coverage_start > 0.5 / project.program.sample_rate:
                start_frame = 0
                end_frame = int(round(coverage_start * project.program.sample_rate))
                runtime_warnings.append(
                    _warning(
                        "track_short_start",
                        "warning",
                        track.id,
                        start_frame,
                        end_frame,
                        project.program.sample_rate,
                        project.program.total_frames,
                        "input begins after program time zero; missing range is silent",
                        "silence_padding",
                    )
                )
            if coverage_end < project.program.duration_seconds - 0.5 / project.program.sample_rate:
                start_frame = max(
                    0, int(math.floor(coverage_end * project.program.sample_rate))
                )
                runtime_warnings.append(
                    _warning(
                        "track_short_end",
                        "warning",
                        track.id,
                        start_frame,
                        project.program.total_frames,
                        project.program.sample_rate,
                        project.program.total_frames,
                        (
                            f"input covers program only to {coverage_end:.6f} s; "
                            "remaining range is silent"
                        ),
                        "silence_padding",
                    )
                )

            temp_path = unique_temp_path(temp_dir, f"{track.id}.track", ".f32")
            temp_paths.append(temp_path)
            detector = _SilenceDetector(policy.silence_threshold_dbfs)
            peak = 0.0
            rendered = 0
            truncated_warning: dict[str, Any] | None = None
            with open(temp_path, "wb") as temp_handle:
                frame = 0
                while frame < project.program.total_frames:
                    count = min(limits.chunk_frames, project.program.total_frames - frame)
                    try:
                        chunk = render_track_chunk(
                            info=info,
                            routes=routes,
                            alignment=track.alignment,
                            output_start_frame=frame,
                            output_frames=count,
                            output_sample_rate=project.program.sample_rate,
                            source_frames=info.frames,
                            governor=governor,
                            config=config.sinc,
                        )
                    except WavError as exc:
                        if policy.on_track_error == "fail":
                            raise TrackInputError(
                                f"track {track.id!r} failed during rendering: {exc}"
                            ) from exc
                        truncated_warning = _warning(
                            "wav_read_error",
                            "error",
                            track.id,
                            frame,
                            project.program.total_frames,
                            project.program.sample_rate,
                            project.program.total_frames,
                            f"read failure ({exc}); remaining track range replaced with silence",
                            "silent_placeholder",
                        )
                        break
                    _append_float_temp(temp_handle, chunk)
                    detector.feed(chunk, frame, count)
                    peak = max(peak, float(np.max(np.abs(chunk), initial=0.0)))
                    rendered += count
                    frame += count
            if truncated_warning is not None:
                runtime_warnings.append(truncated_warning)
                coverage_end = min(
                    coverage_end, rendered / project.program.sample_rate
                )

            if peak <= detector.threshold:
                if rendered > 0:
                    runtime_warnings.append(
                        _warning(
                            "track_silent",
                            "warning",
                            track.id,
                            0,
                            project.program.total_frames,
                            project.program.sample_rate,
                            project.program.total_frames,
                            (
                                f"track peak {20.0 * math.log10(max(peak, 1e-12)):.1f} dBFS "
                                "is below the silence threshold"
                            ),
                            "report_only",
                        )
                    )
            else:
                min_silence_frames = int(
                    round(policy.min_silence_seconds * project.program.sample_rate)
                )
                for start, end in _merge_ranges(
                    detector.finish(project.program.total_frames),
                    min_silence_frames,
                    project.program.total_frames,
                ):
                    runtime_warnings.append(
                        _warning(
                            "track_silent_segment",
                            "warning",
                            track.id,
                            start,
                            end,
                            project.program.sample_rate,
                            project.program.total_frames,
                            "aligned track energy is below the silence threshold",
                            "report_only",
                        )
                    )
            runtime = TrackRuntime(
                spec=track,
                info=info,
                routes=routes,
                temp_path=temp_path,
                warnings=runtime_warnings,
                peak=peak,
                rendered_frames=rendered,
                coverage_start_seconds=coverage_start,
                coverage_end_seconds=coverage_end,
            )
            if any(w["severity"] == "warning" for w in runtime_warnings):
                runtime.status = "warning"
            if any(
                w["code"]
                in {"track_short_start", "track_short_end", "wav_read_error"}
                for w in runtime_warnings
            ):
                runtime.status = "degraded"
            runtimes[track.id] = runtime
            warnings.extend(runtime_warnings)

        safe_peak = _safe_peak(project.program.bit_depth)
        for runtime in runtimes.values():
            if runtime.peak > safe_peak:
                runtime.gain = safe_peak / runtime.peak
            if runtime.temp_path is None:
                runtime.gain = 1.0

        timeline = _build_playback_timeline(
            task_id=task_id,
            project=project,
            runtimes=runtimes,
            warnings=warnings,
            resampler=config.sinc,
        )
        timeline, timeline_bytes = _attach_integrity(timeline)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        wav_sha = _mix_and_write_output(
            wav_path=wav_path,
            project=project,
            runtimes=runtimes,
            timeline=timeline,
            timeline_bytes=timeline_bytes,
            limits=limits,
            governor=governor,
        )
        atomic_write_bytes(timeline_path, timeline_bytes)
        timeline_sha = sha256_bytes(timeline_bytes)
        degraded = any(w.get("severity") in {"warning", "error"} for w in warnings)
        return TaskResult(
            task_id=task_id,
            wav_path=wav_path,
            timeline_path=timeline_path,
            timeline=timeline,
            timeline_bytes=timeline_bytes,
            warnings=warnings,
            degraded=degraded,
            output_wav_sha256=wav_sha,
            output_wav_size=wav_path.stat().st_size,
            output_timeline_sha256=timeline_sha,
            output_timeline_size=timeline_path.stat().st_size,
        )
    finally:
        for path in temp_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def task_report_from_result(
    result: TaskResult, *, status: str = "completed"
) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "status": status,
        "degraded": result.degraded,
        "outputs": {
            "wav": str(result.wav_path),
            "timeline": str(result.timeline_path),
            "wav_sha256": result.output_wav_sha256,
            "wav_size": result.output_wav_size,
            "timeline_sha256": result.output_timeline_sha256,
            "timeline_size": result.output_timeline_size,
        },
        "warnings": result.warnings,
        "alignment": [
            {
                "track_id": track["id"],
                "offset_seconds": track["alignment"]["offset_seconds"],
                "drift_factor": track["alignment"]["drift_factor"],
                "drift_ppm": track["alignment"]["drift_ppm"],
                "max_abs_anchor_error_ms": track["alignment"]["max_abs_anchor_error_ms"],
                "anchor_errors": track["alignment"]["anchor_errors"],
                "status": track["processing"]["status"],
                "safety_gain": track["processing"]["safety_gain"],
            }
            for track in result.timeline["tracks"]
        ],
        "cues": result.timeline["cues"],
        "timeline_integrity": result.timeline["integrity"],
    }

