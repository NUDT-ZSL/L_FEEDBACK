"""Event timeline planning, streaming mixdown, and JSON reporting."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .dsp import (
    DEFAULT_KAISER_BETA,
    DEFAULT_SINC_HALF_WIDTH,
    crossfade_sample_count,
    loop_crossfade_block,
    panning_gains,
    resample_into,
)
from .errors import BusinessError, ErrorCode
from .events import EventSpec, load_events
from .wavio import (
    AtomicPcmWavWriter,
    WavInfo,
    WavReader,
    atomic_write_json,
    downmix_chunk,
    inspect_wav,
)

DEFAULT_TARGET_SAMPLE_RATE = 48000
DEFAULT_OUTPUT_BIT_DEPTH = 16
DEFAULT_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_CHUNK_FRAMES = 65536
MAX_IN_MEMORY_ARRAY_BYTES = 256 * 1024 * 1024
FLUSH_INTERVAL_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class EventPlan:
    spec: EventSpec
    info: WavInfo
    trigger_sample: int
    resampled_frames: int
    loop_start_sample: Optional[int]
    loop_end_sample: Optional[int]
    crossfade_samples: int


@dataclass
class MixResult:
    timeline_frames: int
    peak_before_soft_clip: float
    peak_after_soft_clip: float
    dc_offset_before_soft_clip: tuple[float, float]
    event_reports: list[dict]
    mix_storage: str
    source_storage: str


class _ArrayStore:
    """In-memory array or scratch memmap with explicit release semantics."""

    def __init__(
        self,
        array: Union[np.ndarray, np.memmap],
        path: Optional[Path] = None,
        storage: str = "memory",
    ):
        self.array = array
        self.path = path
        self.storage = storage
        self.closed = False

    @classmethod
    def allocate(
        cls,
        shape: tuple[int, ...],
        dtype: np.dtype,
        scratch_dir: Path,
        label: str,
        in_memory_cap_bytes: int,
        *,
        zero: bool,
    ) -> "_ArrayStore":
        dtype = np.dtype(dtype)
        item_count = int(np.prod(shape, dtype=np.int64))
        byte_count = item_count * dtype.itemsize
        if byte_count <= in_memory_cap_bytes:
            array = (
                np.zeros(shape, dtype=dtype)
                if zero
                else np.empty(shape, dtype=dtype)
            )
            return cls(array=array, path=None, storage="memory")

        path = scratch_dir / f"{label}.f64"
        try:
            array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
            return cls(array=array, path=path, storage="memmap")
        except OSError as exc:
            raise BusinessError(
                ErrorCode.OUTPUT_DENIED,
                f"cannot allocate scratch storage: {exc}",
            ) from exc

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        array = self.array
        self.array = None
        if isinstance(array, np.memmap):
            try:
                array.flush()
            except (OSError, ValueError):
                pass
            mmap_object = getattr(array, "_mmap", None)
            del array
            if mmap_object is not None:
                try:
                    mmap_object.close()
                except (OSError, ValueError, BufferError):
                    pass
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self.path = None


def _round_sample(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))


def _report_path_for(
    output_path: Path, report_path: Optional[Union[str, os.PathLike[str]]]
) -> Path:
    if report_path is not None:
        return Path(report_path).expanduser().resolve(strict=False)
    return output_path.with_suffix(".report.json").resolve(strict=False)


def _new_error_at(
    error: BusinessError, event_index: int
) -> BusinessError:
    return BusinessError(
        code=error.code,
        message=error.message,
        event_index=event_index,
        details=error.details,
    )


def _build_plan(spec: EventSpec, target_sample_rate: int) -> EventPlan:
    try:
        info = inspect_wav(spec.source_path)
    except BusinessError as exc:
        raise _new_error_at(exc, spec.index) from exc

    if info.frames <= 0:
        raise BusinessError(
            ErrorCode.INVALID_EVENT,
            "source WAV contains no audio frames",
            event_index=spec.index,
            details={"path": str(spec.source_path)},
        )

    trigger_sample = _round_sample(spec.trigger_seconds, target_sample_rate)
    resampled_frames = int(
        math.floor(
            info.frames * target_sample_rate / info.sample_rate + 0.5
        )
    )
    if resampled_frames <= 0:
        raise BusinessError(
            ErrorCode.INVALID_EVENT,
            "event is too short at the target sample rate",
            event_index=spec.index,
        )

    loop_start_sample: Optional[int] = None
    loop_end_sample: Optional[int] = None
    crossfade_count = 0
    if spec.loop is not None:
        source_duration = info.frames / float(info.sample_rate)
        if spec.loop.end_seconds > source_duration + 1e-12:
            raise BusinessError(
                ErrorCode.INVALID_EVENT,
                (
                    f"loop end {spec.loop.end_seconds:.9f}s exceeds source "
                    f"duration {source_duration:.9f}s"
                ),
                event_index=spec.index,
                details={"path": str(spec.source_path)},
            )
        loop_start_sample = _round_sample(
            spec.loop.start_seconds, target_sample_rate
        )
        loop_end_sample = _round_sample(
            spec.loop.end_seconds, target_sample_rate
        )
        if not 0 <= loop_start_sample < loop_end_sample <= resampled_frames:
            raise BusinessError(
                ErrorCode.INVALID_EVENT,
                (
                    "loop region maps outside the valid target-sample range "
                    f"[0, {resampled_frames}]"
                ),
                event_index=spec.index,
            )
        crossfade_count = crossfade_sample_count(
            loop_end_sample - loop_start_sample, target_sample_rate
        )

    return EventPlan(
        spec=spec,
        info=info,
        trigger_sample=trigger_sample,
        resampled_frames=resampled_frames,
        loop_start_sample=loop_start_sample,
        loop_end_sample=loop_end_sample,
        crossfade_samples=crossfade_count,
    )


def _accumulate_event(
    plan: EventPlan,
    mix: _ArrayStore,
    scratch_dir: Path,
    target_sample_rate: int,
    chunk_frames: int,
    array_cap_bytes: int,
) -> dict:
    spec = plan.spec
    info = plan.info
    source_store: Optional[_ArrayStore] = None
    wave_store = _ArrayStore.allocate(
        shape=(plan.resampled_frames,),
        dtype=np.float64,
        scratch_dir=scratch_dir,
        label=f"event_{spec.index:04d}_audio",
        in_memory_cap_bytes=array_cap_bytes,
        zero=False,
    )
    try:
        source_store = _ArrayStore.allocate(
            shape=(info.frames,),
            dtype=np.float64,
            scratch_dir=scratch_dir,
            label=f"event_{spec.index:04d}_source",
            in_memory_cap_bytes=array_cap_bytes,
            zero=False,
        )
        with WavReader(info.path) as reader:
            bytes_since_flush = 0
            for position, chunk in reader.iter_frames(chunk_frames):
                mono_chunk = downmix_chunk(chunk)
                source_store.array[position : position + mono_chunk.shape[0]] = (
                    mono_chunk
                )
                bytes_since_flush += mono_chunk.nbytes
                if (
                    isinstance(source_store.array, np.memmap)
                    and bytes_since_flush >= FLUSH_INTERVAL_BYTES
                ):
                    source_store.array.flush()
                    bytes_since_flush = 0
            if isinstance(source_store.array, np.memmap):
                source_store.array.flush()

        if info.sample_rate == target_sample_rate:
            bytes_since_flush = 0
            copy_start = 0
            while copy_start < plan.resampled_frames:
                copy_end = min(
                    copy_start + chunk_frames, plan.resampled_frames
                )
                np.copyto(
                    wave_store.array[copy_start:copy_end],
                    source_store.array[copy_start:copy_end],
                )
                bytes_since_flush += (copy_end - copy_start) * 8
                if (
                    isinstance(wave_store.array, np.memmap)
                    and bytes_since_flush >= FLUSH_INTERVAL_BYTES
                ):
                    wave_store.array.flush()
                    bytes_since_flush = 0
                copy_start = copy_end
            if isinstance(wave_store.array, np.memmap):
                wave_store.array.flush()
        else:
            resample_into(
                source_store.array,
                info.sample_rate,
                target_sample_rate,
                wave_store.array,
                half_width=DEFAULT_SINC_HALF_WIDTH,
                beta=DEFAULT_KAISER_BETA,
                output_block_frames=max(chunk_frames, 16384),
            )

        crossfaded_tail_stereo: Optional[np.ndarray] = None
        boundary_before: Optional[float] = None
        boundary_after: Optional[float] = None
        if (
            plan.loop_start_sample is not None
            and plan.loop_end_sample is not None
            and plan.crossfade_samples > 0
        ):
            start = plan.loop_start_sample
            end = plan.loop_end_sample
            fade = plan.crossfade_samples
            tail = np.asarray(wave_store.array[end - fade : end]).copy()
            head = np.asarray(wave_store.array[start : start + fade]).copy()
            head_left, head_right = panning_gains(
                fade,
                target_sample_rate,
                spec.pan_frequency_hz,
                spec.pan_depth,
                spec.pan_phase_degrees,
                start_frame=start,
            )
            tail_left, tail_right = panning_gains(
                fade,
                target_sample_rate,
                spec.pan_frequency_hz,
                spec.pan_depth,
                spec.pan_phase_degrees,
                start_frame=end - fade,
            )
            panned_head = np.column_stack(
                (head * head_left, head * head_right)
            )
            panned_tail = np.column_stack(
                (tail * tail_left, tail * tail_right)
            )
            crossfaded_tail_stereo = np.empty((fade, 2), dtype=np.float64)
            boundary_before = 0.0
            boundary_after = 0.0
            for channel in range(2):
                (
                    crossfaded_tail_stereo[:, channel],
                    channel_before,
                    channel_after,
                ) = loop_crossfade_block(
                    panned_tail[:, channel], panned_head[:, channel]
                )
                boundary_before = max(boundary_before, channel_before)
                boundary_after = max(boundary_after, channel_after)
        elif plan.loop_start_sample is not None and plan.loop_end_sample is not None:
            start = plan.loop_start_sample
            end = plan.loop_end_sample
            start_left, start_right = panning_gains(
                1,
                target_sample_rate,
                spec.pan_frequency_hz,
                spec.pan_depth,
                spec.pan_phase_degrees,
                start_frame=start,
            )
            end_left, end_right = panning_gains(
                1,
                target_sample_rate,
                spec.pan_frequency_hz,
                spec.pan_depth,
                spec.pan_phase_degrees,
                start_frame=end - 1,
            )
            start_value = wave_store.array[start]
            end_value = wave_store.array[end - 1]
            boundary_before = max(
                abs(end_value * end_left[0] - start_value * start_left[0]),
                abs(end_value * end_right[0] - start_value * start_right[0]),
            )
            boundary_after = boundary_before

        frame_start = 0
        mix_chunks_since_flush = 0
        mix_flush_chunks = max(
            1,
            FLUSH_INTERVAL_BYTES // max(1, chunk_frames * 2 * 8),
        )
        while frame_start < plan.resampled_frames:
            frame_end = min(
                frame_start + chunk_frames, plan.resampled_frames
            )
            block = np.asarray(
                wave_store.array[frame_start:frame_end], dtype=np.float64
            ).copy()

            left_gain, right_gain = panning_gains(
                frame_end - frame_start,
                target_sample_rate,
                spec.pan_frequency_hz,
                spec.pan_depth,
                spec.pan_phase_degrees,
                start_frame=frame_start,
            )
            panned_block = np.empty(
                (frame_end - frame_start, 2), dtype=np.float64
            )
            panned_block[:, 0] = block * left_gain
            panned_block[:, 1] = block * right_gain

            if (
                crossfaded_tail_stereo is not None
                and plan.loop_start_sample is not None
                and plan.loop_end_sample is not None
            ):
                fade = plan.crossfade_samples
                fade_start = plan.loop_end_sample - fade
                overlap_start = max(frame_start, fade_start)
                overlap_end = min(frame_end, plan.loop_end_sample)
                if overlap_start < overlap_end:
                    destination_start = overlap_start - frame_start
                    destination_end = overlap_end - frame_start
                    tail_start = overlap_start - fade_start
                    tail_end = overlap_end - fade_start
                    panned_block[destination_start:destination_end] = (
                        crossfaded_tail_stereo[tail_start:tail_end]
                    )
            mix_start = plan.trigger_sample + frame_start
            mix_end = plan.trigger_sample + frame_end
            mix.array[mix_start:mix_end] += panned_block
            mix_chunks_since_flush += 1
            if (
                isinstance(mix.array, np.memmap)
                and mix_chunks_since_flush >= mix_flush_chunks
            ):
                mix.array.flush()
                mix_chunks_since_flush = 0
            frame_start = frame_end

        if isinstance(mix.array, np.memmap):
            mix.array.flush()

        loop_report = None
        if plan.loop_start_sample is not None and plan.loop_end_sample is not None:
            loop_report = {
                "start_seconds": float(spec.loop.start_seconds),
                "end_seconds": float(spec.loop.end_seconds),
                "start_sample": int(plan.loop_start_sample),
                "end_sample": int(plan.loop_end_sample),
                "crossfade_samples": int(plan.crossfade_samples),
                "crossfade_seconds": (
                    plan.crossfade_samples / float(target_sample_rate)
                ),
                "algorithm": "equal_power_tail_to_rotated_head",
                "boundary_jump_before": float(boundary_before or 0.0),
                "boundary_jump_after": float(boundary_after or 0.0),
                "frame_count_preserved": True,
            }

        timing_error = (
            plan.trigger_sample
            - spec.trigger_seconds * target_sample_rate
        )
        mapped_start = (
            plan.loop_start_sample if plan.loop_start_sample is not None else None
        )
        mapped_end = (
            plan.loop_end_sample if plan.loop_end_sample is not None else None
        )
        event_report = {
            "index": int(spec.index),
            "id": spec.event_id,
            "status": "applied",
            "source_path": str(info.path),
            "file": str(info.path),
            "source_format": info.as_dict(),
            "source_encoding": info.encoding,
            "source_sample_rate": int(info.sample_rate),
            "source_bit_depth": int(info.bits_per_sample),
            "source_channels": int(info.channels),
            "source_frames": int(info.frames),
            "source_duration_seconds": info.duration_seconds,
            "trigger_seconds": float(spec.trigger_seconds),
            "trigger_sample": int(plan.trigger_sample),
            "trigger_sample_time_seconds": (
                plan.trigger_sample / float(target_sample_rate)
            ),
            "timing_error_samples": float(timing_error),
            "resampled_frames": int(plan.resampled_frames),
            "output_frames": int(plan.resampled_frames),
            "duration_seconds": (
                plan.resampled_frames / float(target_sample_rate)
            ),
            "resampling": {
                "used": info.sample_rate != target_sample_rate,
                "method": "windowed_sinc_kaiser",
                "half_width": DEFAULT_SINC_HALF_WIDTH,
                "kaiser_beta": DEFAULT_KAISER_BETA,
            },
            "pan_frequency_hz": float(spec.pan_frequency_hz),
            "pan_depth": float(spec.pan_depth),
            "panning": {
                "frequency_hz": float(spec.pan_frequency_hz),
                "depth": float(spec.pan_depth),
                "phase_degrees": float(spec.pan_phase_degrees),
                "algorithm": "equal_power_sine_lfo",
                "mapping": "theta=(depth*sin(2*pi*f*t+phase)+1)*pi/4",
                "center_gain": float(math.sqrt(0.5)),
                "constant_power": True,
            },
            "loop_cleanup": loop_report,
            "loop_start_sample": mapped_start,
            "loop_end_sample": mapped_end,
            "crossfade_samples": int(plan.crossfade_samples),
            "output_path": None,
        }
        return event_report
    finally:
        wave_store.close()
        if source_store is not None:
            source_store.close()


def _mix_events(
    plans: list[EventPlan],
    output_path: Path,
    target_sample_rate: int,
    output_bit_depth: int,
    chunk_frames: int,
    array_cap_bytes: int,
) -> MixResult:
    timeline_frames = max(
        plan.trigger_sample + plan.resampled_frames for plan in plans
    )
    if timeline_frames <= 0:
        raise BusinessError(
            ErrorCode.INVALID_EVENT,
            "event timeline is empty",
        )

    with tempfile.TemporaryDirectory(prefix="audio_event_mixer_") as scratch_name:
        scratch_dir = Path(scratch_name)
        mix_store = _ArrayStore.allocate(
            shape=(timeline_frames, 2),
            dtype=np.float64,
            scratch_dir=scratch_dir,
            label="timeline_mix",
            in_memory_cap_bytes=array_cap_bytes,
            zero=True,
        )
        try:
            event_reports: list[dict] = []
            for plan in plans:
                event_reports.append(
                    _accumulate_event(
                        plan,
                        mix_store,
                        scratch_dir,
                        target_sample_rate,
                        chunk_frames,
                        array_cap_bytes,
                    )
                )

            peak_before = 0.0
            channel_sum = np.zeros(2, dtype=np.float64)
            for start in range(0, timeline_frames, chunk_frames):
                stop = min(start + chunk_frames, timeline_frames)
                chunk = np.asarray(mix_store.array[start:stop]).copy()
                if chunk.size:
                    peak_before = max(
                        peak_before, float(np.max(np.abs(chunk)))
                    )
                    channel_sum += chunk.sum(axis=0, dtype=np.float64)
            dc_offset = (
                float(channel_sum[0] / timeline_frames),
                float(channel_sum[1] / timeline_frames),
            )

            peak_after = 0.0
            try:
                writer = AtomicPcmWavWriter(
                    output_path,
                    sample_rate=target_sample_rate,
                    bit_depth=output_bit_depth,
                )
                with writer:
                    for start in range(0, timeline_frames, chunk_frames):
                        stop = min(start + chunk_frames, timeline_frames)
                        soft_limited = np.tanh(
                            np.asarray(mix_store.array[start:stop])
                        )
                        if soft_limited.size:
                            peak_after = max(
                                peak_after,
                                float(np.max(np.abs(soft_limited))),
                            )
                        writer.write(soft_limited)
                    if writer.frames_written != timeline_frames:
                        raise BusinessError(
                            ErrorCode.OUTPUT_DENIED,
                            "output frame count does not match the timeline",
                            details={
                                "expected": timeline_frames,
                                "actual": writer.frames_written,
                            },
                        )
            except BusinessError:
                raise
            except OSError as exc:
                raise BusinessError(
                    ErrorCode.OUTPUT_DENIED,
                    f"cannot write output WAV: {exc}",
                    details={"path": str(output_path)},
                ) from exc

            for event_report in event_reports:
                event_report["output_path"] = str(output_path)

            return MixResult(
                timeline_frames=timeline_frames,
                peak_before_soft_clip=peak_before,
                peak_after_soft_clip=peak_after,
                dc_offset_before_soft_clip=dc_offset,
                event_reports=event_reports,
                mix_storage=mix_store.storage,
                source_storage=(
                    "memory_or_memmap_per_event"
                    if any(plan.info.sample_rate != target_sample_rate for plan in plans)
                    else "memory_or_memmap_per_event_at_target_rate"
                ),
            )
        finally:
            mix_store.close()


def _success_report(
    mix_result: MixResult,
    plans: list[EventPlan],
    output_path: Path,
    report_path: Path,
    target_sample_rate: int,
    output_bit_depth: int,
    memory_limit_bytes: int,
    array_cap_bytes: int,
    chunk_frames: int,
) -> dict:
    duration = mix_result.timeline_frames / float(target_sample_rate)
    output_encoding = (
        "pcm_s16le" if output_bit_depth == 16 else "pcm_s24le"
    )
    return {
        "schema_version": "1.0",
        "status": "ok",
        "error_code": None,
        "failure_event_index": None,
        "target_sample_rate": int(target_sample_rate),
        "output_sample_rate": int(target_sample_rate),
        "output_bit_depth": int(output_bit_depth),
        "output_path": str(output_path),
        "output_file": str(output_path),
        "report_path": str(report_path),
        "event_count": len(plans),
        "timeline_frames": int(mix_result.timeline_frames),
        "timeline_duration_seconds": duration,
        "timeline": {
            "frames": int(mix_result.timeline_frames),
            "duration_seconds": duration,
            "trigger_rounding": "nearest_sample_half_up",
            "overlap_method": "sample_wise_sum",
            "soft_clip": "tanh",
        },
        "output": {
            "path": str(output_path),
            "format": "WAV",
            "encoding": output_encoding,
            "bit_depth": int(output_bit_depth),
            "channels": 2,
            "sample_rate": int(target_sample_rate),
            "frames": int(mix_result.timeline_frames),
            "duration_seconds": duration,
            "atomic_replace": True,
        },
        "processing": {
            "resampler": "windowed_sinc_kaiser",
            "sinc_half_width": DEFAULT_SINC_HALF_WIDTH,
            "kaiser_beta": DEFAULT_KAISER_BETA,
            "resample_same_rate_copy": True,
            "io_chunk_frames": int(chunk_frames),
            "memory_limit_bytes": int(memory_limit_bytes),
            "in_memory_array_cap_bytes": int(array_cap_bytes),
            "mix_storage": mix_result.mix_storage,
            "source_storage": mix_result.source_storage,
            "peak_before_soft_clip": mix_result.peak_before_soft_clip,
            "peak_after_soft_clip": mix_result.peak_after_soft_clip,
            "dc_offset_before_soft_clip": {
                "left": mix_result.dc_offset_before_soft_clip[0],
                "right": mix_result.dc_offset_before_soft_clip[1],
            },
            "normalized_peak": (
                mix_result.peak_after_soft_clip
                > 0.0
                and mix_result.peak_after_soft_clip <= 1.0
            ),
        },
        "events": mix_result.event_reports,
    }


def _failure_report(
    error: BusinessError,
    output_path: Path,
    target_sample_rate: int,
    event_count: int,
) -> dict:
    return {
        "schema_version": "1.0",
        "status": "error",
        "error_code": error.code,
        "failure_event_index": error.event_index,
        "error": error.as_dict(),
        "target_sample_rate": int(target_sample_rate),
        "output_path": str(output_path),
        "event_count": int(event_count),
        "events": [],
    }


def process_events(
    events_path: Union[str, os.PathLike[str]],
    output_path: Union[str, os.PathLike[str]],
    report_path: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    output_bit_depth: int = DEFAULT_OUTPUT_BIT_DEPTH,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> dict:
    """Process an event JSON into a stereo WAV mix and JSON report.

    The output WAV is staged in the destination directory and atomically
    replaced only after every event has been mixed and written successfully.
    On expected failures, a failure report is written (when possible) and a
    :class:`BusinessError` is raised.
    """

    if memory_limit_bytes < 8 * 1024 * 1024:
        raise ValueError("memory_limit_bytes must be at least 8 MiB")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    output = Path(output_path).expanduser().resolve(strict=False)
    report = _report_path_for(output, report_path)
    events: list[EventSpec] = []
    try:
        if target_sample_rate <= 0:
            raise BusinessError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"invalid target sample rate: {target_sample_rate}",
            )
        if output_bit_depth not in (16, 24):
            raise BusinessError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"unsupported output bit depth: {output_bit_depth}",
            )
        array_cap = max(
            1024 * 1024,
            min(MAX_IN_MEMORY_ARRAY_BYTES, memory_limit_bytes // 8),
        )
        events = load_events(events_path)
        plans = [_build_plan(spec, target_sample_rate) for spec in events]
        mix_result = _mix_events(
            plans,
            output,
            target_sample_rate,
            output_bit_depth,
            chunk_frames,
            array_cap,
        )
        report_payload = _success_report(
            mix_result,
            plans,
            output,
            report,
            target_sample_rate,
            output_bit_depth,
            memory_limit_bytes,
            array_cap,
            chunk_frames,
        )
        atomic_write_json(report, report_payload)
        return report_payload
    except BusinessError as exc:
        failure_payload = _failure_report(
            exc, output, target_sample_rate, len(events)
        )
        try:
            atomic_write_json(report, failure_payload)
        except BusinessError as report_error:
            exc.details = dict(exc.details or {})
            exc.details["report_write_error"] = {
                "code": report_error.code,
                "message": report_error.message,
            }
        raise
