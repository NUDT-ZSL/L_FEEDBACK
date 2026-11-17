"""Resumable, block-streaming spatial audio rendering engine."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

from . import __version__
from .audio import (
    AudioInfo,
    INPUT_SUBTYPES,
    OutputFormat,
    atomic_replace,
    encode_raw_to_wav,
    open_input,
    parse_output_format,
    probe_audio,
    read_source_span,
    validate_bit_depth,
)
from .config import (
    SourceConfig,
    SourceSpec,
    load_source_config,
    relative_angles,
    right_up_forward_basis,
)
from .errors import (
    BusinessError,
    E_CONFIG_INVALID,
    E_HRTF_VERSION,
    E_IO,
    E_MEMORY_LIMIT,
    E_STATE_INVALID,
    ProcessingInterrupted,
)
from .hrtf import HRTFDataset, generate_hrtf
from .state import (
    STATE_SCHEMA_VERSION,
    atomic_write_json,
    load_state,
    remove_path,
    validate_state_identity,
)

CONTRIBUTION_DTYPE = np.float64
CONTRIBUTION_CHANNELS = 2
CONTRIBUTION_FRAME_BYTES = CONTRIBUTION_CHANNELS * np.dtype(CONTRIBUTION_DTYPE).itemsize
MAX_RESAMPLE_RATE = 8.0


@dataclass(frozen=True)
class RenderOptions:
    input_path: str
    config_path: str
    output_path: str
    output_format: str
    block_size: int
    listener_position: Tuple[float, float, float]
    listener_forward: Tuple[float, float, float]
    listener_up: Tuple[float, float, float]
    filter_order: str = "before"
    state_path: str | None = None
    speed_of_sound: float = 343.0
    stop_after_blocks: int | None = None
    stop_after_mix_blocks: int | None = None
    fresh: bool = False
    expected_hrtf_version: str | None = None


@dataclass(frozen=True)
class RenderResult:
    output_path: str
    frames: int
    sample_rate: int
    output_format: str
    block_size: int
    source_count: int
    peak_abs: float
    resumed: bool
    committed_blocks: int
    peak_working_bytes_estimate: int
    input_decoded_bytes: int
    hrtf: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": True,
            "output_path": self.output_path,
            "frames": self.frames,
            "sample_rate": self.sample_rate,
            "channels": CONTRIBUTION_CHANNELS,
            "output_format": self.output_format,
            "block_size": self.block_size,
            "source_count": self.source_count,
            "peak_abs": self.peak_abs,
            "resumed": self.resumed,
            "committed_blocks": self.committed_blocks,
            "peak_working_bytes_estimate": self.peak_working_bytes_estimate,
            "input_decoded_bytes": self.input_decoded_bytes,
            "hrtf": self.hrtf,
        }


@dataclass
class _SourcePlan:
    spec: SourceSpec
    channels: Tuple[int, ...]
    ir: np.ndarray
    direction_index: Tuple[int, int]
    distance_filter_alpha: float | None
    representative_distance: float


def _safe_remove(path: Path) -> None:
    try:
        if path.is_dir():
            remove_path(path)
        elif path.exists():
            path.unlink()
    except OSError:
        pass


def _state_vector(
    state: Mapping[str, Any], key: str, expected_length: int, state_path: Path
) -> np.ndarray:
    raw = state.get(key, [])
    if not isinstance(raw, list):
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state field {key} must be a list",
            {"state_path": str(state_path), "field": key},
        )
    if len(raw) == 0:
        return np.zeros(expected_length, dtype=np.float64)
    if len(raw) != expected_length:
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state field {key} has an invalid length",
            {
                "state_path": str(state_path),
                "field": key,
                "expected": expected_length,
                "actual": len(raw),
            },
        )
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state field {key} contains non-numeric values",
            {"state_path": str(state_path), "field": key},
        ) from exc
    if not np.all(np.isfinite(array)):
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state field {key} contains non-finite values",
            {"state_path": str(state_path), "field": key},
        )
    return array


def _state_scalar(state: Mapping[str, Any], key: str, state_path: Path) -> float:
    try:
        value = float(state.get(key, 0.0))
    except (TypeError, ValueError) as exc:
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state field {key} must be numeric",
            {"state_path": str(state_path), "field": key},
        ) from exc
    if not math.isfinite(value):
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state field {key} must be finite",
            {"state_path": str(state_path), "field": key},
        )
    return value


def _file_frames(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return -1
    if size % CONTRIBUTION_FRAME_BYTES:
        return -1
    return size // CONTRIBUTION_FRAME_BYTES


def _truncate_frames(path: Path, frames: int, state_path: Path, label: str) -> None:
    actual = _file_frames(path)
    if actual < 0:
        raise BusinessError(
            E_STATE_INVALID,
            f"Intermediate source file is missing or truncated: {label}",
            {"path": str(path), "expected_frames": frames},
        )
    if actual < frames:
        raise BusinessError(
            E_STATE_INVALID,
            f"Intermediate source file is shorter than the committed state: {label}",
            {
                "path": str(path),
                "expected_min_frames": frames,
                "actual_frames": actual,
                "state_path": str(state_path),
            },
        )
    if actual > frames:
        with open(path, "r+b") as handle:
            handle.truncate(frames * CONTRIBUTION_FRAME_BYTES)


def _ensure_source_file(path: Path, frames: int, state_path: Path, label: str) -> None:
    if not path.exists():
        if frames != 0:
            raise BusinessError(
                E_STATE_INVALID,
                f"Intermediate source file is missing: {label}",
                {"path": str(path), "expected_frames": frames},
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return
    _truncate_frames(path, frames, state_path, label)


def _write_zero_prefix(handle: Any, frames: int) -> None:
    if frames <= 0:
        return
    chunk_frames = 65536
    zeros = np.zeros(
        (min(frames, chunk_frames), CONTRIBUTION_CHANNELS),
        dtype=CONTRIBUTION_DTYPE,
    )
    remaining = frames
    while remaining > 0:
        count = min(remaining, chunk_frames)
        if count != zeros.shape[0]:
            zeros = np.zeros((count, CONTRIBUTION_CHANNELS), dtype=CONTRIBUTION_DTYPE)
        handle.write(zeros.tobytes())
        remaining -= count


def _append_samples(handle: Any, samples: np.ndarray) -> None:
    handle.write(np.asarray(samples, dtype=CONTRIBUTION_DTYPE).tobytes())


def _read_contribution(
    path: Path, start_frame: int, frames: int, total_frames: int
) -> np.ndarray:
    output = np.zeros((frames, CONTRIBUTION_CHANNELS), dtype=CONTRIBUTION_DTYPE)
    if frames <= 0:
        return output
    valid_start = max(0, start_frame)
    valid_end = min(total_frames, start_frame + frames)
    if valid_end <= valid_start:
        return output
    with open(path, "rb") as handle:
        handle.seek(valid_start * CONTRIBUTION_FRAME_BYTES)
        payload = handle.read((valid_end - valid_start) * CONTRIBUTION_FRAME_BYTES)
    expected = (valid_end - valid_start) * CONTRIBUTION_FRAME_BYTES
    if len(payload) != expected:
        raise BusinessError(
            E_IO,
            "Intermediate source file changed while mixing",
            {
                "path": str(path),
                "expected_bytes": expected,
                "actual_bytes": len(payload),
            },
        )
    block = np.frombuffer(payload, dtype=CONTRIBUTION_DTYPE).reshape(
        -1, CONTRIBUTION_CHANNELS
    )
    destination = valid_start - start_frame
    output[destination : destination + block.shape[0]] = block
    return output


def _convolution_segment(context: np.ndarray, ir_pair: np.ndarray) -> np.ndarray:
    ir_length = ir_pair.shape[1]
    output = np.empty((context.size - ir_length + 1, 2), dtype=np.float64)
    for ear in range(2):
        full = np.convolve(
            context, ir_pair[ear].astype(np.float64, copy=False), mode="full"
        )
        output[:, ear] = full[ir_length - 1 : ir_length - 1 + output.shape[0]]
    return output


def _lowpass_filter(
    data: np.ndarray, alpha: float, state: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, alpha - 1.0], dtype=np.float64)
    filtered, next_state = lfilter(b, a, data, axis=0, zi=state)
    return filtered, next_state


def _estimate_peak_working_bytes(
    block_size: int,
    ir_length: int,
) -> int:
    source_context = (block_size + ir_length) * np.dtype(np.float64).itemsize
    source_estimate = (
        source_context * 12 + ir_length * 2 * np.dtype(np.float64).itemsize * 4 + 4096
    )
    mix_estimate = (
        block_size * CONTRIBUTION_CHANNELS * np.dtype(CONTRIBUTION_DTYPE).itemsize * 4
        + 4096
    )
    return max(source_estimate, mix_estimate)


def _validate_listener_inputs(
    options: RenderOptions,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    listener = np.asarray(options.listener_position, dtype=np.float64)
    forward = np.asarray(options.listener_forward, dtype=np.float64)
    up_hint = np.asarray(options.listener_up, dtype=np.float64)
    if not (
        np.all(np.isfinite(listener))
        and np.all(np.isfinite(forward))
        and np.all(np.isfinite(up_hint))
    ):
        raise BusinessError(
            E_CONFIG_INVALID,
            "Listener position and orientation must be finite",
            {
                "position": list(options.listener_position),
                "forward": list(options.listener_forward),
                "up": list(options.listener_up),
            },
        )
    right, up, forward_unit = right_up_forward_basis(forward, up_hint)
    return listener, right, up, forward_unit


def _resolve_source_channels(
    sources: Sequence[SourceSpec], input_channels: int
) -> List[Tuple[int, ...]]:
    resolved: List[Tuple[int, ...]] = []
    for source in sources:
        channels = (
            source.channels
            if source.channels is not None
            else (source.index % input_channels,)
        )
        if any(channel >= input_channels for channel in channels):
            raise BusinessError(
                E_CONFIG_INVALID,
                "Source channel index is outside the input WAV channel count",
                {
                    "source_index": source.index,
                    "source_id": source.source_id,
                    "channels": list(channels),
                    "input_channels": input_channels,
                },
            )
        resolved.append(tuple(channels))
    return resolved


def _build_source_plans(
    config: SourceConfig,
    input_info: AudioInfo,
    hrtf: HRTFDataset,
    listener: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> List[_SourcePlan]:
    channels = _resolve_source_channels(config.sources, input_info.channels)
    plans: List[_SourcePlan] = []
    for source, source_channels in zip(config.sources, channels):
        relative = source.reference_position - listener
        azimuth, elevation, _distance = relative_angles(relative, right, up, forward)
        direction_index, ir_pair = hrtf.select(
            azimuth,
            elevation,
            azimuth_offset_deg=source.azimuth_offset_deg,
            elevation_offset_deg=source.elevation_offset_deg,
            enabled=source.spatialize,
        )
        representative_distance = source.motion.representative_distance(listener)
        alpha = source.air_absorption.alpha(
            representative_distance, input_info.sample_rate
        )
        plans.append(
            _SourcePlan(
                spec=source,
                channels=source_channels,
                ir=ir_pair,
                direction_index=direction_index,
                distance_filter_alpha=alpha,
                representative_distance=representative_distance,
            )
        )
    return plans


def _source_envelope(
    plan: _SourcePlan,
    sample_indices: np.ndarray,
    sample_rate: int,
    listener: np.ndarray,
    speed_of_sound: float,
) -> Tuple[np.ndarray, np.ndarray]:
    source = plan.spec
    if sample_indices.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    times = sample_indices.astype(np.float64) / float(sample_rate)
    distances = source.motion.distances_at(times, listener)
    distance_gain = source.distance_model.gain(distances)
    gain = float(source.gain) * float(source.hrtf_gain) * distance_gain
    phase_rates = np.ones(sample_indices.size, dtype=np.float64)
    if source.motion.times.size > 1:
        radial_speed = source.motion.radial_velocity_at(times, listener)
        c = float(speed_of_sound)
        denominator = c + radial_speed
        phase_rates = np.clip(
            c / np.maximum(denominator, 0.05),
            1.0 / MAX_RESAMPLE_RATE,
            MAX_RESAMPLE_RATE,
        )
    return gain, phase_rates


def _resample_block(
    handle: Any,
    plan: _SourcePlan,
    sample_indices: np.ndarray,
    phases: np.ndarray,
    input_info: AudioInfo,
) -> np.ndarray:
    if sample_indices.size == 0:
        return np.zeros(0, dtype=np.float64)
    valid = (
        (sample_indices >= 0)
        & (sample_indices < input_info.frames)
        & (phases >= 0.0)
        & (phases < float(input_info.frames))
    )
    output = np.zeros(sample_indices.size, dtype=np.float64)
    if not np.any(valid):
        return output
    valid_phases = phases[valid]
    base = np.floor(valid_phases).astype(np.int64)
    span_first = int(np.min(base))
    span_last = int(np.max(base)) + 1
    span = read_source_span(
        handle,
        plan.channels,
        span_first,
        span_last - span_first + 1,
        input_info.frames,
    )
    local = base - span_first
    fraction = valid_phases - base
    output[valid] = span[local] * (1.0 - fraction) + span[local + 1] * fraction
    return output


def _initial_state(identity: Mapping[str, Any]) -> Dict[str, Any]:
    state = dict(identity)
    state.update(
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "stage": "sources",
            "source_index": 0,
            "block_index": 0,
            "processed_samples": 0,
            "phase": 0.0,
            "filter_zi": [],
            "filter_history": [],
            "source_history": [],
            "mix_offset": 0,
        }
    )
    return state


def _transition_to_next_source(
    state: Dict[str, Any],
    source_index: int,
    state_path: Path,
) -> None:
    state["source_index"] = source_index + 1
    state["block_index"] = 0
    state["processed_samples"] = 0
    state["phase"] = 0.0
    state["filter_zi"] = []
    state["filter_history"] = []
    state["source_history"] = []
    if state["source_index"] >= int(state["num_sources"]):
        state["stage"] = "mix"
        state["mix_offset"] = 0
    atomic_write_json(state_path, state)


def _render_tail(
    plan: _SourcePlan,
    contribution_handle: Any,
    state: Mapping[str, Any],
    state_path: Path,
) -> None:
    ir_length = plan.ir.shape[1]
    overlap_length = ir_length - 1
    if state["filter_order"] == "before" and plan.distance_filter_alpha is not None:
        history = _state_vector(state, "filter_history", overlap_length, state_path)
        context = np.concatenate((history, np.zeros(overlap_length)))
    else:
        history = _state_vector(state, "source_history", overlap_length, state_path)
        context = np.concatenate((history, np.zeros(overlap_length)))

    segment = _convolution_segment(context, plan.ir)
    if state["filter_order"] == "after" and plan.distance_filter_alpha is not None:
        zi = _state_vector(state, "filter_zi", 2, state_path)
        segment, _ = _lowpass_filter(
            segment, plan.distance_filter_alpha, zi.reshape(1, 2)
        )
    _append_samples(contribution_handle, segment)


def _render_source_block(
    plan: _SourcePlan,
    handle: Any,
    input_info: AudioInfo,
    listener: np.ndarray,
    speed_of_sound: float,
    state: Dict[str, Any],
    state_path: Path,
    maybe_handle: Any,
) -> np.ndarray:
    block_size = int(state["block_size"])
    offset = int(state["processed_samples"])
    count = min(block_size, input_info.frames - offset)
    sample_indices = np.arange(offset, offset + count, dtype=np.int64)
    phase_start = _state_scalar(state, "phase", state_path)
    gain, rates = _source_envelope(
        plan, sample_indices, input_info.sample_rate, listener, speed_of_sound
    )

    if count > 0:
        phases = phase_start + np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(rates[:-1], dtype=np.float64))
        )
    else:
        phases = np.zeros(0, dtype=np.float64)
    phase_after = phase_start + float(np.sum(rates, dtype=np.float64))

    source_samples = _resample_block(handle, plan, sample_indices, phases, input_info)
    source_samples *= gain

    ir_length = plan.ir.shape[1]
    overlap_length = ir_length - 1
    filter_order = str(state["filter_order"])
    alpha = plan.distance_filter_alpha

    if filter_order == "before" and alpha is not None:
        history = _state_vector(state, "filter_history", overlap_length, state_path)
        zi = _state_vector(state, "filter_zi", 1, state_path)
        filtered, next_zi = _lowpass_filter(source_samples, alpha, zi.reshape(1))
        context = np.concatenate((history, filtered))
        next_filter_history = (
            context[-overlap_length:].copy() if overlap_length else np.zeros(0)
        )
        next_source_history = np.zeros(overlap_length, dtype=np.float64)
    else:
        history = _state_vector(state, "source_history", overlap_length, state_path)
        context = np.concatenate((history, source_samples))
        next_source_history = (
            context[-overlap_length:].copy() if overlap_length else np.zeros(0)
        )
        next_filter_history = np.zeros(overlap_length, dtype=np.float64)
        next_zi = np.zeros(0, dtype=np.float64)

    segment = _convolution_segment(context, plan.ir)
    if filter_order == "after" and alpha is not None:
        zi = _state_vector(state, "filter_zi", 2, state_path)
        segment, next_zi = _lowpass_filter(segment, alpha, zi.reshape(1, 2))

    _append_samples(maybe_handle, segment)
    state["block_index"] = int(state["block_index"]) + 1
    state["processed_samples"] = offset + count
    state["phase"] = phase_after
    state["filter_zi"] = next_zi.astype(np.float64).ravel().tolist()
    state["filter_history"] = next_filter_history.astype(np.float64).tolist()
    state["source_history"] = next_source_history.astype(np.float64).tolist()
    return segment


def _render_sources(
    input_info: AudioInfo,
    plans: Sequence[_SourcePlan],
    listener: np.ndarray,
    work_dir: Path,
    state: Dict[str, Any],
    state_path: Path,
    speed_of_sound: float,
    stop_after_blocks: int | None,
) -> int:
    committed_blocks = 0
    handle = open_input(input_info.path)
    try:
        source_count = len(plans)
        current_index = int(state["source_index"])
        for completed in range(min(current_index, source_count)):
            completed_plan = plans[completed]
            completed_path = work_dir / f"source_{completed:04d}.f64"
            expected = (
                completed_plan.spec.delay_frames
                + input_info.frames
                + completed_plan.ir.shape[1]
                - 1
            )
            _ensure_source_file(
                completed_path, expected, state_path, completed_plan.spec.source_id
            )
        for future in range(current_index + 1, source_count):
            _safe_remove(work_dir / f"source_{future:04d}.f64")

        while state["stage"] == "sources":
            source_index = int(state["source_index"])
            if source_index >= source_count:
                state["stage"] = "mix"
                state["mix_offset"] = 0
                atomic_write_json(state_path, state)
                break
            plan = plans[source_index]
            source = plan.spec
            contribution_path = work_dir / f"source_{source_index:04d}.f64"
            offset = int(state["processed_samples"])
            block_index = int(state["block_index"])
            if offset < 0 or offset > input_info.frames:
                raise BusinessError(
                    E_STATE_INVALID,
                    "Resume state contains an invalid sample offset",
                    {"state_path": str(state_path), "processed_samples": offset},
                )
            expected_block_index = (offset + int(state["block_size"]) - 1) // int(
                state["block_size"]
            )
            if block_index != expected_block_index:
                raise BusinessError(
                    E_STATE_INVALID,
                    "Resume state block index does not match processed samples",
                    {
                        "state_path": str(state_path),
                        "block_index": block_index,
                        "expected_block_index": expected_block_index,
                    },
                )

            if offset == 0:
                _ensure_source_file(contribution_path, 0, state_path, source.source_id)
            else:
                _ensure_source_file(
                    contribution_path,
                    source.delay_frames + offset,
                    state_path,
                    source.source_id,
                )

            if offset == input_info.frames:
                with open(contribution_path, "ab") as contribution_handle:
                    _render_tail(plan, contribution_handle, state, state_path)
                _transition_to_next_source(state, source_index, state_path)
                continue

            with open(contribution_path, "ab") as contribution_handle:
                if offset == 0:
                    _write_zero_prefix(contribution_handle, source.delay_frames)
                _render_source_block(
                    plan,
                    handle,
                    input_info,
                    listener,
                    speed_of_sound,
                    state,
                    state_path,
                    contribution_handle,
                )
            atomic_write_json(state_path, state)
            committed_blocks += 1
            if stop_after_blocks is not None and committed_blocks >= stop_after_blocks:
                raise ProcessingInterrupted(
                    f"Stopped after {committed_blocks} committed source block(s)"
                )
    finally:
        handle.close()
    return committed_blocks


def _mix_sources(
    input_info: AudioInfo,
    plans: Sequence[_SourcePlan],
    work_dir: Path,
    state: Dict[str, Any],
    state_path: Path,
    output_frames: int,
    stop_after_mix_blocks: int | None,
) -> float:
    mix_path = work_dir / "mix.f64"
    mix_offset = int(state.get("mix_offset", 0))
    if mix_offset < 0 or mix_offset > output_frames:
        raise BusinessError(
            E_STATE_INVALID,
            "Resume state contains an invalid mix offset",
            {"state_path": str(state_path), "mix_offset": mix_offset},
        )
    _ensure_source_file(mix_path, mix_offset, state_path, "final mix")
    block_size = int(state["block_size"])
    peak = 0.0
    source_paths = [work_dir / f"source_{index:04d}.f64" for index in range(len(plans))]
    source_totals = [
        plan.spec.delay_frames + input_info.frames + plan.ir.shape[1] - 1
        for plan in plans
    ]
    for plan, path, total in zip(plans, source_paths, source_totals):
        _ensure_source_file(path, total, state_path, plan.spec.source_id)
    committed_blocks = 0
    with open(mix_path, "ab") as mix_handle:
        for start in range(mix_offset, output_frames, block_size):
            count = min(block_size, output_frames - start)
            accumulator = np.zeros((count, CONTRIBUTION_CHANNELS), dtype=np.float64)
            for source_index, (path, total) in enumerate(
                zip(source_paths, source_totals)
            ):
                contribution = _read_contribution(path, start, count, total)
                accumulator += contribution.astype(np.float64)
            peak = max(peak, float(np.max(np.abs(accumulator))) if count else 0.0)
            limited = np.clip(accumulator, -1.0, 1.0).astype(CONTRIBUTION_DTYPE)
            mix_handle.write(limited.tobytes())
            state["mix_offset"] = start + count
            atomic_write_json(state_path, state)
            committed_blocks += 1
            if (
                stop_after_mix_blocks is not None
                and committed_blocks >= stop_after_mix_blocks
            ):
                raise ProcessingInterrupted(
                    f"Stopped after {committed_blocks} committed mix block(s)"
                )
    return peak


def render(options: RenderOptions) -> RenderResult:
    if options.block_size <= 0:
        raise BusinessError(
            E_CONFIG_INVALID,
            "Block size must be positive",
            {"block_size": options.block_size},
        )
    if options.filter_order not in {"before", "after"}:
        raise BusinessError(
            E_CONFIG_INVALID,
            "filter_order must be 'before' or 'after'",
            {"filter_order": options.filter_order},
        )
    if options.speed_of_sound <= 0.0:
        raise BusinessError(
            E_CONFIG_INVALID,
            "Speed of sound must be positive",
            {"speed_of_sound": options.speed_of_sound},
        )

    output_format: OutputFormat = parse_output_format(options.output_format)
    input_info = probe_audio(options.input_path, hash_content=True)
    input_subtype_info = INPUT_SUBTYPES[input_info.subtype]
    validate_bit_depth(input_subtype_info, output_format)

    config = load_source_config(options.config_path, input_info.sample_rate)
    if (
        options.expected_hrtf_version is not None
        and options.expected_hrtf_version != config.hrtf_version
    ):
        raise BusinessError(
            E_HRTF_VERSION,
            "Requested HRTF dataset version is not available",
            {
                "requested": options.expected_hrtf_version,
                "available": config.hrtf_version,
            },
        )

    listener, right, up, forward = _validate_listener_inputs(options)
    hrtf = generate_hrtf(input_info.sample_rate)
    if config.hrtf_version != hrtf.version:
        raise BusinessError(
            E_HRTF_VERSION,
            "HRTF dataset version mismatch",
            {"config": config.hrtf_version, "dataset": hrtf.version},
        )

    plans = _build_source_plans(config, input_info, hrtf, listener, right, up, forward)
    output_path = Path(options.output_path).resolve()
    state_path = Path(
        options.state_path if options.state_path else str(output_path) + ".state.json"
    ).resolve()
    work_dir = Path(str(state_path) + ".work")
    if state_path == output_path:
        raise BusinessError(
            E_CONFIG_INVALID,
            "State path must not be the same as the output path",
            {"state_path": str(state_path), "output_path": str(output_path)},
        )

    result_frames = (
        input_info.frames
        + max((plan.spec.delay_frames for plan in plans), default=0)
        + hrtf.ir_length
        - 1
    )
    input_decoded_bytes = (
        input_info.frames * input_info.channels * input_info.decoded_sample_bytes
    )
    memory_limit = 2 * input_decoded_bytes
    working_estimate = _estimate_peak_working_bytes(options.block_size, hrtf.ir_length)
    if working_estimate > memory_limit:
        raise BusinessError(
            E_MEMORY_LIMIT,
            "Estimated peak audio working set exceeds twice the input sample bytes",
            {
                "estimated_peak_bytes": working_estimate,
                "limit_bytes": memory_limit,
                "input_decoded_bytes": input_decoded_bytes,
                "block_size": options.block_size,
            },
        )

    identity: Dict[str, Any] = {
        "tool_version": __version__,
        "input_sha256": input_info.sha256,
        "source_config_sha256": config.sha256,
        "block_size": options.block_size,
        "output_format": output_format.name,
        "output_path": str(output_path),
        "sample_rate": input_info.sample_rate,
        "input_frames": input_info.frames,
        "input_channels": input_info.channels,
        "input_subtype": input_info.subtype,
        "hrtf_version": hrtf.version,
        "hrtf_fingerprint": hrtf.fingerprint(),
        "num_sources": len(plans),
        "filter_order": options.filter_order,
        "speed_of_sound": options.speed_of_sound,
    }

    resumed = False
    if state_path.exists() and not options.fresh:
        state = load_state(state_path)
        validate_state_identity(state, identity, state_path)
        resumed = True
    else:
        if options.fresh:
            _safe_remove(state_path)
            _safe_remove(work_dir)
        elif work_dir.exists():
            _safe_remove(work_dir)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BusinessError(
                E_IO,
                f"Unable to create intermediate work directory: {work_dir}",
                {"path": str(work_dir), "reason": str(exc)},
            ) from exc
        state = _initial_state(identity)
        atomic_write_json(state_path, state)

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BusinessError(
            E_IO,
            f"Unable to create intermediate work directory: {work_dir}",
            {"path": str(work_dir), "reason": str(exc)},
        ) from exc

    final_temp = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    raw_mix_path_value = str(work_dir / "mix.f64")
    try:
        if state["stage"] == "sources" and state.get("source_index", 0) >= len(plans):
            state["stage"] = "mix"
            state["mix_offset"] = 0
            atomic_write_json(state_path, state)

        committed_blocks = 0
        if state["stage"] == "sources":
            committed_blocks = _render_sources(
                input_info,
                plans,
                listener,
                work_dir,
                state,
                state_path,
                options.speed_of_sound,
                options.stop_after_blocks,
            )
        elif state["stage"] != "mix":
            raise BusinessError(
                E_STATE_INVALID,
                "Resume state has an unknown processing stage",
                {"state_path": str(state_path), "stage": state["stage"]},
            )

        if state["stage"] == "mix":
            peak_abs = _mix_sources(
                input_info,
                plans,
                work_dir,
                state,
                state_path,
                result_frames,
                options.stop_after_mix_blocks,
            )
        else:
            peak_abs = 0.0

        raw_mix_path = Path(raw_mix_path_value)
        actual_raw_frames = _file_frames(raw_mix_path)
        if actual_raw_frames != result_frames:
            raise BusinessError(
                E_IO,
                "Temporary mix length does not match the planned output length",
                {"expected_frames": result_frames, "actual_frames": actual_raw_frames},
            )
        encoded_frames = encode_raw_to_wav(
            raw_mix_path,
            final_temp,
            input_info.sample_rate,
            output_format.subtype,
            options.block_size,
            sample_dtype=CONTRIBUTION_DTYPE,
        )
        if encoded_frames != result_frames:
            raise BusinessError(
                E_IO,
                "Encoded WAV frame count mismatch",
                {"expected_frames": result_frames, "actual_frames": encoded_frames},
            )
        atomic_replace(final_temp, output_path)
        _safe_remove(state_path)
        _safe_remove(work_dir)
    except ProcessingInterrupted:
        _safe_remove(final_temp)
        raise
    except KeyboardInterrupt:
        _safe_remove(final_temp)
        raise
    except BusinessError:
        _safe_remove(final_temp)
        raise
    except Exception:
        _safe_remove(final_temp)
        raise

    return RenderResult(
        output_path=str(output_path),
        frames=result_frames,
        sample_rate=input_info.sample_rate,
        output_format=output_format.name,
        block_size=options.block_size,
        source_count=len(plans),
        peak_abs=float(peak_abs),
        resumed=resumed,
        committed_blocks=committed_blocks,
        peak_working_bytes_estimate=working_estimate,
        input_decoded_bytes=input_decoded_bytes,
        hrtf=hrtf.metadata(),
    )
