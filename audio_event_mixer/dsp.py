"""Offline DSP primitives: windowed-sinc resampling, panning, loop cleanup."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_SINC_HALF_WIDTH = 24
DEFAULT_KAISER_BETA = 8.6


def _kaiser_window(normalized_distance: np.ndarray, beta: float) -> np.ndarray:
    """Kaiser window evaluated at arbitrary (vectorized) positions."""

    inside = np.abs(normalized_distance) < 1.0
    window = np.zeros_like(normalized_distance, dtype=np.float64)
    if np.any(inside):
        window[inside] = np.i0(
            beta * np.sqrt(1.0 - normalized_distance[inside] ** 2)
        ) / np.i0(beta)
    return window


def _reflect_indices(indices: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    period = 2 * length - 2
    reflected = np.mod(indices, period)
    return np.where(reflected >= length, period - reflected, reflected)


def _resample_geometry(
    source_frames: int,
    source_rate: int,
    target_rate: int,
    half_width: int,
) -> tuple[int, float, float, np.ndarray]:
    target_frames = int(
        math.floor(source_frames * target_rate / source_rate + 0.5)
    )
    ratio = source_rate / float(target_rate)
    cutoff = min(1.0, target_rate / float(source_rate)) * 0.97
    radius = max(half_width, int(math.ceil(half_width / cutoff)))
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    return target_frames, ratio, cutoff, offsets


def resample_into(
    source: np.ndarray,
    source_rate: int,
    target_rate: int,
    destination: np.ndarray,
    *,
    half_width: int = DEFAULT_SINC_HALF_WIDTH,
    beta: float = DEFAULT_KAISER_BETA,
    output_block_frames: int = 16384,
) -> int:
    """Windowed-sinc resample a mono signal into a preallocated array.

    The filter is a Kaiser-windowed sinc normalized per output sample. Edge
    samples use symmetric reflection so a constant signal remains constant
    and resampling introduces no DC offset.
    """

    x = np.asarray(source)
    if x.ndim != 1:
        raise ValueError("source must be mono")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if half_width <= 0 or output_block_frames <= 0:
        raise ValueError("half_width and output_block_frames must be positive")

    target_frames, ratio, cutoff, offsets = _resample_geometry(
        x.shape[0], source_rate, target_rate, half_width
    )
    out = destination
    if out.ndim != 1 or out.shape[0] != target_frames:
        raise ValueError("destination has the wrong shape")
    if target_frames == 0:
        return 0
    if source_rate == target_rate:
        np.copyto(out, x, casting="unsafe")
        return target_frames
    if x.shape[0] == 1:
        out.fill(float(x[0]))
        return target_frames

    radius = max(half_width, int(math.ceil(half_width / cutoff)))
    taps = offsets.size
    bytes_per_block = 16 * 1024 * 1024
    bounded_block = max(1, min(output_block_frames, bytes_per_block // (taps * 8)))
    source_length = x.shape[0]
    positions = np.arange(bounded_block, dtype=np.float64)

    for block_start in range(0, target_frames, bounded_block):
        block_count = min(bounded_block, target_frames - block_start)
        local_positions = positions[:block_count]
        position = (block_start + local_positions) * ratio
        base = np.floor(position).astype(np.int64)
        source_indices = base[:, None] + offsets[None, :]
        distance = position[:, None] - source_indices
        weights = cutoff * np.sinc(cutoff * distance)
        weights *= _kaiser_window(distance / radius, beta)
        row_sum = weights.sum(axis=1)
        row_sum[np.abs(row_sum) < 1e-15] = 1.0
        weights /= row_sum[:, None]
        reflected = _reflect_indices(source_indices, source_length)
        out[block_start : block_start + block_count] = np.sum(
            np.asarray(x)[reflected] * weights, axis=1
        )
        if isinstance(out, np.memmap):
            out.flush()
    return target_frames


def resample_windowed_sinc(
    source: np.ndarray,
    source_rate: int,
    target_rate: int,
    *,
    half_width: int = DEFAULT_SINC_HALF_WIDTH,
    beta: float = DEFAULT_KAISER_BETA,
) -> np.ndarray:
    """Convenience in-memory wrapper around :func:`resample_into`."""

    x = np.asarray(source, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("source must be mono")
    target_frames = int(
        math.floor(x.shape[0] * target_rate / source_rate + 0.5)
    )
    destination = np.empty(target_frames, dtype=np.float64)
    resample_into(
        x,
        source_rate,
        target_rate,
        destination,
        half_width=half_width,
        beta=beta,
    )
    return destination


def panning_gains(
    frame_count: int,
    sample_rate: int,
    frequency_hz: float,
    depth: float,
    phase_degrees: float = 0.0,
    start_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-power left/right gains for a sine panning LFO.

    The normalized pan position is ``depth * sin(2*pi*f*t + phase)`` and is
    mapped to ``theta = (pan + 1) * pi / 4``. Thus depth zero always yields
    equal center gains, and ``left**2 + right**2 == 1`` for every frame.
    """

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frequency_hz < 0.0:
        raise ValueError("frequency_hz must be non-negative")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be within [0, 1]")

    if depth == 0.0:
        center = np.full(frame_count, math.sqrt(0.5), dtype=np.float64)
        return center.copy(), center

    indices = np.arange(frame_count, dtype=np.float64) + float(start_frame)
    phase = math.radians(phase_degrees)
    lfo = depth * np.sin(
        2.0 * math.pi * frequency_hz * indices / sample_rate + phase
    )
    theta = (lfo + 1.0) * math.pi / 4.0
    return np.cos(theta), np.sin(theta)


def pan_mono(
    mono: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    depth: float,
    phase_degrees: float = 0.0,
    start_frame: int = 0,
) -> np.ndarray:
    """Pan a mono float64 signal into an equal-power stereo array."""

    source = np.asarray(mono, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("mono must be one-dimensional")
    left_gain, right_gain = panning_gains(
        source.shape[0],
        sample_rate,
        frequency_hz,
        depth,
        phase_degrees,
        start_frame,
    )
    stereo = np.empty((source.shape[0], 2), dtype=np.float64)
    stereo[:, 0] = source * left_gain
    stereo[:, 1] = source * right_gain
    return stereo


def crossfade_sample_count(loop_frames: int, sample_rate: int) -> int:
    """floor(min(10% of loop length, 5 ms)) in integer sample points."""

    if loop_frames <= 0 or sample_rate <= 0:
        return 0
    ten_percent = math.floor(loop_frames * 0.10)
    five_ms = math.floor(sample_rate * 0.005)
    return min(ten_percent, five_ms)


def equal_power_tail_crossfade(
    tail: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    """Crossfade a loop tail into a rotated copy of the loop head.

    The outgoing tail uses ``cos`` and the incoming head uses ``sin`` over an
    equal-power fade. Rotating the head by one sample makes the final tail
    sample exactly equal to the first loop sample, so wrapping the cleaned
    region has zero sample-value jump while preserving its frame count.
    """

    tail_array = np.asarray(tail, dtype=np.float64)
    head_array = np.asarray(head, dtype=np.float64)
    if tail_array.ndim != 1 or head_array.ndim != 1:
        raise ValueError("tail and head must be one-dimensional")
    if tail_array.shape != head_array.shape or tail_array.size == 0:
        raise ValueError("tail and head must have the same non-zero length")

    count = tail_array.size
    if count == 1:
        return head_array.copy()
    position = np.arange(count, dtype=np.float64) / (count - 1)
    incoming = np.sin(position * math.pi / 2.0)
    outgoing = np.cos(position * math.pi / 2.0)
    target = np.roll(head_array, -1)
    return tail_array * outgoing + target * incoming


def loop_crossfade_block(
    tail: np.ndarray,
    head: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return (crossfaded tail, jump before, jump after)."""

    tail_array = np.asarray(tail, dtype=np.float64)
    head_array = np.asarray(head, dtype=np.float64)
    if tail_array.size == 0 or head_array.size == 0:
        raise ValueError("tail and head must not be empty")
    before = float(abs(tail_array[-1] - head_array[0]))
    blended = equal_power_tail_crossfade(tail_array, head_array)
    after = float(abs(blended[-1] - head_array[0]))
    return blended, before, after


@dataclass(frozen=True)
class LoopCleanupResult:
    samples: np.ndarray
    start_sample: int
    end_sample: int
    crossfade_samples: int
    boundary_jump_before: float
    boundary_jump_after: float


def apply_loop_cleanup(
    mono: np.ndarray,
    sample_rate: int,
    loop_start_sample: int,
    loop_end_sample: int,
) -> LoopCleanupResult:
    """Apply a frame-preserving loop tail/head crossfade."""

    source = np.asarray(mono, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("mono must be one-dimensional")
    start = int(loop_start_sample)
    end = int(loop_end_sample)
    if not 0 <= start < end <= source.shape[0]:
        raise ValueError("loop region is outside the audio")

    output = source.copy()
    fade = crossfade_sample_count(end - start, sample_rate)
    if fade == 0:
        return LoopCleanupResult(
            samples=output,
            start_sample=start,
            end_sample=end,
            crossfade_samples=0,
            boundary_jump_before=float(abs(output[end - 1] - output[start])),
            boundary_jump_after=float(abs(output[end - 1] - output[start])),
        )

    tail = output[end - fade : end].copy()
    head = output[start : start + fade].copy()
    blended, before, after = loop_crossfade_block(tail, head)
    output[end - fade : end] = blended
    return LoopCleanupResult(
        samples=output,
        start_sample=start,
        end_sample=end,
        crossfade_samples=fade,
        boundary_jump_before=before,
        boundary_jump_after=after,
    )
