"""Band-limited fractional-delay resampling used for alignment."""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass

import numpy as np

from .errors import ConfigError
from .resources import ResourceGovernor
from .timeline import AlignmentFit, SourceRoute
from .wav import WavInfo, read_frames


@dataclass(frozen=True)
class SincConfig:
    half_width: int = 16
    max_half_width: int = 64
    beta: float = 8.0

    def __post_init__(self) -> None:
        if self.half_width < 4:
            raise ConfigError("resampler half_width must be >= 4")
        if self.max_half_width < self.half_width:
            raise ConfigError("resampler max_half_width must be >= half_width")


def _bessel_i0(x: np.ndarray | float) -> np.ndarray | float:
    """Abramowitz-Stegun I0 approximation (relative error below 1e-7)."""

    values = np.asarray(x, dtype=np.float64)
    absolute = np.abs(values)
    small_arg = (values / 3.75) ** 2
    small = 1.0 + small_arg * (
        3.5156229
        + small_arg
        * (
            3.0899424
            + small_arg
            * (
                1.2067492
                + small_arg * (0.2659732 + small_arg * (0.0360768 + small_arg * 0.0045813))
            )
        )
    )
    safe_absolute = np.maximum(absolute, 1.0e-12)
    reciprocal = 3.75 / safe_absolute
    large = (
        np.exp(safe_absolute)
        / np.sqrt(safe_absolute)
        * (
            0.39894228
            + reciprocal
            * (
                0.01328592
                + reciprocal
                * (
                    0.00225319
                    + reciprocal
                    * (
                        -0.00157565
                        + reciprocal
                        * (
                            0.00916281
                            + reciprocal
                            * (
                                -0.02057706
                                + reciprocal
                                * (
                                    0.02635537
                                    + reciprocal
                                    * (-0.01647633 + reciprocal * 0.00392377)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    result = np.where(absolute <= 3.75, small, large)
    if np.ndim(result) == 0:
        return float(result)
    return result


@lru_cache(maxsize=16)
def _kaiser_table(beta: float, size: int = 8192):
    grid = np.linspace(0.0, 1.0, size + 1, dtype=np.float64)
    table = np.asarray(_bessel_i0(beta * grid) / _bessel_i0(beta), dtype=np.float64)
    grid.flags.writeable = False
    table.flags.writeable = False
    return grid, table


def _effective_half_width(
    config: SincConfig, source_rate: int, output_rate: int, drift: float
) -> int:
    source_per_output = source_rate / (output_rate * drift)
    scaled = int(math.ceil(config.half_width * max(1.0, source_per_output)))
    return min(config.max_half_width, max(config.half_width, scaled))


def render_track_chunk(
    *,
    info: WavInfo,
    routes: list[tuple[SourceRoute, ...]],
    alignment: AlignmentFit,
    output_start_frame: int,
    output_frames: int,
    output_sample_rate: int,
    source_frames: int,
    governor: ResourceGovernor,
    config: SincConfig = SincConfig(),
) -> np.ndarray:
    """Render one program chunk for one track.

    The mapping is ``program = offset + drift * source``; therefore each output
    frame is first mapped to a fractional source position and then reconstructed
    with a Kaiser-windowed sinc kernel.  Downsampling lowers the kernel cutoff
    to prevent aliasing.  Source frames outside the WAV are represented by
    zeros, which is also the local-degradation placeholder.
    """

    if output_frames <= 0:
        return np.empty((0, len(routes)), dtype=np.float64)
    if not routes:
        raise ConfigError("track has no output routes")
    source_channels = info.channels
    for group in routes:
        for route in group:
            if route.channel < 0 or route.channel >= source_channels:
                raise ConfigError(
                    f"source channel {route.channel} is out of range for "
                    f"{info.path} ({source_channels} channel(s))"
                )

    program_seconds = (
        np.arange(output_start_frame, output_start_frame + output_frames, dtype=np.float64)
        / output_sample_rate
    )
    source_seconds = (
        program_seconds - alignment.offset_seconds
    ) / alignment.drift_factor
    source_positions = source_seconds * info.sample_rate

    half_width = _effective_half_width(
        config, info.sample_rate, output_sample_rate, alignment.drift_factor
    )
    taps = half_width * 2
    source_to_output_ratio = info.sample_rate / (
        output_sample_rate * alignment.drift_factor
    )
    cutoff = 0.5 * min(1.0, 1.0 / source_to_output_ratio)
    cutoff = max(1.0e-5, min(0.5, cutoff))

    base_indices = np.floor(source_positions).astype(np.int64) - half_width + 1
    window_start = int(base_indices[0])
    window_end = int(base_indices[-1] + taps)
    window_length = max(0, window_end - window_start)
    valid_start = max(0, window_start)
    valid_end = min(source_frames, window_end)
    valid_count = max(0, valid_end - valid_start)
    estimated_bytes = (
        window_length * source_channels * 8
        + output_frames * len(routes) * 8
        + output_frames * taps * 8 * 6
        + output_frames * taps * 8
    )
    # The estimate includes decoded source samples and all transient kernel
    # buffers.  Allocations happen only after this budget is reserved.
    with governor.memory(estimated_bytes), governor.wav_handle():
        offsets = np.arange(taps, dtype=np.float64)
        delta = source_positions[:, None] - (
            base_indices[:, None] + offsets[None, :]
        )
        window_arg = delta / float(half_width)
        # A cached Kaiser table permits fractional sample positions while
        # retaining the configured beta.  It is exactly zero at ±half_width.
        radial = np.sqrt(np.maximum(0.0, 1.0 - np.square(window_arg)))
        kaiser_grid, kaiser_values = _kaiser_table(float(config.beta))
        window = np.interp(radial, kaiser_grid, kaiser_values)
        kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * delta) * window
        row_sums = np.sum(kernel, axis=1, keepdims=True)
        kernel = np.divide(
            kernel,
            row_sums,
            out=np.zeros_like(kernel),
            where=np.abs(row_sums) > 1.0e-15,
        )
        samples, actual_start = _read_window(
            info, valid_start, valid_count, governor=None
        )
        padded = np.zeros((window_length, source_channels), dtype=np.float64)
        if actual_start != valid_start and len(samples):
            raise AssertionError("unexpected clamped read start")
        if len(samples):
            begin = valid_start - window_start
            padded[begin : begin + len(samples), :] = samples

        relative_indices = base_indices - window_start
        gather = relative_indices[:, None] + np.arange(taps, dtype=np.int64)[None, :]
        output = np.empty((output_frames, len(routes)), dtype=np.float64)
        for channel_index, group in enumerate(routes):
            combined = np.zeros(window_length, dtype=np.float64)
            for route in group:
                combined += padded[:, route.channel] * route.gain
            output[:, channel_index] = np.sum(combined[gather] * kernel, axis=1)
    return output


def _read_window(
    info: WavInfo,
    start: int,
    count: int,
    governor: ResourceGovernor | None,
) -> tuple[np.ndarray, int]:
    if count <= 0:
        return np.empty((0, info.channels), dtype=np.float64), start
    actual_start, samples = read_frames(info, start, count, governor=governor)
    return samples, actual_start

