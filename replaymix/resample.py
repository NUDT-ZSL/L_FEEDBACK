"""Deterministic fixed-window (two-tap) linear resampling to 48 kHz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timeline import INTERNAL_SAMPLE_RATE

RESAMPLER_VERSION = "linear-2tap-integer-rational-v1"


@dataclass(frozen=True)
class ResampleFit:
    natural_frames: int
    target_frames: int
    zero_padded_frames: int
    trimmed_frames: int
    frame_delta: int


def natural_output_frames(source_frames: int, source_rate: int) -> int:
    if source_frames < 0 or source_rate <= 0:
        raise ValueError("source_frames must be non-negative and source_rate positive")
    return (source_frames * INTERNAL_SAMPLE_RATE) // source_rate


def resample_linear_to_48k(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample with a fixed two-tap linear kernel.

    Source positions are represented as exact integer rationals:
    ``pos_num = i * source_rate`` and ``pos = pos_num / 48000``.
    This avoids platform-dependent floating-point position accumulation.
    """

    source = np.asarray(samples, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("resampler expects a mono one-dimensional array")
    if source_rate <= 0:
        raise ValueError("source_rate must be positive")
    if source.size == 0:
        return np.zeros(0, dtype=np.float64)
    if source_rate == INTERNAL_SAMPLE_RATE:
        return source.copy()

    output_frames = natural_output_frames(source.size, source_rate)
    if output_frames == 0:
        return np.zeros(0, dtype=np.float64)

    index = np.arange(output_frames, dtype=np.int64)
    position_numerator = index * np.int64(source_rate)
    left = position_numerator // INTERNAL_SAMPLE_RATE
    remainder = position_numerator % INTERNAL_SAMPLE_RATE
    fraction = remainder.astype(np.float64) / float(INTERNAL_SAMPLE_RATE)
    right = np.minimum(left + 1, source.size - 1)
    return source[left] + (source[right] - source[left]) * fraction


def fit_to_target(
    samples: np.ndarray, target_frames: int
) -> tuple[np.ndarray, ResampleFit]:
    natural_frames = int(samples.size)
    if target_frames < 0:
        raise ValueError("target_frames must be non-negative")
    delta = natural_frames - target_frames
    if natural_frames < target_frames:
        padded = np.concatenate(
            [samples, np.zeros(target_frames - natural_frames, dtype=np.float64)]
        )
        fit = ResampleFit(
            natural_frames=natural_frames,
            target_frames=target_frames,
            zero_padded_frames=target_frames - natural_frames,
            trimmed_frames=0,
            frame_delta=delta,
        )
        return padded, fit
    if natural_frames > target_frames:
        trimmed = samples[:target_frames].copy()
        fit = ResampleFit(
            natural_frames=natural_frames,
            target_frames=target_frames,
            zero_padded_frames=0,
            trimmed_frames=natural_frames - target_frames,
            frame_delta=delta,
        )
        return trimmed, fit
    return samples.copy(), ResampleFit(
        natural_frames=natural_frames,
        target_frames=target_frames,
        zero_padded_frames=0,
        trimmed_frames=0,
        frame_delta=0,
    )
