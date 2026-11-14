from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Sequence, Tuple

import numpy as np


INTERVAL = Tuple[int, int]


def milliseconds_to_samples(value: float, sample_rate: int) -> int:
    milliseconds = Decimal(str(value))
    samples = milliseconds * Decimal(sample_rate) / Decimal(1000)
    return int(samples.to_integral_value(rounding=ROUND_HALF_UP))


def samples_to_milliseconds(samples: int, sample_rate: int) -> float:
    return round(samples * 1000.0 / sample_rate, 6)


def amplitude_to_dbfs(amplitude: float) -> Optional[float]:
    if amplitude <= 0.0:
        return None
    return 20.0 * math.log10(amplitude)


def cosine_squared_fade_in(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if length == 1:
        return np.zeros(1, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, length, dtype=np.float64)
    curve = np.cos(0.5 * math.pi * (1.0 - positions)) ** 2
    curve[0] = 0.0
    curve[-1] = 1.0
    return curve


def cosine_squared_fade_out(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if length == 1:
        return np.zeros(1, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, length, dtype=np.float64)
    curve = np.cos(0.5 * math.pi * positions) ** 2
    curve[0] = 1.0
    curve[-1] = 0.0
    return curve


def apply_cosine_squared_fades(
    samples: np.ndarray,
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray:
    frame_count = samples.shape[0]
    if frame_count == 0:
        return samples.copy()

    gains = np.ones(frame_count, dtype=np.float64)
    fade_in = min(max(fade_in_samples, 0), frame_count)
    fade_out = min(max(fade_out_samples, 0), frame_count)
    if fade_in:
        gains[:fade_in] = np.minimum(
            gains[:fade_in], cosine_squared_fade_in(fade_in)
        )
    if fade_out:
        gains[frame_count - fade_out:] = np.minimum(
            gains[frame_count - fade_out:],
            cosine_squared_fade_out(fade_out),
        )
    return samples * gains[:, None]


def fade_region_peak(
    samples: np.ndarray,
    fade_in_samples: int,
    fade_out_samples: int,
) -> float:
    frame_count = samples.shape[0]
    peak = 0.0
    fade_in = min(max(fade_in_samples, 0), frame_count)
    fade_out = min(max(fade_out_samples, 0), frame_count)
    if fade_in:
        peak = max(peak, float(np.max(np.abs(samples[:fade_in]))))
    if fade_out:
        peak = max(
            peak, float(np.max(np.abs(samples[frame_count - fade_out:])))
        )
    return peak


def find_placement(
    requested_start: int,
    frame_count: int,
    timeline_frames: int,
    placed_intervals: Sequence[INTERVAL],
) -> Optional[int]:
    """Find the earliest non-overlapping start at or after the request."""

    if frame_count <= 0 or frame_count > timeline_frames:
        return None
    if requested_start < 0:
        return None

    candidate = requested_start
    for start, end in sorted(placed_intervals):
        if end <= candidate:
            continue
        if start < candidate + frame_count:
            candidate = end
        if candidate + frame_count > timeline_frames:
            return None
    if candidate + frame_count > timeline_frames:
        return None
    return candidate
