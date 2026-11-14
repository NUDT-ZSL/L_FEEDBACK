"""Deterministic look-ahead sample-peak limiter."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


LOOKAHEAD_SECONDS = 0.005
RELEASE_SECONDS = 0.050
@dataclass(frozen=True)
class LimiterResult:
    samples: np.ndarray
    max_reduction_db: float
    limited_sample_count: int
    maximum_overshoot: float


def _future_minimum(values: np.ndarray, width: int) -> np.ndarray:
    """Sliding minimum over ``[index, index + width)`` in linear time."""

    count = values.shape[0]
    result = np.empty(count, dtype=np.float64)
    deque_indices = [0] * count
    head = 0
    tail = 0
    next_to_add = 0
    for index in range(count):
        limit = min(count, index + width)
        while next_to_add < limit:
            while tail > head and values[deque_indices[tail - 1]] >= values[next_to_add]:
                tail -= 1
            deque_indices[tail] = next_to_add
            tail += 1
            next_to_add += 1
        while deque_indices[head] < index:
            head += 1
        result[index] = values[deque_indices[head]]
    return result


def limit_peaks(
    samples: np.ndarray,
    sample_rate: int,
    ceiling: float = 1.0,
    lookahead_seconds: float = LOOKAHEAD_SECONDS,
    release_seconds: float = RELEASE_SECONDS,
) -> LimiterResult:
    """Limit all channels with one linked gain envelope.

    The envelope never rises above the gain required by any upcoming sample
    inside the look-ahead window. Downward changes are therefore anticipated;
    release is gradual. A final numerical clamp guarantees no sample exceeds
    the ceiling, and the reported maximum overshoot remains exactly zero.
    """

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("samples must be a two-dimensional array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not (0.0 < ceiling <= 1.0):
        raise ValueError("ceiling must be in (0, 1]")
    if audio.shape[0] == 0:
        return LimiterResult(audio.copy(), 0.0, 0, 0.0)

    peak_per_frame = np.max(np.abs(audio), axis=1)
    target_gain = np.ones_like(peak_per_frame)
    nonzero = peak_per_frame > 0.0
    target_gain[nonzero] = ceiling / peak_per_frame[nonzero]
    np.minimum(target_gain, 1.0, out=target_gain)

    lookahead_frames = max(1, int(round(lookahead_seconds * sample_rate)))
    anticipated_gain = _future_minimum(target_gain, lookahead_frames)

    release_samples = max(1.0, release_seconds * sample_rate)
    release_per_sample = 1.0 - np.exp(-1.0 / release_samples)
    envelope = np.empty_like(target_gain)
    previous = 1.0
    for index in range(target_gain.shape[0]):
        target = float(anticipated_gain[index])
        if target <= previous:
            current = target
        else:
            current = min(target, previous + (1.0 - previous) * release_per_sample)
        envelope[index] = current
        previous = current

    limited = audio * envelope[:, None]
    maximum_overshoot = max(0.0, float(np.max(np.abs(limited)) - ceiling))
    limited = np.clip(limited, -ceiling, ceiling)

    max_reduction_db = 0.0
    if np.any(target_gain < 1.0):
        minimum_gain = float(np.min(target_gain))
        if minimum_gain > 0.0:
            max_reduction_db = max(0.0, -20.0 * np.log10(minimum_gain))
    limited_sample_count = int(np.count_nonzero(target_gain < 1.0))
    return LimiterResult(
        samples=limited,
        max_reduction_db=max_reduction_db,
        limited_sample_count=limited_sample_count,
        maximum_overshoot=maximum_overshoot,
    )
