"""ITU-R BS.1770 integrated loudness and peak-safe normalization."""

from __future__ import annotations

import functools
import math
from collections import deque

import numpy as np


SAMPLE_RATE = 48_000
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1

# ITU-R BS.1770-4 K-weighting coefficients at 48 kHz.
_SHELF_B = np.array(
    [1.53512485958697, -2.69169618940638, 1.19839281085285],
    dtype=np.float64,
)
_SHELF_A = np.array(
    [1.0, -1.69065929318241, 0.73248077421585],
    dtype=np.float64,
)
_HIGHPASS_B = np.array([1.0, -2.0, 1.0], dtype=np.float64)
_HIGHPASS_A = np.array(
    [1.0, -1.99004745483398, 0.99007225036621],
    dtype=np.float64,
)


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _power_to_lufs(power: float) -> float:
    if not math.isfinite(power) or power <= 0.0:
        return -math.inf
    return -0.691 + 10.0 * math.log10(power)


@functools.lru_cache(maxsize=4)
def _biquad_impulse_response(
    b: tuple[float, float, float],
    a: tuple[float, float, float],
    length: int = 16_384,
) -> np.ndarray:
    """Return a very long truncated IIR impulse response.

    The K-weighting poles decay much faster than 16k samples.  Convolving with
    this response avoids a Python sample loop while staying within NumPy-only
    runtime dependencies.
    """

    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    output = np.zeros(length, dtype=np.float64)
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    for index in range(length):
        x0 = impulse[index]
        x1 = impulse[index - 1] if index >= 1 else 0.0
        x2 = impulse[index - 2] if index >= 2 else 0.0
        y1 = output[index - 1] if index >= 1 else 0.0
        y2 = output[index - 2] if index >= 2 else 0.0
        output[index] = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    return output


def _fft_convolve_same(x: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    sample_count = x.shape[0]
    fft_size = 1 << (sample_count + impulse.size - 2).bit_length()
    spectrum = np.fft.rfft(x, n=fft_size, axis=0)
    impulse_spectrum = np.fft.rfft(impulse, n=fft_size)
    filtered = np.fft.irfft(
        spectrum * impulse_spectrum[:, np.newaxis],
        n=fft_size,
        axis=0,
    )
    return filtered[:sample_count]


def _biquad_filter(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    impulse = _biquad_impulse_response(
        tuple(float(value) for value in b),
        tuple(float(value) for value in a),
    )
    return _fft_convolve_same(x, impulse)


def k_weight(x: np.ndarray) -> np.ndarray:
    """Apply the two-stage BS.1770 K-weighting filter."""

    working = np.asarray(x, dtype=np.float64)
    if working.ndim == 1:
        working = working[:, np.newaxis]
    if working.shape[0] == 0:
        return working.copy()
    once = _biquad_filter(working, _SHELF_B, _SHELF_A)
    return _biquad_filter(once, _HIGHPASS_B, _HIGHPASS_A)


def integrated_loudness(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Measure gated integrated loudness in LUFS.

    Segments shorter than 400 ms use one complete measurement block.  This is
    intentional: timeline fragments can be shorter than the BS.1770 block size,
    and still need a stable report value.
    """

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"loudness measurement expects {SAMPLE_RATE} Hz, got {sample_rate}"
        )
    working = np.asarray(x, dtype=np.float64)
    if working.ndim == 1:
        working = working[:, np.newaxis]
    if working.size == 0:
        return -math.inf

    filtered = k_weight(working)
    channel_energy = np.sum(filtered * filtered, axis=1)
    sample_count = channel_energy.shape[0]
    block_size = int(round(BLOCK_SECONDS * sample_rate))
    hop_size = int(round(HOP_SECONDS * sample_rate))

    if sample_count < block_size:
        block_powers = np.array(
            [float(np.mean(channel_energy))], dtype=np.float64
        )
    else:
        cumulative = np.concatenate(
            (np.array([0.0]), np.cumsum(channel_energy, dtype=np.float64))
        )
        starts = np.arange(0, sample_count - block_size + 1, hop_size)
        block_powers = (
            cumulative[starts + block_size] - cumulative[starts]
        ) / block_size

    if block_powers.size == 0 or not np.any(block_powers > 0.0):
        return -math.inf
    block_loudness = np.array(
        [_power_to_lufs(float(power)) for power in block_powers],
        dtype=np.float64,
    )
    absolute_mask = block_loudness >= ABSOLUTE_GATE_LUFS
    if not np.any(absolute_mask):
        return -math.inf

    absolute_power = float(np.mean(block_powers[absolute_mask]))
    absolute_loudness = _power_to_lufs(absolute_power)
    relative_threshold = absolute_loudness + RELATIVE_GATE_LU
    relative_mask = absolute_mask & (
        block_loudness >= relative_threshold
    )
    if not np.any(relative_mask):
        return -math.inf
    gated_power = float(np.mean(block_powers[relative_mask]))
    return _power_to_lufs(gated_power)


def _sliding_future_min(values: np.ndarray, lookahead: int) -> np.ndarray:
    """Minimum of values[i:i+lookahead+1], inclusive."""

    count = values.size
    result = np.empty(count, dtype=np.float64)
    queue: deque[int] = deque()

    def push(index: int) -> None:
        while queue and values[queue[-1]] >= values[index]:
            queue.pop()
        queue.append(index)

    for index in range(min(lookahead + 1, count)):
        push(index)
    for index in range(count):
        next_index = index + lookahead + 1
        if next_index < count:
            push(next_index)
        while queue and queue[0] < index:
            queue.popleft()
        result[index] = values[queue[0]] if queue else 1.0
    return result


def limit_peak(
    x: np.ndarray,
    ceiling: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply a short-lookahead gain limiter followed by a hard safety clamp."""

    working = np.asarray(x, dtype=np.float64)
    if working.size == 0:
        return working.copy()
    per_sample_peak = np.max(np.abs(working), axis=1)
    if float(np.max(per_sample_peak)) <= ceiling:
        return working.copy()

    required_gain = np.minimum(
        1.0,
        ceiling / np.maximum(per_sample_peak, np.finfo(np.float64).tiny),
    )
    lookahead = max(1, int(round(0.005 * sample_rate)))
    target_gain = _sliding_future_min(required_gain, lookahead)

    release = math.exp(-1.0 / (0.020 * sample_rate))
    envelope = np.empty_like(target_gain)
    current = 1.0
    for index, target in enumerate(target_gain):
        if target < current:
            current = target
        else:
            current = release * current + (1.0 - release) * target
        envelope[index] = current

    limited = working * envelope[:, np.newaxis]
    return np.clip(limited, -ceiling, ceiling)


def normalize_segment(
    x: np.ndarray,
    *,
    target_lufs: float = -23.0,
    peak_dbfs: float = -1.0,
    sample_rate: int = SAMPLE_RATE,
    tolerance_lu: float = 0.10,
) -> tuple[np.ndarray, float]:
    """Normalize one segment while respecting its peak ceiling.

    Normal linear gain is used whenever possible.  For high-crest audio whose
    peak ceiling prevents reaching the loudness target, a lookahead limiter is
    engaged and the gain is iteratively corrected.
    """

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy(), -math.inf
    measured = integrated_loudness(x, sample_rate=sample_rate)
    if not math.isfinite(measured):
        raise ValueError("cannot normalize a silent or unmeasurably quiet segment")

    # A tiny extra safety margin keeps PCM16 rounding at or below the requested
    # ceiling in the final report.
    ceiling = _db_to_linear(peak_dbfs - 0.01)
    gain = _db_to_linear(target_lufs - measured)
    best = x.copy()
    best_loudness = measured
    best_error = abs(measured - target_lufs)

    for _ in range(8):
        candidate = limit_peak(x * gain, ceiling, sample_rate=sample_rate)
        candidate_loudness = integrated_loudness(
            candidate, sample_rate=sample_rate
        )
        error = abs(candidate_loudness - target_lufs)
        if math.isfinite(candidate_loudness) and error < best_error:
            best = candidate
            best_loudness = candidate_loudness
            best_error = error
        if math.isfinite(candidate_loudness) and error <= tolerance_lu:
            return candidate, candidate_loudness
        if not math.isfinite(candidate_loudness):
            break
        correction_db = target_lufs - candidate_loudness
        gain *= _db_to_linear(correction_db)
        gain = float(np.clip(gain, 1.0e-8, 1.0e6))

    return best, best_loudness

