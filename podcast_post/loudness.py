"""ITU-R BS.1770 integrated loudness and true-peak measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import BusinessError


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float
    true_peak_dbtp: float


@dataclass(frozen=True)
class NormalizationResult:
    samples: np.ndarray
    input_lufs: float
    input_true_peak_dbtp: float
    gain_db: float
    final_lufs: float
    final_true_peak_dbtp: float
    peak_limited: bool


# BS.1770 K-weighting coefficients for 48 kHz.
_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_HIGHPASS_B = np.array([1.0, -2.0, 1.0])
_HIGHPASS_A = np.array([1.0, -1.99004745483398, 0.99007225036621])


def _biquad(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Causal transposed direct-form II biquad for one channel."""

    result = np.empty_like(signal, dtype=np.float64)
    z1 = 0.0
    z2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for index in range(signal.shape[0]):
        value = float(signal[index])
        output = b0 * value + z1
        z1 = b1 * value - a1 * output + z2
        z2 = b2 * value - a2 * output
        result[index] = output
    return result


def _k_weight(samples: np.ndarray) -> np.ndarray:
    filtered = np.empty_like(samples, dtype=np.float64)
    for channel in range(samples.shape[1]):
        shelf = _biquad(samples[:, channel], _SHELF_B, _SHELF_A)
        filtered[:, channel] = _biquad(shelf, _HIGHPASS_B, _HIGHPASS_A)
    return filtered


def _block_energies(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    block_samples = int(round(0.400 * sample_rate))
    step_samples = int(round(0.100 * sample_rate))
    if samples.shape[0] < block_samples:
        raise BusinessError(
            "E_SILENCE",
            f"audio must contain at least one 400 ms block; got {samples.shape[0]} samples",
        )
    starts = np.arange(0, samples.shape[0] - block_samples + 1, step_samples)
    weights = np.ones(samples.shape[1], dtype=np.float64)
    channel_power = np.mean(np.square(samples), axis=1)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(channel_power)))
    block_power = (cumulative[starts + block_samples] - cumulative[starts]) / block_samples
    # Only the five main channels carry 1.41 weighting, so mono/stereo use 1.0.
    return block_power * float(np.sum(weights))


def integrated_loudness(samples: np.ndarray, sample_rate: int) -> float:
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise BusinessError("E_SILENCE", "cannot measure empty audio")
    weighted = _k_weight(samples)
    energies = _block_energies(weighted, sample_rate)
    with np.errstate(divide="ignore"):
        block_loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, 1.0e-30))
    absolute = block_loudness > -70.0
    if not np.any(absolute):
        raise BusinessError("E_SILENCE", "audio is below the BS.1770 absolute gate")
    absolute_mean = float(np.mean(energies[absolute]))
    relative_gate = -0.691 + 10.0 * math.log10(absolute_mean) - 10.0
    gated = absolute & (block_loudness > relative_gate)
    if not np.any(gated):
        raise BusinessError("E_SILENCE", "audio has no blocks above the BS.1770 gates")
    return float(-0.691 + 10.0 * math.log10(float(np.mean(energies[gated]))))


def _sinc_peak(samples: np.ndarray, oversample: int = 4, half_taps: int = 24) -> float:
    """4x windowed-sinc true-peak estimate around all local absolute maxima."""

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    taps = np.arange(-half_taps + 1, half_taps, dtype=np.float64)
    for channel in range(samples.shape[1]):
        mono = samples[:, channel]
        absolute = np.abs(mono)
        if mono.shape[0] < 3:
            continue
        local = (
            (absolute[1:-1] >= absolute[:-2])
            & (absolute[1:-1] >= absolute[2:])
        )
        candidates = np.flatnonzero(local) + 1
        if candidates.size == 0:
            continue
        candidates = candidates[
            (candidates >= half_taps) & (candidates < mono.shape[0] - half_taps)
        ]
        if candidates.size == 0:
            continue
        for phase in range(1, oversample):
            fraction = phase / oversample
            coefficients = np.sinc(fraction - taps)
            window = 0.5 + 0.5 * np.cos(np.pi * taps / half_taps)
            coefficients *= window
            coefficients /= np.sum(coefficients)
            for start in range(0, candidates.shape[0], 65536):
                chunk = candidates[start : start + 65536]
                indices = chunk[:, None] + taps[None, :].astype(np.int64)
                values = mono[indices] @ coefficients
                if values.size:
                    peak = max(peak, float(np.max(np.abs(values))))
    return peak


def true_peak_dbtp(samples: np.ndarray) -> float:
    peak = _sinc_peak(samples)
    return 20.0 * math.log10(peak) if peak > 0.0 else -math.inf


def measure(
    samples: np.ndarray,
    sample_rate: int,
) -> LoudnessMeasurement:
    return LoudnessMeasurement(
        integrated_lufs=integrated_loudness(samples, sample_rate),
        true_peak_dbtp=true_peak_dbtp(samples),
    )


def _quantize(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int16)
    return values, values.astype(np.float64) / 32768.0


def normalize(
    samples: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    max_true_peak_dbtp: float,
) -> NormalizationResult:
    """Apply global gain, preferring the true-peak cap over loudness target."""

    input_loudness = integrated_loudness(samples, sample_rate)
    input_peak = true_peak_dbtp(samples)
    if not math.isfinite(input_peak):
        raise BusinessError("E_SILENCE", "cannot normalize silent audio")
    loudness_gain = target_lufs - input_loudness
    peak_gain = max_true_peak_dbtp - input_peak
    gain_db = min(loudness_gain, peak_gain)
    peak_limited = loudness_gain > peak_gain + 1.0e-9

    quantized_float = np.zeros_like(samples, dtype=np.float64)
    for _ in range(8):
        scaled = samples * (10.0 ** (gain_db / 20.0))
        _, quantized_float = _quantize(scaled)
        actual_peak = true_peak_dbtp(quantized_float)
        if actual_peak <= max_true_peak_dbtp + 1.0e-9:
            break
        gain_db -= actual_peak - max_true_peak_dbtp + 1.0e-7
        peak_limited = True

    final_loudness = integrated_loudness(quantized_float, sample_rate)
    final_peak = true_peak_dbtp(quantized_float)
    return NormalizationResult(
        samples=quantized_float,
        input_lufs=input_loudness,
        input_true_peak_dbtp=input_peak,
        gain_db=gain_db,
        final_lufs=final_loudness,
        final_true_peak_dbtp=final_peak,
        peak_limited=peak_limited,
    )

