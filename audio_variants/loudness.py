from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .dsp import _next_power_of_two, filter_with_response


def _high_shelf_coefficients(sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    fc = min(1681.974450955533, 0.45 * sample_rate)
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * fc / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = np.array(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=np.float64,
    )
    return b, a


def _highpass_coefficients(sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    fc = min(38.13547087602444, 0.45 * sample_rate)
    q = 0.5003270373238773
    k = math.tan(math.pi * fc / sample_rate)
    a0 = 1.0 + k / q + k * k
    b = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=np.float64,
    )
    return b, a


def k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the ITU-R BS.1770-4 K-weighting filter to each channel."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames, channels)")
    length = audio.shape[0]
    if length == 0:
        return audio.copy()
    fft_size = _next_power_of_two(length + 16384)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    z_inv = np.exp(-2j * np.pi * frequencies / sample_rate)

    shelf_b, shelf_a = _high_shelf_coefficients(sample_rate)
    highpass_b, highpass_a = _highpass_coefficients(sample_rate)
    shelf_response = (shelf_b[0] + shelf_b[1] * z_inv + shelf_b[2] * z_inv * z_inv) / (
        1.0 + shelf_a[1] * z_inv + shelf_a[2] * z_inv * z_inv
    )
    highpass_response = (
        highpass_b[0] + highpass_b[1] * z_inv + highpass_b[2] * z_inv * z_inv
    ) / (1.0 + highpass_a[1] * z_inv + highpass_a[2] * z_inv * z_inv)
    response = shelf_response * highpass_response

    weighted = np.empty_like(audio, dtype=np.float64)
    for channel_index in range(audio.shape[1]):
        weighted[:, channel_index] = filter_with_response(
            audio[:, channel_index], response, fft_size
        )
    return weighted


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate gated integrated loudness in LUFS (BS.1770-4)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames, channels)")
    length = audio.shape[0]
    if length == 0:
        return float("-inf")
    weighted = k_weight(audio, sample_rate)
    block_length = max(1, int(round(0.400 * sample_rate)))
    step = max(1, int(round(0.100 * sample_rate)))

    if length < block_length:
        blocks = [weighted]
    else:
        starts = range(0, length - block_length + 1, step)
        blocks = [weighted[start : start + block_length] for start in starts]
    if not blocks:
        return float("-inf")

    powers = np.empty(len(blocks), dtype=np.float64)
    for index, block in enumerate(blocks):
        channel_power = np.mean(block * block, axis=0)
        powers[index] = float(np.sum(channel_power))
    if not np.all(np.isfinite(powers)):
        return float("-inf")
    positive = powers > 1e-20
    if not np.any(positive):
        return float("-inf")

    block_loudness = np.full_like(powers, float("-inf"))
    block_loudness[positive] = -0.691 + 10.0 * np.log10(powers[positive])
    absolute = block_loudness > -70.0
    if not np.any(absolute):
        return float("-inf")
    absolute_mean = float(np.mean(powers[absolute]))
    relative_threshold = -0.691 + 10.0 * math.log10(absolute_mean) - 10.0
    gated = absolute & (block_loudness > relative_threshold)
    if not np.any(gated):
        return float("-inf")
    final_power = float(np.mean(powers[gated]))
    if final_power <= 0.0:
        return float("-inf")
    return -0.691 + 10.0 * math.log10(final_power)


def ungated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """K-weighted loudness without gating, used as a fallback for very quiet clips."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames, channels)")
    if audio.shape[0] == 0:
        return float("-inf")
    weighted = k_weight(audio, sample_rate)
    power = float(np.sum(np.mean(weighted * weighted, axis=0)))
    if not math.isfinite(power) or power <= 1e-20:
        return float("-inf")
    return -0.691 + 10.0 * math.log10(power)


def _true_peak_channel(channel: np.ndarray, oversampling: int = 8) -> float:
    channel = np.asarray(channel, dtype=np.float64)
    length = channel.shape[0]
    if length == 0:
        return 0.0
    peak = float(np.max(np.abs(channel)))
    if peak == 0.0:
        return 0.0
    if oversampling < 1:
        raise ValueError("oversampling must be at least 1")
    if oversampling == 1:
        return peak

    fft_size = _next_power_of_two(max(2, length + 1))
    spectrum = np.fft.rfft(channel, n=fft_size)
    oversampled_spectrum = np.zeros(
        fft_size * oversampling // 2 + 1, dtype=np.complex128
    )
    oversampled_spectrum[: spectrum.shape[0]] = spectrum
    # Split the original Nyquist bin when the spectrum is extended.
    oversampled_spectrum[spectrum.shape[0] - 1] *= 0.5
    oversampled = (
        np.fft.irfft(oversampled_spectrum, n=fft_size * oversampling) * oversampling
    )
    return max(peak, float(np.max(np.abs(oversampled[: length * oversampling]))))


def true_peak_dbtp(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames, channels)")
    if audio.shape[0] == 0:
        return float("-inf")
    peak = 0.0
    for channel_index in range(audio.shape[1]):
        peak = max(peak, _true_peak_channel(audio[:, channel_index]))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)


def measure(audio: np.ndarray, sample_rate: int) -> Dict[str, Optional[float]]:
    lufs = integrated_loudness(audio, sample_rate)
    peak = true_peak_dbtp(audio)
    return {
        "loudness_lufs": lufs if math.isfinite(lufs) else None,
        "true_peak_dbtp": peak if math.isfinite(peak) else None,
    }


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    peak_limit_dbtp: float = -1.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Normalize integrated loudness, then attenuate to satisfy the peak ceiling."""
    result = np.asarray(audio, dtype=np.float64).copy()
    initial_lufs = integrated_loudness(result, sample_rate)
    input_method = "integrated"
    if not math.isfinite(initial_lufs):
        fallback_lufs = ungated_loudness(result, sample_rate)
        if math.isfinite(fallback_lufs):
            initial_lufs = fallback_lufs
            input_method = "ungated_fallback"
    initial_peak = true_peak_dbtp(result)
    requested_gain_db = 0.0
    if math.isfinite(initial_lufs):
        requested_gain_db = target_lufs - initial_lufs
        result *= 10.0 ** (requested_gain_db / 20.0)
        for _ in range(3):
            current_lufs = integrated_loudness(result, sample_rate)
            if not math.isfinite(current_lufs):
                break
            correction_db = target_lufs - current_lufs
            if abs(correction_db) < 1e-6:
                break
            result *= 10.0 ** (correction_db / 20.0)
            requested_gain_db += correction_db

    predicted_peak = true_peak_dbtp(result)
    sample_peak = (
        20.0 * math.log10(float(np.max(np.abs(result))))
        if result.size and np.max(np.abs(result)) > 0.0
        else float("-inf")
    )
    limiting_peak = max(predicted_peak, sample_peak)
    limiting_gain_db = 0.0
    limited = False
    enforcement_threshold = peak_limit_dbtp - 0.02
    if math.isfinite(limiting_peak) and limiting_peak > peak_limit_dbtp:
        limiting_gain_db = enforcement_threshold - limiting_peak
        result *= 10.0 ** (limiting_gain_db / 20.0)
        limited = True

    actual = measure(result, sample_rate)
    return result, {
        "target_lufs": target_lufs,
        "input_lufs": initial_lufs if math.isfinite(initial_lufs) else None,
        "input_loudness_method": input_method,
        "input_true_peak_dbtp": initial_peak if math.isfinite(initial_peak) else None,
        "requested_gain_db": requested_gain_db,
        "limiting_gain_db": limiting_gain_db,
        "peak_limited": limited,
        "actual_lufs": actual["loudness_lufs"],
        "actual_true_peak_dbtp": actual["true_peak_dbtp"],
    }
