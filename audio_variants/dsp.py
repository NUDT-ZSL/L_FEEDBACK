from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _next_power_of_two(value: int) -> int:
    return 1 << int(math.ceil(math.log2(max(1, value))))


def _biquad_response(
    frequencies: np.ndarray, sample_rate: int, b: np.ndarray, a: np.ndarray
) -> np.ndarray:
    z_inv = np.exp(-2j * np.pi * frequencies / sample_rate)
    numerator = b[0] + b[1] * z_inv + b[2] * z_inv * z_inv
    denominator = 1.0 + a[1] * z_inv + a[2] * z_inv * z_inv
    return numerator / denominator


def filter_with_response(
    audio: np.ndarray, response: np.ndarray, fft_size: int
) -> np.ndarray:
    """Apply a frequency response using zero-padded FFT convolution."""
    spectrum = np.fft.rfft(audio, n=fft_size)
    return np.fft.irfft(spectrum * response, n=fft_size)[: audio.shape[0]]


def apply_biquad(
    audio: np.ndarray, sample_rate: int, b: np.ndarray, a: np.ndarray
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("apply_biquad expects a one-dimensional channel")
    length = audio.shape[0]
    if length == 0:
        return audio.copy()
    fft_size = _next_power_of_two(length + 16384)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    response = _biquad_response(frequencies, sample_rate, b, a)
    return filter_with_response(audio, response, fft_size)


def lowpass_coefficients(
    sample_rate: int, cutoff_hz: float, q: float = 1.0 / math.sqrt(2.0)
) -> Tuple[np.ndarray, np.ndarray]:
    if cutoff_hz <= 0.0 or cutoff_hz >= sample_rate / 2.0:
        raise ValueError("low-pass cutoff must be between 0 and Nyquist")
    w0 = 2.0 * np.pi * cutoff_hz / sample_rate
    cosine = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    b = (
        np.array(
            [(1.0 - cosine) / 2.0, 1.0 - cosine, (1.0 - cosine) / 2.0],
            dtype=np.float64,
        )
        / a0
    )
    a = np.array([1.0, -2.0 * cosine, 1.0 - alpha], dtype=np.float64) / a0
    return b, a


def lowpass_channel(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float
) -> np.ndarray:
    if cutoff_hz >= 0.4999 * sample_rate:
        return np.asarray(audio, dtype=np.float64).copy()
    b, a = lowpass_coefficients(sample_rate, cutoff_hz)
    return apply_biquad(audio, sample_rate, b, a)


def resample_cubic(audio: np.ndarray, output_length: int) -> np.ndarray:
    """Resample a one-dimensional channel to an exact output length."""
    audio = np.asarray(audio, dtype=np.float64)
    input_length = audio.shape[0]
    output_length = int(output_length)
    if output_length <= 0:
        return np.zeros(0, dtype=np.float64)
    if input_length == 0:
        return np.zeros(output_length, dtype=np.float64)
    if input_length == 1:
        return np.full(output_length, audio[0], dtype=np.float64)
    if output_length == input_length:
        return audio.copy()

    positions = np.linspace(0.0, float(input_length - 1), output_length)
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    indices = np.clip(
        np.stack((left - 1, left, left + 1, left + 2), axis=1),
        0,
        input_length - 1,
    )
    values = audio[indices]
    f2 = fraction * fraction
    f3 = f2 * fraction
    result = 0.5 * (
        (2.0 * values[:, 1])
        + (-values[:, 0] + values[:, 2]) * fraction
        + (2.0 * values[:, 0] - 5.0 * values[:, 1] + 4.0 * values[:, 2] - values[:, 3])
        * f2
        + (-values[:, 0] + 3.0 * values[:, 1] - 3.0 * values[:, 2] + values[:, 3]) * f3
    )
    return result


def _principal_phase(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _phase_vocoder_fft_size(sample_rate: int, length: int) -> int:
    target = int(round(sample_rate * 0.040))
    size = 256
    while size < target and size < 8192:
        size *= 2
    if length >= 64:
        size = min(size, _next_power_of_two(max(64, length)))
    return max(64, size)


def time_stretch_channel(
    audio: np.ndarray, sample_rate: int, stretch_ratio: float
) -> np.ndarray:
    """Phase-vocoder time stretch that preserves pitch."""
    audio = np.asarray(audio, dtype=np.float64)
    input_length = audio.shape[0]
    if input_length == 0:
        return audio.copy()
    output_length = max(1, int(round(input_length * stretch_ratio)))
    if abs(stretch_ratio - 1.0) < 1e-9:
        return np.asarray(audio, dtype=np.float64).copy()
    if input_length < 32:
        return resample_cubic(
            np.interp(
                np.linspace(0.0, 1.0, input_length),
                np.linspace(0.0, 1.0, input_length),
                audio,
            ),
            output_length,
        )

    fft_size = _phase_vocoder_fft_size(sample_rate, input_length)
    hop = max(1, fft_size // 4)
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(fft_size, dtype=np.float64) / fft_size
    )
    padded = np.pad(
        audio,
        (
            fft_size // 2,
            fft_size + int(math.ceil(fft_size * abs(stretch_ratio))),
        ),
        mode="constant",
    )
    frame_count = int(math.ceil(input_length / hop)) + 2
    output_positions = np.rint(
        np.arange(frame_count, dtype=np.float64) * hop * stretch_ratio
    ).astype(np.int64)
    synthesis_length = max(output_length, int(output_positions[-1]) + fft_size + 1)
    synthesized = np.zeros(synthesis_length, dtype=np.float64)
    normalization = np.zeros(synthesis_length, dtype=np.float64)
    angular_frequency = 2.0 * np.pi * np.arange(fft_size // 2 + 1) / fft_size
    previous_phase = None
    synthesis_phase = None

    for frame_index in range(frame_count):
        source_position = frame_index * hop
        if source_position + fft_size > padded.shape[0]:
            break
        frame = padded[source_position : source_position + fft_size] * window
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        if previous_phase is None:
            synthesis_phase = phase.copy()
        else:
            output_hop = int(
                output_positions[frame_index] - output_positions[frame_index - 1]
            )
            phase_delta = _principal_phase(
                phase - previous_phase - angular_frequency * hop
            )
            instantaneous = angular_frequency + phase_delta / hop
            synthesis_phase = synthesis_phase + instantaneous * output_hop
        rebuilt = np.fft.irfft(magnitude * np.exp(1j * synthesis_phase), n=fft_size)
        start = int(output_positions[frame_index])
        stop = start + fft_size
        synthesized[start:stop] += rebuilt * window
        normalization[start:stop] += window * window
        previous_phase = phase

    valid = normalization > 1e-12
    synthesized[valid] /= normalization[valid]
    synthesized[~valid] = 0.0
    result = synthesized[:output_length]
    if result.shape[0] < output_length:
        result = np.pad(result, (0, output_length - result.shape[0]))
    return result


def create_variant(
    audio: np.ndarray,
    sample_rate: int,
    pitch_semitones: float,
    stretch_ratio: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Apply pitch shift, time stretch, and low-pass filtering per channel."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames, channels)")
    pitch_ratio = 2.0 ** (pitch_semitones / 12.0)
    final_length = max(1, int(round(audio.shape[0] * stretch_ratio)))
    transformed = np.empty((final_length, audio.shape[1]), dtype=np.float64)

    for channel_index in range(audio.shape[1]):
        channel = audio[:, channel_index]
        intermediate_ratio = stretch_ratio * pitch_ratio
        intermediate_length = max(1, int(round(channel.shape[0] * intermediate_ratio)))
        if abs(intermediate_ratio - 1.0) < 1e-9:
            intermediate = channel.copy()
        else:
            intermediate = time_stretch_channel(
                channel, sample_rate, intermediate_ratio
            )
        if intermediate.shape[0] != intermediate_length:
            intermediate = resample_cubic(intermediate, intermediate_length)
        shifted = resample_cubic(intermediate, final_length)
        transformed[:, channel_index] = lowpass_channel(shifted, sample_rate, cutoff_hz)
    return transformed
