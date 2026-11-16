"""Digital signal processing helpers.

All processing is deterministic and offline:

* :func:`resample_to_rate` uses SciPy's polyphase FIR resampler,
* :func:`normalize_channels` converts mono/stereo to the internal stereo
  float32 representation,
* :func:`time_stretch` performs phase-vocoder time scaling (pitch preserved)
  via STFT frame interpolation and always returns exactly the requested
  number of frames.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly, stft, istft

from .errors import BuildError, ExitCode

INTERNAL_SAMPLE_RATE = 48000
STFT_NFFT = 2048
STFT_HOP = 512


def _gcd(a: int, b: int) -> int:
    return math.gcd(int(a), int(b))


def resample_to_rate(data: np.ndarray, source_rate: int,
                     target_rate: int = INTERNAL_SAMPLE_RATE) -> np.ndarray:
    """Resample a ``(frames, channels)`` float array to ``target_rate``."""
    if source_rate == target_rate:
        return np.asarray(data, dtype=np.float32, order="C")
    divisor = _gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    # resample_poly works along axis 0; keep per-channel independence.
    result = resample_poly(data.astype(np.float64), up, down, axis=0, padtype="line")
    expected = int(round(data.shape[0] * target_rate / source_rate))
    result = result[:expected]
    if result.shape[0] < expected:
        pad = np.zeros((expected - result.shape[0], data.shape[1]), dtype=np.float64)
        result = np.concatenate([result, pad], axis=0)
    return np.ascontiguousarray(result, dtype=np.float32)


def normalize_channels(data: np.ndarray) -> np.ndarray:
    """Normalize to stereo ``(frames, 2)`` float32 without changing timing."""
    if data.ndim != 2:
        raise BuildError(ExitCode.UNSUPPORTED_FORMAT,
                         "audio data must have two dimensions",
                         {"shape": list(data.shape)})
    if data.shape[1] == 2:
        return np.asarray(data, dtype=np.float32, order="C")
    if data.shape[1] == 1:
        stereo = np.repeat(data.astype(np.float32), 2, axis=1)
        return np.ascontiguousarray(stereo)
    if data.shape[1] > 2:
        # Defensive down-mix for unexpected multi-channel input.
        mono = data.astype(np.float32).mean(axis=1, keepdims=True)
        return np.ascontiguousarray(np.repeat(mono, 2, axis=1))
    raise BuildError(ExitCode.UNSUPPORTED_FORMAT,
                     "audio data must contain at least one channel",
                     {"shape": list(data.shape)})


def _stretch_channel(mono: np.ndarray, factor: float, sample_rate: int) -> np.ndarray:
    noverlap = STFT_NFFT - STFT_HOP
    _, _, spectrum = stft(
        mono, fs=sample_rate, window="hann", nperseg=STFT_NFFT,
        noverlap=noverlap, boundary="even", padded=True,
    )
    source_frames = np.arange(spectrum.shape[1], dtype=np.float64)
    target_count = max(1, int(round(spectrum.shape[1] / factor)))
    target_frames = np.arange(target_count, dtype=np.float64) * factor
    target_frames = np.clip(target_frames, 0.0, source_frames[-1])
    lower = np.floor(target_frames).astype(np.int64)
    upper = np.minimum(lower + 1, spectrum.shape[1] - 1)
    fraction = (target_frames - lower).astype(np.complex128)
    stretched = spectrum[:, lower] * (1.0 - fraction) + spectrum[:, upper] * fraction

    _, rebuilt = istft(
        stretched, fs=sample_rate, window="hann", nperseg=STFT_NFFT,
        noverlap=noverlap, input_onesided=True, boundary=True,
    )
    return rebuilt


def time_stretch(data: np.ndarray, factor: float,
                 sample_rate: int = INTERNAL_SAMPLE_RATE,
                 exact_frames: int | None = None) -> np.ndarray:
    """Stretch ``(frames, channels)`` audio in time without pitch shift.

    Args:
        data: Input audio at the internal sample rate.
        factor: Time stretch factor (``>1`` makes audio longer/slower).
        sample_rate: Internal sample rate.
        exact_frames: When given, the output is trimmed or zero-padded to
            this exact frame count so section lengths match the timeline.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise BuildError(ExitCode.MAPPING_CONFLICT,
                         "time stretch factor must be finite and positive",
                         {"factor": factor})
    if factor == 1.0:
        output = np.asarray(data, dtype=np.float32, order="C")
    else:
        channels = [
            _stretch_channel(data[:, channel].astype(np.float64), factor, sample_rate)
            for channel in range(data.shape[1])
        ]
        output = np.stack(channels, axis=1).astype(np.float32)

    if exact_frames is not None:
        if output.shape[0] >= exact_frames:
            output = output[:exact_frames]
        else:
            padding = np.zeros((exact_frames - output.shape[0], output.shape[1]),
                               dtype=np.float32)
            output = np.concatenate([output, padding], axis=0)
    return np.ascontiguousarray(output, dtype=np.float32)


def equal_power_crossfade(length: int) -> np.ndarray:
    """Return a fade-in gain curve; fade-out is ``sqrt(1 - gain**2)``."""
    if length <= 0:
        return np.ones(0, dtype=np.float64)
    phase = np.linspace(0.0, np.pi / 2.0, length, dtype=np.float64)
    return np.sin(phase)


def apply_gain_db(data: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0.0:
        return data
    linear = np.power(10.0, gain_db / 20.0, dtype=np.float64)
    return (data.astype(np.float64) * linear).astype(np.float32)
