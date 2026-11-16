"""Sample-rate conversion and channel layout normalization.

``soxr`` (SoX Resampler, high-quality band-limited resampling) is used when
available. A self-contained windowed-sinc polyphase resampler implemented with
NumPy is provided as a deterministic fallback so the package still works if the
optional dependency cannot be installed.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly when soxr is installed
    import soxr  # type: ignore

    _HAS_SOXR = True
except Exception:  # pragma: no cover
    soxr = None  # type: ignore
    _HAS_SOXR = False


def has_soxr() -> bool:
    return _HAS_SOXR


def _resample_fallback(audio: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Windowed-sinc (Kaiser window) polyphase resampler.

    Linear interpolation of the fractional source position combined with a
    fixed-length low-pass kernel. Inputs are float64; output length is chosen
    so sample ``n`` corresponds to time ``n / out_rate``.
    """
    if in_rate == out_rate:
        return audio.copy()

    ratio = out_rate / in_rate
    out_len = int(np.floor(audio.shape[0] * ratio))
    cutoff = min(1.0, ratio) * 0.97
    half_taps = 32
    beta = 14.0

    # Source sample positions for every output frame.
    out_positions = np.arange(out_len, dtype=np.float64) / ratio
    base = np.floor(out_positions).astype(np.int64)
    fraction = out_positions - base

    # Kernel offsets relative to each output position.
    taps = np.arange(-half_taps, half_taps + 1, dtype=np.float64)
    # distances[k, j] = source sample (base[k] + j - half_taps) minus position
    distances = taps[np.newaxis, :] - fraction[:, np.newaxis]
    x = distances * cutoff
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(x == 0.0, 1.0, np.sin(np.pi * x) / (np.pi * x))
    window = np.i0(beta * np.sqrt(np.clip(1.0 - (distances / half_taps) ** 2, 0.0, 1.0))) / np.i0(beta)
    kernel = sinc * window * cutoff
    kernel /= kernel.sum(axis=1, keepdims=True)

    source_indices = base[:, np.newaxis] + taps[np.newaxis, :].astype(np.int64)
    valid = (source_indices >= 0) & (source_indices < audio.shape[0])
    safe_indices = np.clip(source_indices, 0, audio.shape[0] - 1)
    weighted = kernel * valid
    frames = audio[safe_indices] * weighted[:, :, np.newaxis]
    return frames.sum(axis=1)


def resample_audio(audio: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Resample an ``(n, channels)`` float array to ``out_rate``."""
    if in_rate == out_rate:
        return np.asarray(audio, dtype=np.float64).copy()
    if _HAS_SOXR:
        # HQ quality: steep filter, high signal-to-noise ratio.
        return soxr.resample(
            np.asarray(audio, dtype=np.float32), in_rate, out_rate, quality="HQ"
        ).astype(np.float64)
    return _resample_fallback(np.asarray(audio, dtype=np.float64), in_rate, out_rate)


def convert_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    """Normalize channel layout.

    mono -> stereo: duplicate the single channel (equal power, 0 dB pan).
    stereo -> mono: energy-preserving average, L/R scaled by 1/sqrt(2).
    """
    if audio.ndim != 2:
        raise ValueError("convert_channels expects an (samples, channels) array")
    source_channels = audio.shape[1]
    if source_channels == target_channels:
        return audio
    if source_channels == 1 and target_channels == 2:
        return np.repeat(audio, 2, axis=1)
    if source_channels == 2 and target_channels == 1:
        mono = (audio[:, 0] + audio[:, 1]) / np.sqrt(2.0)
        return mono[:, np.newaxis]
    raise ValueError(
        f"unsupported channel conversion: {source_channels} -> {target_channels} "
        "(only mono/stereo sources are supported)"
    )


def normalize_source(
    audio: np.ndarray,
    source_rate: int,
    source_channels: int,
    project_rate: int,
    project_channels: int,
) -> np.ndarray:
    """Resample first, then convert the channel layout."""
    resampled = resample_audio(audio, source_rate, project_rate)
    return convert_channels(resampled, project_channels)
