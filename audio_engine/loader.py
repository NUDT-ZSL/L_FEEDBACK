"""WAV loading and band-limited resampling to the engine sample rate."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from . import ENGINE_SAMPLE_RATE
from .errors import BusinessError, ERR_BAD_SAMPLE_RATE, ERR_MISSING_FILE

ALLOWED_SAMPLE_RATES = (44100, 48000)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a 16-bit PCM WAV (mono or stereo) as float64 samples.

    Returns (data, sample_rate) with data shaped (n_samples, n_channels);
    the original channel count is preserved.
    """
    if not os.path.isfile(path):
        raise BusinessError(ERR_MISSING_FILE, f"WAV file not found: {path}")
    info = sf.info(path)
    if info.samplerate not in ALLOWED_SAMPLE_RATES:
        raise BusinessError(
            ERR_BAD_SAMPLE_RATE,
            f"{path}: sample rate {info.samplerate} Hz, "
            f"expected one of {ALLOWED_SAMPLE_RATES}",
        )
    data = sf.read(path, dtype="float64", always_2d=True)[0]
    return data, info.samplerate


def resample_to_engine(
    data: np.ndarray,
    sr_in: int,
    sr_out: int = ENGINE_SAMPLE_RATE,
    num_taps: int = 128,
    kaiser_beta: float = 10.0,
) -> np.ndarray:
    """Band-limited resampling via windowed-sinc (Kaiser) interpolation.

    The sinc kernel is dilated by min(1, sr_out/sr_in) so that, when
    downsampling, the interpolation filter doubles as an anti-aliasing
    lowpass. Channel count is preserved.
    """
    if sr_in == sr_out:
        return data.copy()
    if data.ndim != 2:
        raise ValueError("expected (n_samples, n_channels) input")

    ratio = sr_out / sr_in
    n_in, n_ch = data.shape
    n_out = int(round(n_in * ratio))
    scale = min(1.0, ratio)  # anti-alias cutoff scale
    half = num_taps // 2
    taps = np.arange(num_taps)
    window = np.kaiser(num_taps, kaiser_beta)

    out = np.zeros((n_out, n_ch), dtype=np.float64)
    chunk = 16384
    for start in range(0, n_out, chunk):
        stop = min(start + chunk, n_out)
        n = np.arange(start, stop, dtype=np.float64)
        pos = n / ratio  # continuous input position per output sample
        i0 = np.floor(pos).astype(np.int64) - half + 1
        idx = i0[:, None] + taps[None, :]          # (B, K) input indices
        dist = pos[:, None] - idx                  # fractional distances
        kern = np.sinc(dist * scale) * window[None, :] * scale
        valid = (idx >= 0) & (idx < n_in)
        idx_c = np.clip(idx, 0, n_in - 1)
        kern = np.where(valid, kern, 0.0)
        for ch in range(n_ch):
            gathered = data[idx_c, ch]
            out[start:stop, ch] = np.einsum("bk,bk->b", gathered, kern)
    return out
