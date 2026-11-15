"""True-peak measurement via FFT oversampling (BS.1770 style, 192 kHz domain)."""

from __future__ import annotations

import numpy as np

TARGET_TP_RATE = 192_000


def oversample(x: np.ndarray, fs: int, target_fs: int = TARGET_TP_RATE) -> np.ndarray:
    """Resample a 1-D signal up to target_fs by spectral zero-padding."""
    factor = max(1, int(round(target_fs / fs)))
    if factor <= 1:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x
    n_out = n * factor
    spec = np.fft.rfft(x)
    out = np.zeros(n_out // 2 + 1, dtype=spec.dtype)
    if n % 2 == 0:
        nyq = n // 2
        out[:nyq] = spec[:nyq]
        out[nyq] = spec[nyq] * 0.5
        out[n_out // 2 - nyq + 1] = spec[nyq] * 0.5
    else:
        out[: (n + 1) // 2] = spec
    return np.fft.irfft(out, n_out) * factor


def true_peak_dbfs(pcm: np.ndarray, fs: int) -> float:
    """True peak of (samples, channels) pcm in dBTP (192 kHz domain)."""
    peak = 0.0
    for c in range(pcm.shape[1]):
        up = oversample(pcm[:, c], fs)
        if up.size:
            peak = max(peak, float(np.max(np.abs(up))))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)
