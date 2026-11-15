"""ITU-R BS.1770-4 K-weighting filters and integrated loudness."""

from __future__ import annotations

import numpy as np

# Filter design constants from ITU-R BS.1770-4 (valid for any sample rate).
_SHELF_G = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_FC = 1681.9744509555319
_HP_Q = 0.5003270373238773
_HP_FC = 38.13547087602444

_IR_LEN = 8192  # taps used to represent the cascade impulse response


def _high_shelf(fs: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.tan(np.pi * _SHELF_FC / fs)
    vh = 10.0 ** (_SHELF_G / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / _SHELF_Q + k * k
    b = np.array([
        (vh + vb * k / _SHELF_Q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / _SHELF_Q + k * k) / a0,
    ])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _SHELF_Q + k * k) / a0])
    return b, a


def _high_pass(fs: float) -> tuple[np.ndarray, np.ndarray]:
    # Matches ffmpeg's ebur128 RLB stage exactly (numerator intentionally
    # not normalized by a0, as in libavfilter/f_ebur128.c), keeping the
    # deviation from the reference implementation well under 0.1 LU at
    # any sample rate.
    k = np.tan(np.pi * _HP_FC / fs)
    a0 = 1.0 + k / _HP_Q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _HP_Q + k * k) / a0])
    return b, a


def _cascade_impulse_response(fs: float) -> np.ndarray:
    """Impulse response of shelf -> high-pass cascade (zero initial state)."""

    def run(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        x1 = x2 = y1 = y2 = 0.0
        for i, xn in enumerate(x):
            yn = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2, x1 = x1, xn
            y2, y1 = y1, yn
            y[i] = yn
        return y

    impulse = np.zeros(_IR_LEN)
    impulse[0] = 1.0
    b1, a1 = _high_shelf(fs)
    b2, a2 = _high_pass(fs)
    return run(b2, a2, run(b1, a1, impulse))


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = x.shape[0] + h.shape[0] - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)
    return y[: x.shape[0]]


_ir_cache: dict[float, np.ndarray] = {}


def k_weighted(x: np.ndarray, fs: float) -> np.ndarray:
    """Apply the BS.1770 K-weighting cascade to a 1-D signal."""
    h = _ir_cache.get(fs)
    if h is None:
        h = _cascade_impulse_response(fs)
        _ir_cache[fs] = h
    return _fft_convolve(np.asarray(x, dtype=np.float64), h)


def channel_weight(name: str) -> float:
    """BS.1770 channel weighting (LFE excluded, surrounds +1.5 dB)."""
    if name in ("LFE", "LFE2"):
        return 0.0
    if name in ("BL", "BR", "SL", "SR", "BC", "SSL", "SSR"):
        return 1.41
    return 1.0


def integrated_loudness(
    pcm: np.ndarray, fs: float, weights: list[float]
) -> float:
    """BS.1770-4 integrated loudness in LUFS.

    pcm: (samples, channels) float array. weights: per-channel BS.1770
    weights (0.0 channels are excluded). Uses 400 ms blocks with 75%
    overlap, an absolute gate at -70 LUFS and a relative gate 10 LU
    below the ungated average.
    """
    n, ch = pcm.shape
    block = int(round(0.400 * fs))
    hop = int(round(0.100 * fs))
    if n < block:
        return float("-inf")

    filtered = np.empty_like(pcm, dtype=np.float64)
    for c in range(ch):
        if weights[c] > 0.0:
            filtered[:, c] = k_weighted(pcm[:, c], fs)

    n_blocks = 1 + (n - block) // hop
    energy = np.empty(n_blocks)
    idx = np.arange(block)
    for j in range(n_blocks):
        seg = filtered[j * hop + idx]
        energy[j] = np.sum([(weights[c] * np.mean(seg[:, c] ** 2)) for c in range(ch) if weights[c] > 0.0])

    with np.errstate(divide="ignore"):
        block_loudness = -0.691 + 10.0 * np.log10(energy)

    keep = block_loudness >= -70.0
    if not np.any(keep):
        return float("-inf")
    gated_energy = energy[keep]
    rel_threshold = -0.691 + 10.0 * np.log10(np.mean(gated_energy)) - 10.0
    keep2 = keep & (block_loudness > rel_threshold)
    if not np.any(keep2):
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(energy[keep2])))
