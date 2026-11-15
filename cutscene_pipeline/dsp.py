"""Deterministic DSP primitives implemented with NumPy only.

Includes rational-ratio resampling (windowed-sinc polyphase), BS.1770
K-weighting filters, EBU R128 integrated loudness, 4x-oversampled true
peak measurement and a deterministic lookahead peak limiter.
"""

from __future__ import annotations

import math

import numpy as np

# BS.1770-4 K-weighting filter coefficients for 48 kHz.
SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
HP_B = (1.0, -2.0, 1.0)
HP_A = (1.0, -1.99004745483398, 0.99007225036621)

LOUDNESS_OFFSET = -0.691  # BS.1770 absolute offset (LU)
ABS_GATE_LUFS = -70.0
REL_GATE_LU = 10.0


def resample_rational(x: np.ndarray, up: int, down: int, half_len: int = 32) -> np.ndarray:
    """Resample 1-D signal by rational factor up/down using a centered
    windowed-sinc kernel (zero group delay, deterministic)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    ratio = up / down
    out_len = int(math.ceil(x.size * ratio))
    m = np.arange(out_len, dtype=np.float64)
    pos = m * (down / up)
    k0 = np.floor(pos).astype(np.int64)
    frac = pos - k0
    cutoff = min(1.0, ratio)
    y = np.zeros(out_len, dtype=np.float64)
    for j in range(-half_len + 1, half_len + 1):
        idx = k0 + j
        valid = (idx >= 0) & (idx < x.size)
        t = frac - j
        w = t / half_len
        window = np.where(np.abs(w) <= 1.0, 0.5 + 0.5 * np.cos(np.pi * w), 0.0)
        kernel = cutoff * np.sinc(cutoff * t) * window
        contrib = np.zeros(out_len, dtype=np.float64)
        contrib[valid] = x[idx[valid]]
        y += contrib * kernel
    return y


def apply_biquad(x: np.ndarray, b, a) -> np.ndarray:
    """Direct-form I biquad, pure Python loop (deterministic)."""
    b0, b1, b2 = b
    _, a1, a2 = a
    y = np.empty(x.shape[0], dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(x.shape[0]):
        xn = float(x[i])
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xn
        y2, y1 = y1, yn
        y[i] = yn
    return y


_IR_LENGTH = 32768  # impulse-response truncation for FFT filtering
_ir_cache: dict = {}


def _impulse_response(b, a) -> np.ndarray:
    key = (tuple(b), tuple(a))
    if key not in _ir_cache:
        delta = np.zeros(_IR_LENGTH, dtype=np.float64)
        delta[0] = 1.0
        _ir_cache[key] = apply_biquad(delta, b, a)
    return _ir_cache[key]


def fft_filter(x: np.ndarray, b, a) -> np.ndarray:
    """Exact biquad filtering via FFT convolution with the truncated
    impulse response (deterministic, much faster than the sample loop)."""
    h = _impulse_response(b, a)
    n = x.shape[0] + h.shape[0] - 1
    spec = np.fft.rfft(x, n) * np.fft.rfft(h, n)
    return np.fft.irfft(spec, n)[:x.shape[0]]


def k_weight(x: np.ndarray) -> np.ndarray:
    return fft_filter(fft_filter(x, SHELF_B, SHELF_A), HP_B, HP_A)


def integrated_loudness(multich: np.ndarray, sample_rate: int = 48000) -> float:
    """EBU R128 / BS.1770-4 integrated loudness (LUFS).

    multich: (n_samples, n_channels) float array. Returns -inf for silence.
    """
    n = multich.shape[0]
    if n == 0:
        return float("-inf")
    block = int(round(0.400 * sample_rate))
    hop = int(round(0.100 * sample_rate))
    filtered = [k_weight(multich[:, c]) for c in range(multich.shape[1])]
    energies = []
    start = 0
    while start + block <= n:
        e = 0.0
        for ch in filtered:
            seg = ch[start:start + block]
            e += float(np.dot(seg, seg)) / block  # channel weight = 1.0
        energies.append(e)
        start += hop
    if not energies:
        # Shorter than one block: measure the whole signal as one block.
        e = 0.0
        for ch in filtered:
            e += float(np.dot(ch, ch)) / n
        energies.append(e)
    energies = np.asarray(energies, dtype=np.float64)
    with np.errstate(divide="ignore"):
        block_loudness = LOUDNESS_OFFSET + 10.0 * np.log10(energies)
    abs_mask = block_loudness >= ABS_GATE_LUFS
    if not np.any(abs_mask):
        return float("-inf")
    gated = energies[abs_mask]
    rel_threshold = LOUDNESS_OFFSET + 10.0 * math.log10(float(np.mean(gated))) - REL_GATE_LU
    final_mask = block_loudness[abs_mask] >= max(rel_threshold, ABS_GATE_LUFS)
    if not np.any(final_mask):
        return float("-inf")
    mean_energy = float(np.mean(gated[final_mask]))
    if mean_energy <= 0.0:
        return float("-inf")
    return LOUDNESS_OFFSET + 10.0 * math.log10(mean_energy)


def true_peak_dbtp(multich: np.ndarray, oversample: int = 4) -> float:
    """True peak in dBTP via 4x oversampling (BS.1770 Annex 2).

    Uses reflection-padded FFT spectral interpolation: exact bandlimited
    upsampling, deterministic and O(n log n).
    """
    peak = 0.0
    for c in range(multich.shape[1]):
        x = multich[:, c]
        n = x.shape[0]
        if n == 0:
            continue
        pad = min(256, n - 1) if n > 1 else 0
        if pad > 0:
            xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]])
        else:
            xp = x
        spec = np.fft.rfft(xp)
        n_up = xp.shape[0] * oversample
        up = np.fft.irfft(spec, n_up) * oversample
        interior = up[pad * oversample:(pad + n) * oversample]
        if interior.size:
            peak = max(peak, float(np.max(np.abs(interior))))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)


def peak_limit(multich: np.ndarray, ceiling_linear: float, sample_rate: int = 48000,
               release_ms: float = 50.0) -> np.ndarray:
    """Deterministic channel-linked lookahead peak limiter.

    Instant attack on the peak envelope, exponential release.
    """
    env = np.max(np.abs(multich), axis=1)
    n = env.shape[0]
    gain = np.empty(n, dtype=np.float64)
    release_coef = math.exp(-1.0 / (release_ms * 0.001 * sample_rate))
    g = 1.0
    for i in range(n):
        needed = 1.0 if env[i] <= ceiling_linear else ceiling_linear / env[i]
        if needed < g:
            g = needed  # instant attack
        else:
            g = needed + (g - needed) * release_coef
        gain[i] = g
    return multich * gain[:, None]


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(v: float) -> float:
    return float("-inf") if v <= 0.0 else 20.0 * math.log10(v)
