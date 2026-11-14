"""EBU R128 loudness measurement and true-peak detection.

Implements ITU-R BS.1770-4 K-weighting, channel weighting, gating and
integrated/momentary loudness in LUFS using only NumPy, plus true-peak
measurement via 4x oversampling.
"""
from __future__ import annotations

import numpy as np

# BS.1770-4 channel weights.  Index follows the WAVE channel ordering used by
# this project (see wavio.py): mono / stereo / 5.1 (L R C LFE Ls Rs).
CHANNEL_WEIGHTS = {
    1: np.array([1.0]),
    2: np.array([1.0, 1.0]),
    6: np.array([1.0, 1.0, 1.0, 0.0, np.sqrt(1.5), np.sqrt(1.5)]),
}

ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_OFFSET = -10.0
SILENCE_GATE_LUFS = -60.0
MIN_SILENCE_SECONDS = 0.2


def _rbj_biquad_coeffs(kind: str, fs: float, f0: float, gain_db: float = 0.0,
                       q: float = 1.0 / np.sqrt(2.0)):
    """Return RBJ cookbook biquad coefficients (b, a)."""
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    if kind == "high_shelf":
        alpha = sin_w / 2.0 * np.sqrt((amp + 1.0 / amp) * (1.0 / q - 1.0) + 2.0)
        b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w + 2.0 * np.sqrt(amp) * alpha)
        b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w)
        b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w - 2.0 * np.sqrt(amp) * alpha)
        a0 = (amp + 1.0) - (amp - 1.0) * cos_w + 2.0 * np.sqrt(amp) * alpha
        a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w)
        a2 = (amp + 1.0) - (amp - 1.0) * cos_w - 2.0 * np.sqrt(amp) * alpha
    elif kind == "high_pass":
        alpha = sin_w / (2.0 * q)
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = (1.0 + cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
    else:
        raise ValueError(f"unsupported biquad kind: {kind}")
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized transposed direct-form II biquad (scipy lfilter semantics).

    Coefficients are normalized so a[0] == 1.  The state update is fully
    vectorized over samples via two scalar accumulators, avoiding a Python
    loop per sample.
    """
    b0, b1, b2 = b
    _, a1, a2 = a
    y = np.empty_like(x)
    state1 = 0.0
    state2 = 0.0
    for i in range(x.shape[0]):
        value = b0 * x[i] + state1
        state1 = b1 * x[i] - a1 * value + state2
        state2 = b2 * x[i] - a2 * value
        y[i] = value
    return y


def k_weighted(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply the BS.1770-4 K-weighting chain to every channel."""
    b1, a1 = _rbj_biquad_coeffs("high_shelf", fs, 1681.974450955533, 3.61,
                                0.7071752369554196)
    b2, a2 = _rbj_biquad_coeffs("high_pass", fs, 38.13547087602444, 0.0,
                                0.5003270373238773)
    out = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        stage1 = _lfilter(b1, a1, signal[:, ch])
        out[:, ch] = _lfilter(b2, a2, stage1)
    return out


def _weighted_mean_square(kw: np.ndarray, n_channels: int) -> float:
    weights = CHANNEL_WEIGHTS[n_channels]
    return float(np.mean(kw * kw, axis=0) @ weights)


def measure_integrated_loudness(signal: np.ndarray, fs: float) -> float:
    """Integrated loudness in LUFS using 400 ms blocks and dual gating."""
    n_channels = signal.shape[1]
    if n_channels not in CHANNEL_WEIGHTS:
        raise ValueError(f"unsupported channel count: {n_channels}")
    kw = k_weighted(signal, fs)
    block = int(round(0.4 * fs))
    hop = int(round(0.1 * fs))
    if kw.shape[0] < block:
        return -np.inf
    n_blocks = 1 + (kw.shape[0] - block) // hop
    energies = np.empty(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        energies[i] = _weighted_mean_square(kw[start:start + block], n_channels)
    loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, 1e-20))
    gated = loudness[loudness > ABSOLUTE_GATE_LUFS]
    if gated.size == 0:
        return -np.inf
    relative_threshold = gated.mean() + RELATIVE_GATE_OFFSET
    final = loudness[loudness > relative_threshold]
    if final.size == 0:
        return -np.inf
    return float(final.mean())


def momentary_loudness_track(signal: np.ndarray, fs: float,
                             frame_seconds: float = 0.2,
                             hop_seconds: float = 0.05):
    """Return (loudness per frame in LUFS, hop in seconds)."""
    n_channels = signal.shape[1]
    kw = k_weighted(signal, fs)
    frame = int(round(frame_seconds * fs))
    hop = max(1, int(round(hop_seconds * fs)))
    if kw.shape[0] < frame:
        return np.array([], dtype=float), hop / fs
    n_frames = 1 + (kw.shape[0] - frame) // hop
    loudness = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop
        energy = _weighted_mean_square(kw[start:start + frame], n_channels)
        loudness[i] = -0.691 + 10.0 * np.log10(max(energy, 1e-20))
    return loudness, hop / fs


def find_silence_segments(signal: np.ndarray, fs: float,
                          threshold_lufs: float = SILENCE_GATE_LUFS,
                          min_seconds: float = MIN_SILENCE_SECONDS):
    """Detect contiguous silent regions below -60 LUFS lasting >= 200 ms."""
    loudness, hop = momentary_loudness_track(signal, fs, 0.2, 0.05)
    if loudness.size == 0:
        return []
    silent = loudness < threshold_lufs
    frame_seconds = 0.2
    segments = []
    i = 0
    n = silent.size
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and silent[j + 1]:
            j += 1
        start = i * hop
        end = j * hop + frame_seconds
        if end - start >= min_seconds - 1e-9:
            segments.append((start, end))
        i = j + 1
    return segments


def _sinc_lowpass(num_taps: int, cutoff_norm: float) -> np.ndarray:
    """Windowed-sinc low-pass filter; cutoff_norm is relative to Nyquist."""
    half = (num_taps - 1) // 2
    n = np.arange(-half, half + 1)
    h = cutoff_norm * np.sinc(cutoff_norm * n)
    window = np.hanning(num_taps)
    h *= window
    h /= h.sum()
    return h


def oversample_4x(signal: np.ndarray) -> np.ndarray:
    """4x oversample (n, ch) signal with an anti-imaging FIR filter."""
    n_samples, n_channels = signal.shape
    up = np.zeros((n_samples * 4, n_channels), dtype=np.float64)
    up[::4] = signal
    kernel = _sinc_lowpass(129, 0.25) * 4.0
    filtered = np.empty_like(up)
    for ch in range(n_channels):
        filtered[:, ch] = np.convolve(up[:, ch], kernel, mode="same")
    return filtered


def downsample_4x(signal_4x: np.ndarray) -> np.ndarray:
    """Invert `oversample_4x` with the matching anti-aliasing filter."""
    kernel = _sinc_lowpass(129, 0.25)
    filtered = np.empty_like(signal_4x)
    for ch in range(signal_4x.shape[1]):
        filtered[:, ch] = np.convolve(signal_4x[:, ch], kernel, mode="same")
    return filtered[::4]


def true_peak_db(signal: np.ndarray) -> float:
    """True-peak level in dBTP using 4x oversampling."""
    over = oversample_4x(signal)
    peak = float(np.max(np.abs(over))) if over.size else 0.0
    if peak <= 0.0:
        return -np.inf
    return float(20.0 * np.log10(peak))
