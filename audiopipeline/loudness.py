"""Loudness and true-peak measurement (ITU-R BS.1770-4).

Implemented with NumPy only:

* K-weighting is the two-stage BS.1770 filter chain (high-shelf + high-pass),
  applied through an overlap-save FFT convolution so long clips stay fast.
* Integrated loudness uses 400 ms blocks with 75 % overlap and the dual
  (absolute -70 LUFS / relative -10 LU) gating of BS.1770-4.
* True peak is measured on a 4x zero-padded FFT up-sampled signal, processed
  in overlapping chunks to keep memory bounded.
"""

from __future__ import annotations

import numpy as np

from .config import SAMPLE_RATE

# BS.1770-4 K-weighting biquad coefficients (stage 1 high-shelf, stage 2 HPF).
_B1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_A1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
_B2 = np.array([1.0, -2.0, 1.0])
_A2 = np.array([1.0, -1.99004745483398, 0.99007225036621])

_GATING_OFFSET_LU = -10.0
_WEIGHT = 1.0  # stereo channels both carry weight 1.0
_OFFSET_DB = -0.691

# Overlap-save geometry.
_FFT_N = 8192
_TAPS_L = 4096
_BLOCK_B = _FFT_N - _TAPS_L


def _k_weighting_response(n_fft: int) -> np.ndarray:
    """Combined causal K-weighting impulse response truncated to ``_TAPS_L``."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    omega = 2.0 * np.pi * freqs / SAMPLE_RATE
    z = np.exp(-1j * omega)

    def resp(b: np.ndarray, a: np.ndarray) -> np.ndarray:
        num = b[0] + b[1] * z + b[2] * z**2
        den = a[0] + a[1] * z + a[2] * z**2
        return num / den

    full = resp(_B1, _A1) * resp(_B2, _A2)
    impulse = np.fft.irfft(full, n=n_fft)[:_TAPS_L].astype(np.float64)
    return np.fft.rfft(impulse, n=n_fft)


_H_K = _k_weighting_response(_FFT_N)


def k_weight(audio: np.ndarray) -> np.ndarray:
    """Apply the K-weighting chain to a ``(frames, channels)`` array."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    frames, channels = x.shape
    # Zero-pad the beginning so the first linear-convolution block is
    # recovered from samples [_TAPS_L-1:] of each circular convolution.
    padded = np.vstack([np.zeros((_TAPS_L - 1, channels), dtype=np.float64), x])
    out = np.empty_like(x)
    pos = 0
    while pos < frames:
        block = padded[pos : pos + _FFT_N]
        if block.shape[0] < _FFT_N:  # final partial block
            block = np.vstack(
                [block, np.zeros((_FFT_N - block.shape[0], channels))]
            )
        spec = np.fft.rfft(block, n=_FFT_N, axis=0) * _H_K[:, None]
        conv = np.fft.irfft(spec, n=_FFT_N, axis=0)[_TAPS_L - 1 :]
        take = min(_BLOCK_B, frames - pos)
        out[pos : pos + take] = conv[:take]
        pos += take
    return out


def integrated_loudness(
    audio: np.ndarray,
    block_ms: float = 400.0,
    overlap: float = 0.75,
    absolute_gate_lufs: float = -70.0,
) -> float | None:
    """Return gated integrated loudness in LUFS, or ``None`` for silence.

    Clips shorter than one measurement block fall back to a single
    full-clip estimate (the short-event fallback required by the spec).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    frames = x.shape[0]
    if frames == 0:
        return None
    weighted = k_weight(x)

    block = int(round(SAMPLE_RATE * block_ms / 1000.0))
    hop = max(1, int(round(block * (1.0 - overlap))))

    if frames < block:
        # Short-event fallback: one ungated measurement over the whole clip.
        power = float(np.mean(weighted**2) * weighted.shape[1] * _WEIGHT)
        return _OFFSET_DB + 10.0 * np.log10(power) if power > 0.0 else None

    powers: list[float] = []
    for start in range(0, frames - block + 1, hop):
        segment = weighted[start : start + block]
        powers.append(float(np.mean(segment**2) * segment.shape[1] * _WEIGHT))
    if not powers:
        return None
    p = np.asarray(powers, dtype=np.float64)
    loud = _OFFSET_DB + 10.0 * np.log10(np.where(p > 0, p, 1e-20))

    stage1 = p[loud >= absolute_gate_lufs]
    if stage1.size == 0:
        return None
    ungated = _OFFSET_DB + 10.0 * np.log10(float(np.mean(stage1)))
    relative_gate = ungated + _GATING_OFFSET_LU
    stage2 = stage1[
        (_OFFSET_DB + 10.0 * np.log10(stage1))
        >= max(absolute_gate_lufs, relative_gate)
    ]
    if stage2.size == 0:
        return None
    return float(_OFFSET_DB + 10.0 * np.log10(float(np.mean(stage2))))


def _upsample_chunk(chunk: np.ndarray, factor: int) -> np.ndarray:
    n = chunk.shape[0]
    spec = np.fft.rfft(chunk, n=n, axis=0)
    up_spec = np.zeros((n * factor // 2 + 1, chunk.shape[1]), dtype=np.complex128)
    up_spec[: spec.shape[0]] = spec
    return np.fft.irfft(up_spec, n=n * factor, axis=0) * factor


def true_peak_db(audio: np.ndarray, factor: int = 4) -> float | None:
    """Measure true-peak level in dBTP via band-limited FFT up-sampling.

    Each block is wrapped in zero padding before the FFT, so the only
    signal discontinuities are real signal starts/ends.  Blocks advance by
    half their length and only the central (boundary-free) up-sampled
    samples are trusted, tiling the whole clip without edge leakage.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    frames = x.shape[0]
    if frames == 0:
        return None
    factor = max(1, int(factor))
    window = 4096
    pad = 2048
    block = window + 2 * pad           # 8192 input samples per FFT
    hop = window // 2                  # central trusted region
    padded = np.zeros((frames + 2 * pad, x.shape[1]), dtype=np.float64)
    padded[pad : pad + frames] = x

    max_abs = 0.0
    last_start = max(0, padded.shape[0] - block)
    starts = list(range(0, last_start + 1, hop))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    trusted_lo, trusted_hi = pad * factor, (pad + window) * factor
    for start in starts:
        chunk = padded[start : start + block]
        up = _upsample_chunk(chunk, factor)
        peak = float(np.abs(up[trusted_lo:trusted_hi]).max())
        if peak > max_abs:
            max_abs = peak
    if max_abs <= 0.0:
        return None
    return float(20.0 * np.log10(max_abs))


def gain_to_reach_target(
    measured_lufs: float | None, target_lufs: float, max_gain_db: float
) -> float:
    """Compute a clamped linear gain that moves ``measured`` toward ``target``."""
    if measured_lufs is None or not np.isfinite(measured_lufs):
        return 1.0
    db = target_lufs - measured_lufs
    db = float(np.clip(db, -max_gain_db, max_gain_db))
    return float(10.0 ** (db / 20.0))
