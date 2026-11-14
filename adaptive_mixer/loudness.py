"""ITU-R BS.1770-4 K-weighted loudness analysis (mono), pure NumPy.

The K-weighting network is the cascade of two biquad stages:
  * stage 1: high-shelf,   +3.999 dB @ 1681.974 Hz, Q = 0.7071509
  * stage 2: high-pass,    38.135 Hz,               Q = 0.7071

The IIR cascade is converted once into a truncated FIR impulse response and
applied through overlap-add FFT convolution, keeping every task dependency
inside NumPy (no SciPy / external audio tools).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 48000
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1
ABSOLUTE_GATE_LKFS = -70.0
RELATIVE_GATE_OFFSET_LU = -10.0
OFFSET_DB = -0.691


def _biquad_high_shelf(sample_rate: int, gain_db: float, freq_hz: float,
                       q: float) -> Tuple[np.ndarray, np.ndarray]:
    w0 = 2.0 * np.pi * freq_hz / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    amp = 10.0 ** (gain_db / 40.0)
    cos_w, sin_w, sqrt_a = np.cos(w0), np.sin(w0), np.sqrt(amp)
    b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w)
    b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w - 2.0 * sqrt_a * alpha)
    a0 = (amp + 1.0) - (amp - 1.0) * cos_w + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w)
    a2 = (amp + 1.0) - (amp - 1.0) * cos_w - 2.0 * sqrt_a * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_high_pass(sample_rate: int, freq_hz: float,
                      q: float) -> Tuple[np.ndarray, np.ndarray]:
    w0 = 2.0 * np.pi * freq_hz / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w = np.cos(w0)
    b = np.array([(1.0 + cos_w) / 2.0, -(1.0 + cos_w), (1.0 + cos_w) / 2.0])
    a0 = 1.0 + alpha
    a = np.array([1.0, -2.0 * cos_w / a0, (1.0 - alpha) / a0])
    return b / a0, a


# Canonical BS.1770-4 biquad coefficients at 48 kHz (a0 normalised to 1).
# Stage "pre-filter" merges the 1681.97 Hz high-shelf and 38.1 Hz high-pass;
# the "RLB" stage is the revised low-frequency B-weighting filter.
_CANONICAL_48K = {
    48000: (
        (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
         np.array([1.0, -1.69065929318241, 0.73248077421582])),
        (np.array([1.0, -2.0, 1.0]),
         np.array([1.0, -1.99004745483398, 0.99007225036621])),
    )
}


def _iir_impulse_response(b: np.ndarray, a: np.ndarray,
                          taps: int) -> np.ndarray:
    """Direct-form-I response to a unit impulse (one-time, short)."""
    out = np.zeros(taps, dtype=np.float64)
    # Feed a unit impulse through y[n] = b0*x[n] + sum bk*x[n-k]
    # - sum ak*y[n-k]; with impulse input only the b0 term survives.
    for n in range(taps):
        value = b[0] if n == 0 else 0.0
        for k in range(1, len(b)):
            if n - k == 0:
                value += b[k]
        for k in range(1, len(a)):
            if n - k >= 0:
                value -= a[k] * out[n - k]
        out[n] = value
    return out


def _truncate_tail(ir: np.ndarray, floor: float = 1e-7,
                   run: int = 512) -> np.ndarray:
    peak = np.max(np.abs(ir)) or 1.0
    below = np.abs(ir) < floor * peak
    idx = len(ir)
    consecutive = 0
    for n in range(len(ir)):
        if below[n]:
            consecutive += 1
            if consecutive >= run:
                idx = n - run + 1
                break
        else:
            consecutive = 0
    return ir[: max(idx, 1)]


def k_weighting_ir(sample_rate: int = SAMPLE_RATE,
                   max_taps: int = 16384) -> np.ndarray:
    """Combined, truncated FIR impulse response of the K-weighting network."""
    cache_key = ("kw_ir", sample_rate, max_taps)
    cached = globals().get("_IR_CACHE", {}).get(cache_key)
    if cached is not None:
        return cached
    if sample_rate in _CANONICAL_48K:
        (b1, a1), (b2, a2) = _CANONICAL_48K[sample_rate]
    else:
        b1, a1 = _biquad_high_shelf(sample_rate, 3.999, 1681.974,
                                    0.7071509)
        b2, a2 = _biquad_high_pass(sample_rate, 38.135, 0.7071)
    h1 = _iir_impulse_response(b1, a1, max_taps)
    h2 = _iir_impulse_response(b2, a2, max_taps)
    nfft = 1
    while nfft < len(h1) + len(h2) - 1:
        nfft <<= 1
    combined = np.fft.irfft(np.fft.rfft(h1, nfft)
                            * np.fft.rfft(h2, nfft), nfft)
    combined = combined[: len(h1) + len(h2) - 1]
    result = _truncate_tail(combined)
    globals().setdefault("_IR_CACHE", {})[cache_key] = result
    return result


def _overlap_add(x: np.ndarray, fir: np.ndarray,
                 block: int = 65536) -> np.ndarray:
    fir_len = len(fir)
    nfft = 1
    while nfft < block + fir_len - 1:
        nfft <<= 1
    fir_spec = np.fft.rfft(fir, nfft)
    out = np.zeros(len(x) + fir_len - 1, dtype=np.float64)
    for start in range(0, len(x), block):
        chunk = x[start: start + block]
        spec = np.fft.rfft(chunk, nfft)
        conv = np.fft.irfft(spec * fir_spec, nfft)
        end = min(start + nfft, len(out))
        out[start:end] += conv[: end - start]
    return out[: len(x)]


def k_weight(samples: np.ndarray,
             fir: Optional[np.ndarray] = None) -> np.ndarray:
    if fir is None:
        fir = k_weighting_ir()
    return _overlap_add(np.asarray(samples, dtype=np.float64), fir)


def _gated_mean_square(weighted: np.ndarray, sample_rate: int) -> Optional[float]:
    block = int(round(BLOCK_SECONDS * sample_rate))
    hop = int(round(HOP_SECONDS * sample_rate))
    if len(weighted) < block:
        padded = np.zeros(block, dtype=np.float64)
        padded[: len(weighted)] = weighted
        weighted = padded
    starts = np.arange(0, len(weighted) - block + 1, hop)
    if len(starts) == 0:
        starts = np.array([0])
    # z_f per BS.1770-4: gated mean-square of the K-weighted mono signal
    # (channel weight = 1); absolute level is set by the -0.691 dB offset,
    # which compensates the +0.691 dB K-weighting gain at 1 kHz.
    powers = np.empty(len(starts), dtype=np.float64)
    for i, start in enumerate(starts):
        window = weighted[start: start + block]
        powers[i] = float(np.mean(window * window))
    with np.errstate(divide="ignore"):
        block_loudness = OFFSET_DB + 10.0 * np.log10(powers)
    abs_keep = block_loudness > ABSOLUTE_GATE_LKFS
    if not np.any(abs_keep):
        return None
    relative_gate = OFFSET_DB + 10.0 * np.log10(np.mean(powers[abs_keep])) \
        + RELATIVE_GATE_OFFSET_LU
    rel_keep = block_loudness > relative_gate
    if not np.any(rel_keep):
        return None
    return float(np.mean(powers[rel_keep]))


def integrated_loudness(samples: np.ndarray,
                        sample_rate: int = SAMPLE_RATE,
                        fir: Optional[np.ndarray] = None) -> float:
    """Integrated (gated) loudness in LUFS; ``-inf`` for digital silence."""
    weighted = k_weight(samples, fir)
    mean_power = _gated_mean_square(weighted, sample_rate)
    if mean_power is None or mean_power <= 0.0:
        return float("-inf")
    return OFFSET_DB + 10.0 * np.log10(mean_power)


def short_term_loudness(samples: np.ndarray,
                        sample_rate: int = SAMPLE_RATE,
                        fir: Optional[np.ndarray] = None) -> np.ndarray:
    """400 ms short-term loudness values (100 ms hop), LUFS."""
    weighted = k_weight(samples, fir)
    block = int(round(BLOCK_SECONDS * sample_rate))
    hop = int(round(HOP_SECONDS * sample_rate))
    if len(weighted) < block:
        padded = np.zeros(block, dtype=np.float64)
        padded[: len(weighted)] = weighted
        weighted = padded
    starts = np.arange(0, len(weighted) - block + 1, hop)
    if len(starts) == 0:
        starts = np.array([0])
    values = np.empty(len(starts), dtype=np.float64)
    for i, start in enumerate(starts):
        window = weighted[start: start + block]
        power = float(np.mean(window * window))
        values[i] = OFFSET_DB + 10.0 * np.log10(power) if power > 0.0 \
            else float("-inf")
    return values


def lufs_to_linear_gain(current_lufs: float, target_lufs: float) -> Tuple[float, float]:
    """Return ``(linear_gain, gain_db)`` moving ``current`` onto ``target``."""
    if not np.isfinite(current_lufs):
        return 1.0, 0.0
    gain_db = target_lufs - current_lufs
    return 10.0 ** (gain_db / 20.0), gain_db


@dataclass
class LoudnessReport:
    integrated_lufs: float
    short_term_lufs: np.ndarray
    sample_peak: float


def analyze(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
            fir: Optional[np.ndarray] = None) -> LoudnessReport:
    weighted = k_weight(samples, fir)
    mean_power = _gated_mean_square(weighted, sample_rate)
    integrated = float("-inf") if mean_power is None else \
        OFFSET_DB + 10.0 * np.log10(mean_power)
    return LoudnessReport(
        integrated_lufs=integrated,
        short_term_lufs=short_term_loudness(samples, sample_rate, fir),
        sample_peak=float(np.max(np.abs(samples))) if len(samples) else 0.0,
    )
