"""True-peak look-ahead limiter.

True-peak detection follows the ITU-R BS.1770 measurement convention: the
signal is up-sampled 4x through a windowed-sinc anti-aliasing low-pass and
the inter-sample maximum is measured there.  The limiter is an offline
look-ahead design with instantaneous attack and a time-constant release;
gain reduction is applied at the over-sampled rate and the signal is
low-pass filtered and decimated back to the base rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

TRUE_PEAK_CEILING_DBTP = -1.0
UPSAMPLE_FACTOR = 4
RELEASE_MS = 50.0


def _antialias_fir(up: int = UPSAMPLE_FACTOR, taps: int = 193) -> np.ndarray:
    """Normalised windowed-sinc LP used for both up- and down-sampling."""
    half = (taps - 1) / 2.0
    n = np.arange(taps) - half
    cutoff = np.pi / up
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(n == 0.0, cutoff / np.pi,
                     np.sin(cutoff * n) / (np.pi * n))
    window = np.hamming(taps)
    h *= window
    h /= np.sum(h)
    return h


_FIR_CACHE: Optional[np.ndarray] = None


def _fir() -> np.ndarray:
    global _FIR_CACHE
    if _FIR_CACHE is None:
        _FIR_CACHE = _antialias_fir()
    return _FIR_CACHE


def _upsample(samples: np.ndarray, up: int, fir: np.ndarray) -> np.ndarray:
    # Polyphase zero-stuffing: the interpolation filter carries the
    # insertion gain ``up`` so over-sampled amplitudes match the base signal.
    zero_stuffed = np.zeros(len(samples) * up, dtype=np.float64)
    zero_stuffed[::up] = samples * up
    return np.convolve(zero_stuffed, fir, mode="same")


def _downsample(samples: np.ndarray, up: int, fir: np.ndarray) -> np.ndarray:
    filtered = np.convolve(samples, fir, mode="same")
    return filtered[::up]


def true_peak(samples: np.ndarray,
              up: int = UPSAMPLE_FACTOR) -> float:
    """True-peak amplitude (linear) measured on the 4x over-sampled signal."""
    if len(samples) == 0:
        return 0.0
    over = _upsample(np.asarray(samples, dtype=np.float64), up, _fir())
    return float(np.max(np.abs(over)))


def true_peak_db(samples: np.ndarray, up: int = UPSAMPLE_FACTOR) -> float:
    peak = true_peak(samples, up)
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


@dataclass
class LimiterResult:
    output: np.ndarray
    gain_reduction_db: float
    true_peak_db: float
    limited: bool


def limit(samples: np.ndarray,
          ceiling_db: float = TRUE_PEAK_CEILING_DBTP,
          sample_rate: int = 48000,
          up: int = UPSAMPLE_FACTOR,
          release_ms: float = RELEASE_MS) -> LimiterResult:
    samples = np.asarray(samples, dtype=np.float64)
    fir = _fir()
    ceiling = 10.0 ** (ceiling_db / 20.0)
    over = _upsample(samples, up, fir)
    over_rate = sample_rate * up
    peak = np.max(np.abs(over)) if len(over) else 0.0

    if peak <= ceiling:
        return LimiterResult(samples.copy(), 0.0,
                             true_peak_db(samples), False)

    # Instant-attack, release-limited peak envelope implementing
    # e[n] = max(|x[n]|, e[n-1] * coeff).  Each instant attack restarts a
    # decay segment, so the envelope is a concatenation of per-segment
    # cumulative maxima, fully vectorised with no Python sample loop.
    release_coeff = float(np.exp(-1.0 /
                                 (release_ms / 1000.0 * over_rate)))
    magnitude = np.abs(over)
    if not np.any(magnitude > 0.0):
        return LimiterResult(samples.copy(), 0.0,
                             true_peak_db(samples), False)
    env_prev = np.concatenate(([0.0], magnitude[:-1])) * release_coeff
    attacks = magnitude > env_prev
    segment_id = np.cumsum(attacks)
    segment_start = np.maximum.accumulate(
        np.where(attacks, np.arange(len(over)), 0))
    ages = np.arange(len(over)) - segment_start
    order = np.argsort(segment_id, kind="stable")
    sorted_mag = magnitude[order]
    sorted_seg = segment_id[order]
    # Reset cumulative maximum at every segment boundary.
    reset = np.concatenate(([True], np.diff(sorted_seg) != 0))
    sorted_mag = np.where(reset, sorted_mag,
                          np.maximum.accumulate(
                              np.where(reset, -np.inf, sorted_mag)))
    peaks = np.empty(len(over), dtype=np.float64)
    peaks[order] = sorted_mag
    start_level = np.zeros(len(over), dtype=np.float64)
    start_level[segment_start] = magnitude[segment_start]
    envelope = np.maximum(peaks,
                          start_level * release_coeff ** ages)
    required = np.where(envelope > ceiling, ceiling / np.maximum(envelope,
                                                                 1e-20), 1.0)
    max_reduction = float(-20.0 * np.log10(np.min(required)))

    limited_over = over * required
    decimated = _downsample(limited_over, up, fir)
    output = _fit_length(decimated, len(samples))

    # Safety net for filter overshoot: uniform trim if any TP remains above.
    measured = true_peak(output, up)
    if measured > ceiling:
        trim = ceiling / measured * 0.9995
        output *= trim
        max_reduction += float(-20.0 * np.log10(trim))
        measured = true_peak(output, up)

    return LimiterResult(
        output=output,
        gain_reduction_db=max_reduction,
        true_peak_db=20.0 * np.log10(max(measured, 1e-20)),
        limited=True,
    )


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) == length:
        return samples
    if len(samples) > length:
        return samples[:length]
    padded = np.zeros(length, dtype=np.float64)
    padded[: len(samples)] = samples
    return padded
