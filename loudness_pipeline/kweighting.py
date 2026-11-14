"""BS.1770-4 K-weighting filter derivation and offline filtering.

The filter is specified at 48 kHz by ITU-R BS.1770-4. For other sample rates
the standard requires coefficients that preserve the same analog response.
The derivation below uses the documented analog prototype parameters and a
bilinear transform, then cascades the high-shelf and RLB high-pass stages.

The actual filtering is done in the frequency domain. This keeps the runtime
dependency set to NumPy while still processing long recordings efficiently.
Zero padding absorbs the IIR tail before an FFT's circular boundary wraps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


_SHELF_F0_HZ = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_VB_EXPONENT = 0.4996667741545416

_RLB_F0_HZ = 38.13547087602444
_RLB_Q = 0.5003270373238773


@dataclass(frozen=True)
class Biquad:
    """Causal second-order transfer function, normalized to a[0] == 1."""

    b: Tuple[float, float, float]
    a: Tuple[float, float, float]


@dataclass(frozen=True)
class KWeightingCoefficients:
    shelf: Biquad
    rlb: Biquad


def _bilinear_high_shelf(sample_rate: int) -> Biquad:
    k = math.tan(math.pi * _SHELF_F0_HZ / float(sample_rate))
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh ** _SHELF_VB_EXPONENT
    denominator = 1.0 + k / _SHELF_Q + k * k
    b0 = (vh + vb * k / _SHELF_Q + k * k) / denominator
    b1 = 2.0 * (k * k - vh) / denominator
    b2 = (vh - vb * k / _SHELF_Q + k * k) / denominator
    a1 = 2.0 * (k * k - 1.0) / denominator
    a2 = (1.0 - k / _SHELF_Q + k * k) / denominator
    return Biquad((b0, b1, b2), (1.0, a1, a2))


def _bilinear_rlb_high_pass(sample_rate: int) -> Biquad:
    k = math.tan(math.pi * _RLB_F0_HZ / float(sample_rate))
    denominator = 1.0 + k / _RLB_Q + k * k
    a1 = 2.0 * (k * k - 1.0) / denominator
    a2 = (1.0 - k / _RLB_Q + k * k) / denominator
    return Biquad((1.0, -2.0, 1.0), (1.0, a1, a2))


def kweighting_coefficients(sample_rate: int) -> KWeightingCoefficients:
    """Derive the two BS.1770 K-weighting biquads for a sample rate."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return KWeightingCoefficients(
        shelf=_bilinear_high_shelf(sample_rate),
        rlb=_bilinear_rlb_high_pass(sample_rate),
    )


def filter_kweighting(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply K-weighting to a ``(frames, channels)`` float64 array.

    Filtering is linear and time-invariant per channel. The same result is
    produced for every invocation because no randomized or approximate
    iterative stage is used.
    """

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("samples must be a two-dimensional array")
    if audio.shape[0] == 0:
        return audio.copy()

    coefficients = kweighting_coefficients(sample_rate)
    shelf = coefficients.shelf
    rlb = coefficients.rlb
    b = np.convolve(shelf.b, rlb.b)
    a = np.convolve(shelf.a, rlb.a)

    frame_count = audio.shape[0]
    padding = max(sample_rate, 4096)
    fft_length = frame_count + padding
    fft_length = 1 << (fft_length - 1).bit_length()

    frequencies = np.fft.rfftfreq(fft_length, d=1.0 / float(sample_rate))
    z_inverse = np.exp(-2j * np.pi * frequencies / float(sample_rate))
    numerator = np.zeros_like(z_inverse, dtype=np.complex128)
    denominator = np.zeros_like(z_inverse, dtype=np.complex128)
    for coefficient in b[::-1]:
        numerator = numerator * z_inverse + coefficient
    for coefficient in a[::-1]:
        denominator = denominator * z_inverse + coefficient
    response = numerator / denominator

    filtered = np.empty_like(audio, dtype=np.float64)
    for channel in range(audio.shape[1]):
        spectrum = np.fft.rfft(audio[:, channel], n=fft_length)
        spectrum *= response
        filtered[:, channel] = np.fft.irfft(spectrum, n=fft_length)[:frame_count]
    return filtered

