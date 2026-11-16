"""Band-limited resampling to 48 kHz mono (anti-aliased)."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import TARGET_SR


def resample_to_48k(signal: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample mono float signal to 48 kHz with band-limited interpolation."""
    signal = np.asarray(signal, dtype=np.float64)
    if source_sr == TARGET_SR:
        return signal.copy()
    if signal.size == 0:
        return signal.copy()

    factor = gcd(TARGET_SR, source_sr)
    up = TARGET_SR // factor
    down = source_sr // factor
    # resample_poly designs a windowed sinc low-pass at min(1/up, 1/down) of
    # the up-sampled Nyquist rate: band-limited interpolation with anti-alias
    # protection for both up- and down-sampling ratios.
    return resample_poly(signal, up, down).astype(np.float64)


def resample_status(source_sr: int) -> dict:
    """Report fragment describing the conversion for the final report."""
    return {
        "source_sample_rate": int(source_sr),
        "target_sample_rate": TARGET_SR,
        "resampled": bool(source_sr != TARGET_SR),
        "method": "band_limited_fir_polyphase" if source_sr != TARGET_SR else "passthrough",
    }
