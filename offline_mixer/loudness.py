"""ITU-R BS.1770-4 gated integrated loudness measurement and normalization.

Stage 1: high-shelf (pre-filter), stage 2: high-pass. Mean-square energy is
accumulated in 400 ms blocks (75% overlap); an absolute gate at -70 LKFS and
a relative gate 10 LU below the ungated mean implement the two-stage gating.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from . import TARGET_SR

ABSOLUTE_GATE_LKFS = -70.0
RELATIVE_GATE_OFFSET_LU = -10.0
BLOCK_SECONDS = 0.4
BLOCK_OVERLAP = 0.75

# Canonical BS.1770 filter coefficients (RBWS implementation, per ITU-R BS.1770-4).
_STAGE1_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_STAGE1_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_STAGE2_B = np.array([1.0, -2.0, 1.0])
_STAGE2_A = np.array([1.0, -1.99004745483398, 0.99007225036621])


def _k_weight(signal: np.ndarray) -> np.ndarray:
    filtered = lfilter(_STAGE1_B, _STAGE1_A, signal)
    return lfilter(_STAGE2_B, _STAGE2_A, filtered)


def _block_mean_squares(weighted: np.ndarray, sample_rate: int) -> np.ndarray:
    block_len = int(round(BLOCK_SECONDS * sample_rate))
    hop = int(round(BLOCK_SECONDS * (1.0 - BLOCK_OVERLAP) * sample_rate))
    if weighted.size < block_len:
        # Short segment: evaluate a single zero-padded full block.
        padded = np.zeros(block_len, dtype=np.float64)
        padded[: weighted.size] = weighted
        return np.array([np.mean(padded**2)])

    starts = np.arange(0, weighted.size - block_len + 1, hop)
    if starts.size == 0:
        starts = np.array([0])
    ms_values = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        block = weighted[start : start + block_len]
        ms_values[index] = np.mean(block**2)
    return ms_values


def integrated_loudness(signal: np.ndarray, sample_rate: int = TARGET_SR) -> float:
    """Return gated integrated loudness in LUFS; -inf (silence) -> NaN."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or not np.any(signal):
        return float("nan")

    weighted = _k_weight(signal)
    block_ms = _block_mean_squares(weighted, sample_rate)
    block_loudness = -0.691 + 10.0 * np.log10(np.maximum(block_ms, 1e-20))

    absolute_pass = block_loudness > ABSOLUTE_GATE_LKFS
    if not np.any(absolute_pass):
        return float("nan")

    ungated_loudness = -0.691 + 10.0 * math.log10(np.mean(block_ms[absolute_pass]))
    relative_gate = ungated_loudness + RELATIVE_GATE_OFFSET_LU
    gated = block_loudness[absolute_pass & (block_loudness > relative_gate)]
    if gated.size == 0:
        return float("nan")

    gated_ms = 10.0 ** ((gated + 0.691) / 10.0)
    return -0.691 + 10.0 * math.log10(np.mean(gated_ms))


def normalize_loudness(
    signal: np.ndarray, target_lufs: float, sample_rate: int = TARGET_SR
) -> tuple[np.ndarray, float, float, bool]:
    """Normalize a mono signal to target LUFS.

    Returns (normalized_signal, gain_db, measured_lufs, is_silence).
    Silent segments are returned untouched with 0 dB gain.
    """
    measured = integrated_loudness(signal, sample_rate)
    if math.isnan(measured):
        return signal.copy(), 0.0, float("nan"), True

    gain_db = float(target_lufs) - measured
    linear_gain = 10.0 ** (gain_db / 20.0)
    normalized = signal * linear_gain

    # Re-measure to confirm the realized loudness meets the +/-0.5 LU contract.
    realized = integrated_loudness(normalized, sample_rate)
    if not math.isnan(realized):
        measured_out = realized
    else:  # pragma: no cover - numerical fallback
        measured_out = measured + gain_db
    return normalized, gain_db, measured_out, False
