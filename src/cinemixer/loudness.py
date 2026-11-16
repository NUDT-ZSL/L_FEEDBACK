"""ITU-R BS.1770-4 integrated loudness measurement.

Implements the K-weighting chain (stage 1 high-shelf, stage 2 high-pass),
mean-square energy accumulation over overlapping 400 ms blocks stepped by
100 ms, and the dual-gating procedure (absolute gate at -70 LKFS plus the
relative gate 10 LU below the ungated mean). Mono and stereo sources use the
standard channel weight of 1.0 per channel.
"""

from __future__ import annotations

import math

import numpy as np

ABSOLUTE_GATE_LKFS = -70.0
RELATIVE_GATE_OFFSET_LU = -10.0
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1


def _rbj_biquad_coeffs(
    sample_rate: int,
    kind: str,
    frequency: float,
    gain_db: float = 0.0,
    q: float = 0.7071,
) -> tuple[float, ...]:
    """Return normalized Direct Form I coefficients (b0,b1,b2,a1,a2)."""
    omega = 2.0 * math.pi * frequency / sample_rate
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    amplitude = 10.0 ** (gain_db / 40.0)
    alpha = sin_omega / (2.0 * q)

    if kind == "high_shelf":
        b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega + 2.0 * math.sqrt(amplitude) * alpha)
        b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega)
        b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega - 2.0 * math.sqrt(amplitude) * alpha)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega + 2.0 * math.sqrt(amplitude) * alpha
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega)
        a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega - 2.0 * math.sqrt(amplitude) * alpha
    elif kind == "high_pass":
        b0 = (1.0 + cos_omega) / 2.0
        b1 = -(1.0 + cos_omega)
        b2 = (1.0 + cos_omega) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
    else:  # pragma: no cover - internal call sites are fixed
        raise ValueError(f"unsupported biquad kind: {kind}")
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _apply_biquad(samples: np.ndarray, coeffs: tuple[float, ...]) -> np.ndarray:
    b0, b1, b2, a1, a2 = coeffs
    output = np.empty_like(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(samples):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[i] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return output


def k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting filters to every channel."""
    shelf = _rbj_biquad_coeffs(sample_rate, "high_shelf", 1681.76322283435, gain_db=4.0, q=0.7071752369554196)
    high_pass = _rbj_biquad_coeffs(sample_rate, "high_pass", 38.1354708760245, q=0.5003270373238773)
    weighted = np.empty_like(audio)
    for channel in range(audio.shape[1]):
        filtered = _apply_biquad(audio[:, channel], shelf)
        weighted[:, channel] = _apply_biquad(filtered, high_pass)
    return weighted


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Return integrated loudness in LKFS/LUFS, or ``-inf`` when fully silent."""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.shape[0] == 0:
        return float("-inf")

    weighted = k_weight(audio.astype(np.float64, copy=False), sample_rate)
    block_size = int(round(BLOCK_SECONDS * sample_rate))
    hop_size = int(round(HOP_SECONDS * sample_rate))
    if audio.shape[0] < block_size:
        # Single truncated block for very short renders.
        block_energy = np.mean(weighted**2, axis=0)
        gated_loudness = -0.691 + 10.0 * math.log10(max(block_energy.sum(), 1.0e-12))
        return gated_loudness if gated_loudness >= ABSOLUTE_GATE_LKFS else float("-inf")

    energies = []
    for start in range(0, audio.shape[0] - block_size + 1, hop_size):
        block = weighted[start : start + block_size]
        energies.append(float(np.mean(block**2, axis=0).sum()))
    energies_arr = np.asarray(energies, dtype=np.float64)
    loudness_per_block = -0.691 + 10.0 * np.log10(np.maximum(energies_arr, 1.0e-12))

    absolute_kept = loudness_per_block >= ABSOLUTE_GATE_LKFS
    if not np.any(absolute_kept):
        return float("-inf")

    ungated_mean = float(np.mean(energies_arr[absolute_kept]))
    relative_gate = (-0.691 + 10.0 * math.log10(max(ungated_mean, 1.0e-12))) + RELATIVE_GATE_OFFSET_LU
    final_kept = loudness_per_block >= relative_gate
    if not np.any(final_kept):
        return float("-inf")
    gated_mean = float(np.mean(energies_arr[final_kept]))
    return -0.691 + 10.0 * math.log10(max(gated_mean, 1.0e-12))


def linear_gain_for_target(measured_lufs: float, target_lufs: float) -> float:
    """Linear amplitude gain that moves measured loudness onto the target."""
    if not math.isfinite(measured_lufs):
        return 1.0
    delta_db = target_lufs - measured_lufs
    return float(10.0 ** (delta_db / 20.0))
