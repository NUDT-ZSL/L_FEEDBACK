from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
LOUDNESS_OFFSET = -0.691
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1


def _k_weighting_coefficients(
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the two BS.1770-4 biquad sections for an arbitrary rate."""

    # Stage 1: high-shelf pre-filter.
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf_b = np.array(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    shelf_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )

    # Stage 2: RLB high-pass filter.
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / sample_rate)
    a0 = 1.0 + k / q + k * k
    highpass_b = np.array(
        [1.0 / a0, -2.0 / a0, 1.0 / a0], dtype=np.float64
    )
    highpass_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    return shelf_b, shelf_a, highpass_b, highpass_a


def _biquad_response(
    b: np.ndarray, a: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    z_inv = np.exp(-1j * omega)
    return (b[0] + b[1] * z_inv + b[2] * z_inv * z_inv) / (
        a[0] + a[1] * z_inv + a[2] * z_inv * z_inv
    )


def _zero_state_k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the cascaded biquads with a zero-state response.

    A zero-padded FFT is used so the implementation only depends on NumPy.
    The IIR tail is discarded into the padding, preserving the causal
    zero-state response without a Python per-sample loop.
    """

    frame_count = samples.shape[0]
    if frame_count == 0:
        return samples.copy()
    nfft = 1
    needed = max(2 * frame_count, frame_count + 8192)
    while nfft < needed:
        nfft <<= 1

    shelf_b, shelf_a, hp_b, hp_a = _k_weighting_coefficients(sample_rate)
    omega = 2.0 * math.pi * np.arange(nfft // 2 + 1) / nfft
    response = _biquad_response(shelf_b, shelf_a, omega)
    response *= _biquad_response(hp_b, hp_a, omega)

    spectrum = np.fft.rfft(samples, n=nfft, axis=0)
    filtered = np.fft.irfft(spectrum * response[:, None], n=nfft, axis=0)
    return np.asarray(filtered[:frame_count], dtype=np.float64)


def channel_weight(label: str) -> float:
    upper = label.upper()
    if upper == "LFE":
        return 0.0
    if upper in ("BL", "BR", "SL", "SR", "BC"):
        return 1.41
    return 1.0


def integrated_loudness(
    samples: np.ndarray,
    sample_rate: int,
    channel_labels: Sequence[str],
) -> Optional[float]:
    """Calculate gated integrated loudness in LUFS.

    Returns None for digital silence or when no gating block is measurable.
    """

    if samples.ndim != 2 or samples.shape[1] != len(channel_labels):
        raise ValueError("channel labels do not match audio shape")
    if samples.shape[0] == 0:
        return None
    if not np.any(samples):
        return None

    weighted = _zero_state_k_weight(samples, sample_rate)
    block_frames = max(
        1, int(round(BLOCK_SECONDS * sample_rate))
    )
    hop_frames = max(1, int(round(HOP_SECONDS * sample_rate)))
    frame_count = weighted.shape[0]

    if frame_count < block_frames:
        block_frames = frame_count
        starts = np.array([0], dtype=np.int64)
    else:
        starts = np.arange(
            0, frame_count - block_frames + 1, hop_frames, dtype=np.int64
        )

    squared = weighted * weighted
    cumulative = np.vstack(
        [np.zeros((1, weighted.shape[1]), dtype=np.float64),
         np.cumsum(squared, axis=0, dtype=np.float64)]
    )
    means = (
        cumulative[starts + block_frames] - cumulative[starts]
    ) / float(block_frames)
    weights = np.array(
        [channel_weight(label) for label in channel_labels],
        dtype=np.float64,
    )
    block_energy = means @ weights
    valid = block_energy > 0.0
    if not np.any(valid):
        return None

    block_loudness = np.full(block_energy.shape, -np.inf, dtype=np.float64)
    block_loudness[valid] = (
        LOUDNESS_OFFSET + 10.0 * np.log10(block_energy[valid])
    )
    absolute = valid & (block_loudness > ABSOLUTE_GATE_LUFS)
    if not np.any(absolute):
        return None

    relative_energy = float(np.mean(block_energy[absolute]))
    relative_threshold = (
        LOUDNESS_OFFSET
        + 10.0 * math.log10(relative_energy)
        + RELATIVE_GATE_LU
    )
    gated = absolute & (block_loudness > relative_threshold)
    if not np.any(gated):
        return None

    integrated_energy = float(np.mean(block_energy[gated]))
    return LOUDNESS_OFFSET + 10.0 * math.log10(integrated_energy)
