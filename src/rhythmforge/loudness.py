"""Reproducible integrated loudness normalization and peak protection.

The measurement follows the ITU-R BS.1770 / EBU R 128 structure:

1. K-weighting (a high-shelf stage followed by a high-pass stage),
2. channel-weighted mean square energy,
3. 400 ms blocks with 75 % overlap and an absolute + relative gate,
4. integrated loudness in LUFS from the gated blocks.

The gain returned by :func:`loudness_gain` is then reused unchanged for the
individual stems so that ``sum(stems) == master`` remains true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

INTEGRATION_SECONDS = 0.4
OVERLAP_RATIO = 0.75
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_OFFSET_LU = -10.0
EPS = 1e-12


def _k_weight(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """BS.1770 K-weighting approximated by two documented biquad stages."""
    # Stage 1: high shelf (pre-filter), ~+4 dB above roughly 1.5 kHz.
    shelf_f = 1681.974450955533
    shelf_gain = 3.999508694409066
    # Stage 2: high-pass at ~38 Hz.
    hp_f = 38.13547087602444
    q_shelf = 0.7071752369554196
    q_hp = 0.5003270373238773

    def biquad(frequency: float, q: float, gain_db: float,
               kind: str) -> tuple[np.ndarray, np.ndarray]:
        w0 = 2.0 * np.pi * frequency / sample_rate
        alpha = np.sin(w0) / (2.0 * q)
        if kind == "highpass":
            a0 = 1.0 + alpha
            cos_w0 = np.cos(w0)
            b = np.array([
                (1.0 + np.cos(w0)) / 2.0,
                -(1.0 + np.cos(w0)),
                (1.0 + np.cos(w0)) / 2.0,
            ]) / a0
            a = np.array([
                1.0,
                -2.0 * cos_w0 / a0,
                (1.0 - alpha) / a0,
            ])
            return b, a

        # RBJ high-shelf biquad (Audio EQ cookbook), normalized by common a0.
        # DC gain is 0 dB; Nyquist gain equals ``gain_db`` (~+4 dB K-weight).
        amplitude = 10.0 ** (gain_db / 40.0)
        cos_w0 = np.cos(w0)
        sqrt_a = np.sqrt(amplitude)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_w0 \
            + 2.0 * sqrt_a * alpha
        b = np.array([
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_w0
                         + 2.0 * sqrt_a * alpha),
            -2.0 * amplitude * ((amplitude - 1.0)
                                + (amplitude + 1.0) * cos_w0),
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_w0
                         - 2.0 * sqrt_a * alpha),
        ]) / a0
        a = np.array([
            1.0,
            2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_w0) / a0,
            ((amplitude + 1.0) - (amplitude - 1.0) * cos_w0
             - 2.0 * sqrt_a * alpha) / a0,
        ])
        return b, a

    shelf_b, shelf_a = biquad(shelf_f, q_shelf, shelf_gain, "shelf")
    hp_b, hp_a = biquad(hp_f, q_hp, 0.0, "highpass")
    filtered = lfilter(shelf_b, shelf_a, mono.astype(np.float64))
    filtered = lfilter(hp_b, hp_a, filtered)
    return filtered


def _block_energies(stereo: np.ndarray, sample_rate: int) -> np.ndarray:
    block = max(1, int(round(INTEGRATION_SECONDS * sample_rate)))
    hop = max(1, int(round(block * (1.0 - OVERLAP_RATIO))))
    if stereo.shape[0] < block:
        padded = np.zeros((block, stereo.shape[1]), dtype=np.float64)
        padded[:stereo.shape[0]] = stereo
        stereo = padded
    weighted = np.stack([_k_weight(stereo[:, channel], sample_rate)
                         for channel in range(stereo.shape[1])], axis=1)
    starts = np.arange(0, weighted.shape[0] - block + 1, hop)
    energies = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        window = weighted[start:start + block]
        # Equal channel weight (1.0); mean square over frames and channels.
        energies[index] = float(np.mean(np.square(window)))
    return energies


def integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure gated integrated loudness in LUFS."""
    energies = _block_energies(np.asarray(audio, dtype=np.float64), sample_rate)
    block_loudness = -0.691 + 10.0 * np.log10(energies + EPS)
    absolute_mask = block_loudness > ABSOLUTE_GATE_LUFS
    if not np.any(absolute_mask):
        return float(-np.inf)
    interim = -0.691 + 10.0 * np.log10(np.mean(energies[absolute_mask]) + EPS)
    relative_threshold = interim + RELATIVE_GATE_OFFSET_LU
    final_mask = absolute_mask & (block_loudness > relative_threshold)
    if not np.any(final_mask):
        final_mask = absolute_mask
    gated_mean = float(np.mean(energies[final_mask]))
    return float(-0.691 + 10.0 * np.log10(gated_mean + EPS))


@dataclass
class NormalizationResult:
    input_lufs: float
    target_lufs: float
    loudness_gain_linear: float
    loudness_gain_db: float
    peak_gain_linear: float
    peak_gain_db: float
    applied_gain_linear: float
    applied_gain_db: float
    output_lufs: float
    output_peak: float
    peak_limit_linear: float

    def as_dict(self) -> dict:
        return {
            "input_lufs": None if np.isneginf(self.input_lufs) else self.input_lufs,
            "target_lufs": self.target_lufs,
            "loudness_gain_db": self.loudness_gain_db,
            "peak_protection_gain_db": self.peak_gain_db,
            "applied_gain_db": self.applied_gain_db,
            "output_lufs": None if np.isneginf(self.output_lufs) else self.output_lufs,
            "output_sample_peak_linear": self.output_peak,
            "peak_limit_linear": self.peak_limit_linear,
        }


def loudness_gain(audio: np.ndarray, sample_rate: int, target_lufs: float,
                  true_peak_dbfs: float) -> NormalizationResult:
    """Compute one reproducible gain satisfying loudness and peak constraints."""
    input_lufs = integrated_lufs(audio, sample_rate)
    if np.isneginf(input_lufs):
        loudness_linear = 1.0
    else:
        loudness_linear = float(10.0 ** ((target_lufs - input_lufs) / 20.0))

    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak_limit = float(10.0 ** (true_peak_dbfs / 20.0))
    projected_peak = current_peak * loudness_linear
    if projected_peak > peak_limit and projected_peak > EPS:
        peak_linear = peak_limit / projected_peak
    else:
        peak_linear = 1.0
    applied = loudness_linear * peak_linear

    output = audio.astype(np.float64) * applied
    output_lufs = integrated_lufs(output.astype(np.float32), sample_rate)
    output_peak = float(np.max(np.abs(output))) if output.size else 0.0

    def to_db(value: float) -> float:
        return float(20.0 * np.log10(value + EPS))

    return NormalizationResult(
        input_lufs=input_lufs,
        target_lufs=float(target_lufs),
        loudness_gain_linear=loudness_linear,
        loudness_gain_db=to_db(loudness_linear),
        peak_gain_linear=peak_linear,
        peak_gain_db=to_db(peak_linear),
        applied_gain_linear=applied,
        applied_gain_db=to_db(applied),
        output_lufs=output_lufs,
        output_peak=output_peak,
        peak_limit_linear=peak_limit,
    )
