"""Granular pitch-preserving time scaling (overlap-add synthesis)."""

from __future__ import annotations

import numpy as np

from . import MAX_RATIO, MIN_RATIO, TARGET_SR

GRAIN_MS = 40.0
HOP_MS = 10.0


class AlignmentError(ValueError):
    """Raised when the required speed ratio lies outside [0.8, 1.25]."""


def time_scale_ratio(source_seconds: float, target_seconds: float) -> float:
    """Speed ratio = source duration / target duration (>1 = accelerate)."""
    if target_seconds <= 0:
        raise AlignmentError("target duration is zero")
    return float(source_seconds) / float(target_seconds)


def ratio_allowed(ratio: float) -> bool:
    return MIN_RATIO <= ratio <= MAX_RATIO


def granular_time_stretch(
    signal: np.ndarray,
    ratio: float,
    sample_rate: int = TARGET_SR,
    target_length: int | None = None,
) -> np.ndarray:
    """Stretch/compress `signal` by `ratio` using windowed overlap-add grains.

    Analysis hop is fixed at 10 ms with 40 ms Hann grains; synthesis hop
    scales with the ratio, which preserves pitch while changing duration.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if target_length is None:
        target_length = int(round(signal.size / ratio))
    if target_length <= 0 or signal.size == 0:
        return np.zeros(max(target_length, 0), dtype=np.float64)

    grain_len = max(2, int(round(GRAIN_MS * sample_rate / 1000.0)))
    analysis_hop = max(1, int(round(HOP_MS * sample_rate / 1000.0)))
    synth_hop = max(1, int(round(analysis_hop * ratio)))
    window = np.hanning(grain_len)

    output = np.zeros(target_length + grain_len, dtype=np.float64)
    weight = np.zeros_like(output)

    analysis_positions = np.arange(0, signal.size, analysis_hop)
    for grain_index, analysis_start in enumerate(analysis_positions):
        synth_start = grain_index * synth_hop
        if synth_start >= target_length:
            break
        grain = np.zeros(grain_len, dtype=np.float64)
        take = min(grain_len, signal.size - analysis_start)
        if take > 0:
            grain[:take] = signal[analysis_start : analysis_start + take]
        grain *= window
        end = min(synth_start + grain_len, output.size)
        output[synth_start:end] += grain[: end - synth_start]
        weight[synth_start:end] += window[: end - synth_start]

    valid = weight > 1e-8
    output[valid] /= weight[valid]
    return output[:target_length]


def align_segment(
    signal: np.ndarray, target_seconds: float, sample_rate: int = TARGET_SR
) -> tuple[np.ndarray, float, dict | None]:
    """Align a clip to the timeline target duration.

    Returns (aligned_signal, ratio, error). `error` is None on success,
    otherwise a serializable reason dict and the original signal is returned.
    """
    source_seconds = signal.size / float(sample_rate)
    target_length = int(round(target_seconds * sample_rate))
    if target_length <= 0:
        return (
            signal.copy(),
            float("nan"),
            {
                "code": "E3001",
                "reason": "zero_or_negative_target_duration",
                "ratio": None,
                "allowed_range": [MIN_RATIO, MAX_RATIO],
            },
        )

    ratio = time_scale_ratio(source_seconds, target_seconds)
    if not ratio_allowed(ratio):
        return (
            signal.copy(),
            ratio,
            {
                "code": "E3001",
                "reason": "speed_ratio_out_of_range",
                "ratio": round(ratio, 4),
                "allowed_range": [MIN_RATIO, MAX_RATIO],
            },
        )

    aligned = granular_time_stretch(signal, ratio, sample_rate, target_length)
    return aligned, ratio, None
