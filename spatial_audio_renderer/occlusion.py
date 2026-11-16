"""AABB occlusion counting and stateful low-pass filtering."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .geometry import segment_intersects_aabb_array
from .scene import OccluderSpec


def occlusion_curves(
    occluders: Sequence[OccluderSpec],
    listener_positions: np.ndarray,
    source_positions: np.ndarray,
    sample_rate: int,
    base_cutoff_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample layer count, cutoff, and linear occlusion gain."""
    count = len(listener_positions)
    depths = np.zeros(count, dtype=np.int32)
    cutoffs = np.full(count, float(base_cutoff_hz), dtype=np.float64)
    gains = np.ones(count, dtype=np.float64)
    for occluder in occluders:
        hit = segment_intersects_aabb_array(
            listener_positions,
            source_positions,
            occluder.minimum,
            occluder.maximum,
        )
        if not np.any(hit):
            continue
        depths[hit] += 1
        gains[hit] *= float(occluder.attenuation)
        relative = min(
            0.97,
            max(0.02, float(occluder.cutoff_hz) / max(base_cutoff_hz, 1.0)),
        )
        cutoffs[hit] *= relative
    cutoffs = np.clip(
        cutoffs, 80.0, 0.45 * float(sample_rate)
    )
    return depths, cutoffs, gains


class OcclusionFilter:
    """One-pole low-pass plus smoothed attenuation with persistent state."""

    def __init__(self, sample_rate: int, smooth_seconds: float = 0.012) -> None:
        self.sample_rate = int(sample_rate)
        self.gain_coefficient = math.exp(
            -1.0 / max(1.0, smooth_seconds * self.sample_rate)
        )
        self.lowpass_state = 0.0
        self.gain_state = 1.0
        self._initialized = False

    def process(
        self,
        samples: np.ndarray,
        cutoff_hz: np.ndarray,
        target_gain: np.ndarray,
    ) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64)
        cutoff_hz = np.asarray(cutoff_hz, dtype=np.float64)
        target_gain = np.asarray(target_gain, dtype=np.float64)
        if not (
            len(samples) == len(cutoff_hz) == len(target_gain)
        ):
            raise ValueError("occlusion inputs must have identical lengths")
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float64)
        if not self._initialized:
            self.gain_state = float(target_gain[0])
            self._initialized = True
        output = np.empty(len(samples), dtype=np.float64)
        alpha = np.exp(
            -2.0 * math.pi * cutoff_hz / float(self.sample_rate)
        )
        lowpass = self.lowpass_state
        gain = self.gain_state
        for index in range(len(samples)):
            lowpass = (
                (1.0 - alpha[index]) * samples[index]
                + alpha[index] * lowpass
            )
            gain = (
                target_gain[index]
                + self.gain_coefficient
                * (gain - target_gain[index])
            )
            output[index] = lowpass * gain
        self.lowpass_state = lowpass
        self.gain_state = gain
        return output
