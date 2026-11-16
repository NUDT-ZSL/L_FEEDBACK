"""Streaming simplified ITU-R BS.1770 K-weighted loudness meter."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


class Biquad:
    """Causal direct-form-I biquad with persistent state."""

    def __init__(
        self, numerator: np.ndarray, denominator: np.ndarray
    ) -> None:
        numerator = np.asarray(numerator, dtype=np.float64)
        denominator = np.asarray(denominator, dtype=np.float64)
        if abs(denominator[0]) < 1.0e-20:
            raise ValueError("biquad a0 must not be zero")
        self.b = numerator / denominator[0]
        self.a = denominator / denominator[0]
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        output = np.empty(len(samples), dtype=np.float64)
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2
        for index, value in enumerate(samples):
            result = (
                b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            )
            x2 = x1
            x1 = value
            y2 = y1
            y1 = result
            output[index] = result
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        return output


def high_shelf_coefficients(
    sample_rate: int,
    gain_db: float = 4.0,
    center_hz: float = 1681.974450955533,
    slope: float = 1.0,
) -> tuple:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * center_hz / float(sample_rate)
    cosine = math.cos(omega)
    alpha = (
        math.sin(omega)
        / 2.0
        * math.sqrt(
            (amplitude + 1.0 / amplitude)
            * (1.0 / slope - 1.0)
            + 2.0
        )
    )
    root = 2.0 * math.sqrt(amplitude) * alpha
    b0 = amplitude * (
        (amplitude + 1.0)
        + (amplitude - 1.0) * cosine
        + root
    )
    b1 = -2.0 * amplitude * (
        (amplitude - 1.0) + (amplitude + 1.0) * cosine
    )
    b2 = amplitude * (
        (amplitude + 1.0)
        + (amplitude - 1.0) * cosine
        - root
    )
    a0 = (
        (amplitude + 1.0)
        - (amplitude - 1.0) * cosine
        + root
    )
    a1 = 2.0 * (
        (amplitude - 1.0) - (amplitude + 1.0) * cosine
    )
    a2 = (
        (amplitude + 1.0)
        - (amplitude - 1.0) * cosine
        - root
    )
    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def high_pass_coefficients(
    sample_rate: int, center_hz: float = 38.13547087602444, q: float = 0.5
) -> tuple:
    omega = 2.0 * math.pi * center_hz / float(sample_rate)
    cosine = math.cos(omega)
    alpha = math.sin(omega) / (2.0 * q)
    b0 = (1.0 + cosine) / 2.0
    b1 = -(1.0 + cosine)
    b2 = (1.0 + cosine) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha
    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


class LoudnessMeter:
    """Simplified K-weighted integrated loudness with BS.1770 gating."""

    def __init__(self, sample_rate: int, channels: int = 2) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        shelf_b, shelf_a = high_shelf_coefficients(self.sample_rate)
        highpass_b, highpass_a = high_pass_coefficients(self.sample_rate)
        self.shelf_filters = [
            Biquad(shelf_b, shelf_a) for _ in range(self.channels)
        ]
        self.highpass_filters = [
            Biquad(highpass_b, highpass_a) for _ in range(self.channels)
        ]
        self.segment_length = max(1, int(round(0.1 * self.sample_rate)))
        self._segment_sum = 0.0
        self._segment_count = 0
        self.segment_energies: List[float] = []

    def process(self, samples: np.ndarray) -> None:
        block = np.asarray(samples, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != self.channels:
            raise ValueError("meter input must be (samples, channels)")
        if len(block) == 0:
            return
        weighted = np.empty_like(block)
        for channel in range(self.channels):
            filtered = self.shelf_filters[channel].process(block[:, channel])
            weighted[:, channel] = self.highpass_filters[channel].process(
                filtered
            )
        energy = np.mean(weighted * weighted, axis=1)
        position = 0
        while position < len(energy):
            needed = self.segment_length - self._segment_count
            take = min(needed, len(energy) - position)
            self._segment_sum += float(
                np.sum(energy[position : position + take])
            )
            self._segment_count += take
            position += take
            if self._segment_count == self.segment_length:
                self.segment_energies.append(
                    self._segment_sum / float(self.segment_length)
                )
                self._segment_sum = 0.0
                self._segment_count = 0

    @staticmethod
    def _loudness_from_power(power: float) -> float:
        if power <= 1.0e-20:
            return -math.inf
        return -0.691 + 10.0 * math.log10(power)

    def integrated_loudness(self) -> Optional[float]:
        if len(self.segment_energies) < 4:
            return None
        segments = np.asarray(self.segment_energies, dtype=np.float64)
        block_power = (
            segments[:-3] + segments[1:-2] + segments[2:-1] + segments[3:]
        ) / 4.0
        absolute_gate = np.array(
            [self._loudness_from_power(value) for value in block_power]
        )
        after_absolute = block_power[absolute_gate > -70.0]
        if len(after_absolute) == 0:
            return -math.inf
        absolute_mean = float(np.mean(after_absolute))
        relative_gate = self._loudness_from_power(absolute_mean) - 10.0
        block_loudness = np.array(
            [self._loudness_from_power(value) for value in block_power]
        )
        after_relative = block_power[block_loudness > relative_gate]
        if len(after_relative) == 0:
            return -math.inf
        return self._loudness_from_power(float(np.mean(after_relative)))
