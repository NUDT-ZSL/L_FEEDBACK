"""Binaural cues and a compact minimum-phase HRTF approximation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CueBlock:
    itd_seconds: np.ndarray
    left_ild: np.ndarray
    right_ild: np.ndarray
    left_shadow: np.ndarray
    right_shadow: np.ndarray
    elevation: np.ndarray
    lateral: np.ndarray


@dataclass
class _FilterParameters:
    cutoff_hz: float
    notch_hz: float
    notch_radius: float


@dataclass
class _EarState:
    lowpass: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0


class FractionalDelayLine:
    """A cubic-interpolating delay line for continuously varying ITD."""

    def __init__(self, max_delay_samples: float) -> None:
        self.capacity = max(8, int(math.ceil(max_delay_samples)) + 8)
        self.buffer = np.zeros(self.capacity, dtype=np.float64)
        self.write_index = 0

    def _read(self, position: float) -> float:
        base = int(math.floor(position))
        fraction = position - base
        p0 = self.buffer[(base - 1) % self.capacity]
        p1 = self.buffer[base % self.capacity]
        p2 = self.buffer[(base + 1) % self.capacity]
        p3 = self.buffer[(base + 2) % self.capacity]
        fraction2 = fraction * fraction
        fraction3 = fraction2 * fraction
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * fraction
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * fraction2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * fraction3
        )

    def process(
        self,
        samples: np.ndarray,
        left_delay: np.ndarray,
        right_delay: np.ndarray,
    ) -> tuple:
        left = np.empty(len(samples), dtype=np.float64)
        right = np.empty(len(samples), dtype=np.float64)
        for index, sample in enumerate(samples):
            self.buffer[self.write_index] = float(sample)
            left[index] = self._read(
                self.write_index - float(left_delay[index])
            )
            right[index] = self._read(
                self.write_index - float(right_delay[index])
            )
            self.write_index = (self.write_index + 1) % self.capacity
        return left, right


class MinimumPhaseHRTF:
    """Direction-dependent head-shadow low-pass plus pinna notch.

    Both IIR sections have all poles and zeros inside (or, for a limiting
    case, on) the unit circle and therefore belong to the minimum-phase
    family used here as a lightweight HRTF approximation.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self.state_left = _EarState()
        self.state_right = _EarState()
        self.previous_left = self._parameters(0.0, 0.0, 0.0)
        self.previous_right = self._parameters(0.0, 0.0, 0.0)
        self._initialized = False

    def _parameters(
        self, shadow: float, elevation: float, lateral: float
    ) -> _FilterParameters:
        nyquist_limit = 0.45 * self.sample_rate
        base_cutoff = min(18000.0, nyquist_limit)
        cutoff = base_cutoff * (1.0 - 0.82 * max(0.0, shadow) ** 1.3)
        cutoff = float(np.clip(cutoff, 900.0, nyquist_limit))
        notch_hz = 7200.0 + 1800.0 * min(
            1.0, abs(elevation) / (0.5 * math.pi)
        )
        notch_hz = float(np.clip(notch_hz, 1500.0, nyquist_limit))
        notch_radius = 0.48 + 0.18 * min(1.0, abs(lateral))
        return _FilterParameters(cutoff, notch_hz, notch_radius)

    def _block_parameters(
        self,
        target: _FilterParameters,
        previous: _FilterParameters,
        length: int,
    ) -> tuple:
        amount = (np.arange(length, dtype=np.float64) + 1.0) / max(
            1, length
        )
        cutoff = previous.cutoff_hz + amount * (
            target.cutoff_hz - previous.cutoff_hz
        )
        notch_hz = previous.notch_hz + amount * (
            target.notch_hz - previous.notch_hz
        )
        notch_radius = previous.notch_radius + amount * (
            target.notch_radius - previous.notch_radius
        )
        nyquist_limit = 0.45 * self.sample_rate
        cutoff = np.clip(cutoff, 20.0, nyquist_limit)
        notch_hz = np.clip(notch_hz, 20.0, nyquist_limit)
        notch_radius = np.clip(notch_radius, 0.0, 0.95)
        lowpass_alpha = np.exp(
            -2.0 * math.pi * cutoff / float(self.sample_rate)
        )
        omega = 2.0 * math.pi * notch_hz / float(self.sample_rate)
        cosine = np.cos(omega)
        pole_radius = np.maximum(notch_radius + 0.28, 0.88)
        pole_radius = np.minimum(pole_radius, 0.985)
        b0 = np.ones(length, dtype=np.float64)
        b1 = -2.0 * notch_radius * cosine
        b2 = notch_radius * notch_radius
        a1 = -2.0 * pole_radius * cosine
        a2 = pole_radius * pole_radius
        dc = (1.0 + b1 + b2) / (1.0 + a1 + a2)
        return lowpass_alpha, b0 / dc, b1 / dc, b2 / dc, a1, a2

    @staticmethod
    def _process_ear(
        samples: np.ndarray,
        coefficients: tuple,
        state: _EarState,
    ) -> np.ndarray:
        alpha, b0, b1, b2, a1, a2 = coefficients
        output = np.empty(len(samples), dtype=np.float64)
        lowpass = state.lowpass
        x1 = state.x1
        x2 = state.x2
        y1 = state.y1
        y2 = state.y2
        for index in range(len(samples)):
            lowpassed = (
                (1.0 - alpha[index]) * samples[index] + alpha[index] * lowpass
            )
            lowpass = lowpassed
            value = (
                b0[index] * lowpassed
                + b1[index] * x1
                + b2[index] * x2
                - a1[index] * y1
                - a2[index] * y2
            )
            x2 = x1
            x1 = lowpassed
            y2 = y1
            y1 = value
            output[index] = value
        state.lowpass = lowpass
        state.x1 = x1
        state.x2 = x2
        state.y1 = y1
        state.y2 = y2
        return output

    def process(
        self,
        delayed_left: np.ndarray,
        delayed_right: np.ndarray,
        left_shadow: np.ndarray,
        right_shadow: np.ndarray,
        elevation: np.ndarray,
        lateral: np.ndarray,
    ) -> tuple:
        length = len(delayed_left)
        if length == 0:
            return delayed_left, delayed_right
        target_left = self._parameters(
            float(np.mean(left_shadow)),
            float(np.mean(elevation)),
            float(np.mean(lateral)),
        )
        target_right = self._parameters(
            float(np.mean(right_shadow)),
            float(np.mean(elevation)),
            float(np.mean(lateral)),
        )
        if not self._initialized:
            self.previous_left = target_left
            self.previous_right = target_right
            self._initialized = True
        coefficients_left = self._block_parameters(
            target_left, self.previous_left, length
        )
        coefficients_right = self._block_parameters(
            target_right, self.previous_right, length
        )
        filtered_left = self._process_ear(
            delayed_left, coefficients_left, self.state_left
        )
        filtered_right = self._process_ear(
            delayed_right, coefficients_right, self.state_right
        )
        self.previous_left = target_left
        self.previous_right = target_right
        return filtered_left, filtered_right


class BinauralRenderer:
    """Convert a mono source stream into stereo with ITD, ILD, and HRTF."""

    def __init__(self, sample_rate: int, head_radius: float = 0.0875) -> None:
        self.sample_rate = int(sample_rate)
        self.max_itd_seconds = 2.0 * head_radius / 343.0
        max_delay = self.max_itd_seconds * self.sample_rate + 4.0
        self.delay_line = FractionalDelayLine(max_delay)
        self.hrtf = MinimumPhaseHRTF(sample_rate)

    def process(self, mono: np.ndarray, cues: CueBlock) -> np.ndarray:
        mono = np.asarray(mono, dtype=np.float64)
        length = len(mono)
        if length == 0:
            return np.zeros((0, 2), dtype=np.float64)
        itd_samples = cues.itd_seconds * float(self.sample_rate)
        left_delay = np.maximum(0.0, itd_samples)
        right_delay = np.maximum(0.0, -itd_samples)
        delayed_left, delayed_right = self.delay_line.process(
            mono, left_delay, right_delay
        )
        delayed_left *= cues.left_ild
        delayed_right *= cues.right_ild
        filtered_left, filtered_right = self.hrtf.process(
            delayed_left,
            delayed_right,
            cues.left_shadow,
            cues.right_shadow,
            cues.elevation,
            cues.lateral,
        )
        return np.column_stack((filtered_left, filtered_right))
