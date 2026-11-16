"""Zero-lookahead stereo-linked dynamic range compressor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ParameterError


@dataclass(frozen=True)
class CompressorConfig:
    threshold_db: float = -18.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 120.0
    knee_db: float = 6.0
    makeup_db: float = 0.0

    def validate(self) -> None:
        values = (
            self.threshold_db,
            self.ratio,
            self.attack_ms,
            self.release_ms,
            self.knee_db,
            self.makeup_db,
        )
        if any(not math.isfinite(value) for value in values):
            raise ParameterError("compressor parameters must be finite")
        if not -120.0 <= self.threshold_db <= 24.0:
            raise ParameterError("compressor threshold must be in [-120, 24] dB")
        if self.ratio < 1.0:
            raise ParameterError("compressor ratio must be >= 1")
        if self.attack_ms <= 0.0 or self.release_ms <= 0.0:
            raise ParameterError("compressor attack/release must be positive")
        if self.knee_db < 0.0:
            raise ParameterError("compressor knee must be >= 0 dB")


class Compressor:
    """Peak-linked compressor with soft knee and smooth dB gain envelope."""

    def __init__(
        self, sample_rate: int, config: CompressorConfig
    ) -> None:
        config.validate()
        self.sample_rate = int(sample_rate)
        self.config = config
        self.attack_coefficient = math.exp(
            -1.0 / (self.sample_rate * config.attack_ms / 1000.0)
        )
        self.release_coefficient = math.exp(
            -1.0 / (self.sample_rate * config.release_ms / 1000.0)
        )
        self.gain_reduction_db = 0.0
        self.gain_envelope_db: np.ndarray = np.zeros(0, dtype=np.float64)

    def _desired_reduction(self, input_db: float) -> float:
        over = input_db - self.config.threshold_db
        slope = 1.0 - 1.0 / self.config.ratio
        knee = self.config.knee_db
        if knee <= 0.0:
            return max(0.0, over) * slope
        if over <= -0.5 * knee:
            return 0.0
        if over >= 0.5 * knee:
            return over * slope
        return slope * ((over + 0.5 * knee) ** 2) / (2.0 * knee)

    def process(self, stereo: np.ndarray) -> np.ndarray:
        stereo = np.asarray(stereo, dtype=np.float64)
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("compressor input must be (samples, 2)")
        output = np.empty_like(stereo)
        reduction = self.gain_reduction_db
        envelope = np.empty(len(stereo), dtype=np.float64)
        for index, frame in enumerate(stereo):
            peak = max(abs(float(frame[0])), abs(float(frame[1])))
            input_db = 20.0 * math.log10(max(peak, 1.0e-9))
            target = self._desired_reduction(input_db)
            if target > reduction:
                coefficient = self.attack_coefficient
            else:
                coefficient = self.release_coefficient
            reduction = target + coefficient * (reduction - target)
            envelope[index] = reduction
            gain = (
                10.0 ** ((self.config.makeup_db - reduction) / 20.0)
            )
            output[index] = frame * gain
        self.gain_reduction_db = reduction
        self.gain_envelope_db = envelope
        return output
