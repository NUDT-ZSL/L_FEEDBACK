"""Peak-detected soft-knee dynamics compressor.

The compressor is feed-forward:

1. A per-sample peak detector drives an envelope follower with independent
   one-pole attack/release time constants (``exp(-1/(rate*time))``).
2. Static gain reduction follows the classic soft-knee curve (below the knee
   no reduction, inside the knee a quadratic blend, above the knee the
   configured ratio).
3. Optional look-ahead delays the signal while the gain is applied ahead of
   time; look-ahead is capped at 5 ms by project validation so introduced
   latency never exceeds the requirement.
"""

from __future__ import annotations

import numpy as np

from .project import CompressorSpec

EPS = 1.0e-12


def db_to_linear(db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def linear_to_db(level: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(level, EPS))


def _time_constant_samples(milliseconds: float, sample_rate: int) -> float:
    return max(sample_rate * milliseconds / 1000.0, 1.0)


def _soft_knee_gain_db(level_db: np.ndarray, spec: CompressorSpec) -> np.ndarray:
    """Static compressor gain (dB, <= 0) for a detector level in dB."""
    threshold = spec.threshold_db
    knee = spec.knee_db
    ratio = spec.ratio
    lower = threshold - knee / 2.0
    upper = threshold + knee / 2.0

    gain = np.zeros_like(level_db)

    # Above the knee: standard hard-ratio compression.
    above = level_db > upper
    gain[above] = -(1.0 - 1.0 / ratio) * (level_db[above] - threshold)

    # Inside the knee: quadratic 0 -> full-ratio blend.
    inside = (level_db >= lower) & (~above)
    if knee > 0.0:
        delta = level_db[inside] - lower
        gain[inside] = -((1.0 - 1.0 / ratio) * delta**2) / (2.0 * knee)
    return gain


class SoftKneeCompressor:
    """Stateful per-track compressor (state persists across events)."""

    def __init__(self, spec: CompressorSpec, sample_rate: int):
        self.spec = spec
        self.sample_rate = sample_rate
        self.lookahead_samples = int(round(sample_rate * spec.lookahead_ms / 1000.0))
        self.envelope_db = 0.0
        self.pending_gain = 1.0

    def reset(self) -> None:
        self.envelope_db = 0.0
        self.pending_gain = 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] == 0:
            return audio.copy()

        spec = self.spec
        attack_coef = float(np.exp(-1.0 / _time_constant_samples(spec.attack_ms, self.sample_rate)))
        release_coef = float(np.exp(-1.0 / _time_constant_samples(spec.release_ms, self.sample_rate)))

        # Stereo-linked peak detector on the instantaneous absolute level.
        peak = np.max(np.abs(audio), axis=1)
        target_gain_db = _soft_knee_gain_db(linear_to_db(peak), spec)

        envelope = np.empty_like(target_gain_db)
        state = self.envelope_db
        for i, target in enumerate(target_gain_db):
            # More negative gain means deeper reduction -> attack phase.
            coefficient = attack_coef if target < state - EPS else release_coef
            state = coefficient * state + (1.0 - coefficient) * target
            envelope[i] = state
        self.envelope_db = float(state)

        gain_linear = db_to_linear(envelope)
        delay = self.lookahead_samples
        if delay > 0:
            # Delay the signal while the gain schedule moves earlier, so
            # transients are controlled before they reach the output.
            delayed = np.zeros_like(audio)
            aligned_gain = np.empty_like(gain_linear)
            if delay < audio.shape[0]:
                delayed[delay:] = audio[:-delay]
                aligned_gain[:-delay] = gain_linear[delay:]
                aligned_gain[-delay:] = self.pending_gain
            else:
                aligned_gain[:] = self.pending_gain
            self.pending_gain = float(gain_linear[-1])
            return delayed * aligned_gain[:, np.newaxis]
        return audio * gain_linear[:, np.newaxis]


def make_compressor(spec: CompressorSpec, sample_rate: int) -> SoftKneeCompressor:
    return SoftKneeCompressor(spec, sample_rate)


def compressor_latency_ms(spec: CompressorSpec, sample_rate: int) -> float:
    """Actual introduced latency in milliseconds (look-ahead only)."""
    return 1000.0 * int(round(sample_rate * spec.lookahead_ms / 1000.0)) / sample_rate
