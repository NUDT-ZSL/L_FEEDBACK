"""Stateful DSP primitives: one-pole lowpass, delay line, resampler.

All state is plain float64 scalars/arrays so it can be serialized into
snapshots with exact round-trip precision.
"""

import math

import numpy as np


def cutoff_to_coef(cutoff_hz, sample_rate):
    """One-pole smoothing coefficient for a -3dB point at cutoff_hz."""
    cutoff_hz = min(max(cutoff_hz, 1.0), 0.49 * sample_rate)
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)


class OnePoleLowpass:
    """One-pole lowpass with per-sample coefficient ramp (de-zippered)."""

    def __init__(self):
        self.y1 = 0.0
        self.coef = None  # smoothed coefficient carried across frames

    def process(self, x, target_coef):
        n = len(x)
        out = np.empty(n, dtype=np.float64)
        c0 = self.coef if self.coef is not None else target_coef
        y1 = self.y1
        step = (target_coef - c0) / n
        coef = c0
        for i in range(n):
            coef += step
            y1 = y1 + coef * (x[i] - y1)
            out[i] = y1
        self.y1 = y1
        self.coef = target_coef
        return out

    def state_dict(self):
        return {"y1": self.y1, "coef": self.coef}

    def load_state(self, state):
        self.y1 = float(state["y1"])
        self.coef = None if state["coef"] is None else float(state["coef"])


class DelayLine:
    """Integer-sample delay line for early reflections."""

    def __init__(self, delay_samples, gain):
        self.delay = int(delay_samples)
        self.gain = float(gain)
        self.buf = np.zeros(self.delay, dtype=np.float64)
        self.pos = 0

    def process(self, x):
        n = len(x)
        idx = (self.pos + np.arange(n)) % self.delay
        out = self.buf[idx].copy()
        self.buf[idx] = x
        self.pos = int((self.pos + n) % self.delay)
        return out * self.gain

    def state_dict(self):
        return {
            "delay": self.delay,
            "gain": self.gain,
            "buf": self.buf.tolist(),
            "pos": self.pos,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(int(state["delay"]), float(state["gain"]))
        obj.buf = np.array(state["buf"], dtype=np.float64)
        obj.pos = int(state["pos"])
        return obj


class Resampler:
    """Variable-rate linear-interpolation resampler (doppler shift).

    The rate factor is ramped linearly across each frame so pitch stays
    continuous at trajectory keyframes and velocity discontinuities.
    """

    def __init__(self):
        self.phase = 0.0        # absolute input position of next output sample
        self.factor_prev = 1.0  # smoothed factor at the previous frame boundary

    def process(self, dry_source, n_out, factor_end):
        factors = np.linspace(self.factor_prev, factor_end, n_out)
        offsets = np.cumsum(factors) - factors[0]
        positions = self.phase + offsets
        out = dry_source.read(positions)
        self.phase = self.phase + float(offsets[-1]) + float(factors[-1])
        self.factor_prev = factor_end
        return out

    def state_dict(self):
        return {"phase": self.phase, "factor_prev": self.factor_prev}

    def load_state(self, state):
        self.phase = float(state["phase"])
        self.factor_prev = float(state["factor_prev"])
