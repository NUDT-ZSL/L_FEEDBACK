"""Piecewise-linear parameter curves and time-varying one-pole low-pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    """Piecewise-linear curve defined over normalised time ``0..1``.

    ``points`` holds ``(t, value)`` pairs with monotonically increasing ``t``.
    Values outside the range clamp to the endpoints.
    """

    points: tuple[tuple[float, float], ...]

    @classmethod
    def constant(cls, value: float) -> "Curve":
        return cls(((0.0, float(value)), (1.0, float(value))))

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        if len(self.points) == 2 and self.points[0][1] == self.points[1][1]:
            return np.full_like(t, self.points[0][1], dtype=np.float64)
        ts = np.asarray([p[0] for p in self.points], dtype=np.float64)
        vs = np.asarray([p[1] for p in self.points], dtype=np.float64)
        return np.interp(t, ts, vs, left=vs[0], right=vs[-1])


def lowpass_alpha(cutoff_hz: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per-sample one-pole low-pass coefficient ``alpha``.

    ``y[n] = alpha * x[n] + (1 - alpha) * y[n-1]`` where
    ``alpha = 1 - exp(-2*pi*fc/fs)``. Cut-off is clamped to the stable range
    so coefficient updates never blow up.
    """
    nyquist = sample_rate * 0.5
    fc = np.clip(np.asarray(cutoff_hz, dtype=np.float64), 1.0, nyquist * 0.999)
    return 1.0 - np.exp(-2.0 * np.pi * fc / sample_rate)


class OnePoleLowPass:
    """Sample-accurate time-varying one-pole filter with persistent state.

    The recurrence ``y[n] = alpha[n]*x[n] + (1-alpha[n])*y[n-1]`` is evaluated
    in fixed-size internal chunks via closed-form cumulative products so that
    results are independent of the caller's block size while the state stays
    continuous across calls.
    """

    CHUNK = 64

    def __init__(self) -> None:
        self._state = 0.0

    def reset(self) -> None:
        self._state = 0.0

    def process(self, samples: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Process one block; ``alpha`` is one coefficient per sample."""
        alpha = np.asarray(alpha, dtype=np.float64)
        out = np.empty_like(samples, dtype=np.float64)
        state = self._state
        # Fixed 64-frame sub-chunks preserve block-size independence. The
        # chunk is short enough that cumulative pole products never underflow,
        # while vectorisation keeps long renders fast.
        for start in range(0, samples.shape[0], self.CHUNK):
            end = min(start + self.CHUNK, samples.shape[0])
            a = alpha[start:end]
            pole = 1.0 - a
            cum_pole = np.cumprod(pole)
            weighted = a * samples[start:end] / cum_pole
            out[start:end] = cum_pole * (state + np.cumsum(weighted))
            state = float(out[end - 1])
        self._state = float(state)
        return out


def db_to_gain(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)
