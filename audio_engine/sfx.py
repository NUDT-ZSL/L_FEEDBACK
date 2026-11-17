"""Skill SFX scheduling, stereo panning and soft limiting."""

from __future__ import annotations

import math

import numpy as np


def constant_power_pan(pan: float) -> tuple[float, float]:
    """Constant-power pan law; pan in [-1, 1] -> (left, right) gains."""
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def schedule_sfx(
    bus: np.ndarray,
    sfx: np.ndarray,
    start_sample: int,
    gain: float,
    pan: float,
) -> None:
    """Additively mix a (mono or stereo) SFX into the stereo bus.

    Overlapping sounds always sum; nothing is overwritten.
    """
    if bus.ndim != 2 or bus.shape[1] != 2:
        raise ValueError("bus must be shaped (n_samples, 2)")
    n_bus = bus.shape[0]
    src0 = max(0, -start_sample)
    dst0 = max(0, start_sample)
    count = min(sfx.shape[0] - src0, n_bus - dst0)
    if count <= 0:
        return
    left, right = constant_power_pan(pan)
    segment = sfx[src0 : src0 + count]
    if segment.shape[1] == 1:
        bus[dst0 : dst0 + count, 0] += segment[:, 0] * gain * left
        bus[dst0 : dst0 + count, 1] += segment[:, 0] * gain * right
    else:
        bus[dst0 : dst0 + count, 0] += segment[:, 0] * gain * left
        bus[dst0 : dst0 + count, 1] += segment[:, 1] * gain * right


def soft_limit(mix: np.ndarray, ceiling: float) -> np.ndarray:
    """Soft-limit with a tanh curve only when the peak exceeds ``ceiling``.

    Never hard-clips: output magnitude stays strictly below ``ceiling``.
    """
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > ceiling:
        return ceiling * np.tanh(mix / ceiling)
    return mix
