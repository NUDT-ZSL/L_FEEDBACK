"""Pan laws and fade curves shared by the mixer and the report writer."""

from __future__ import annotations

import math

import numpy as np


def equal_power_pan(pan: float) -> tuple[float, float]:
    """Constant-power sine/cosine pan law.

    ``pan`` in [-1, 1]: -1 hard left, 0 centre (-3 dB each), +1 hard right.
    Gains satisfy ``L^2 + R^2 == 1`` so total power is constant.
    """
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def linear_gain_curve(length: int, fade_in: bool) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, length, endpoint=True, dtype=np.float64)
    return ramp if fade_in else ramp[::-1]


def equal_power_gain_curve(length: int, fade_in: bool) -> np.ndarray:
    """Equal-power fade: gain follows cos/sin over a quarter circle."""
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    ramp = np.linspace(0.0, math.pi / 2.0, length, endpoint=True, dtype=np.float64)
    gains = np.sin(ramp) if fade_in else np.cos(ramp)
    return gains


def fade_curve(length: int, curve: str, fade_in: bool) -> np.ndarray:
    if curve == "linear":
        return linear_gain_curve(length, fade_in)
    if curve == "equal_power":
        return equal_power_gain_curve(length, fade_in)
    raise ValueError(f"unsupported fade curve {curve!r}")


def db_to_gain(gain_db: float) -> float:
    return 10.0 ** (gain_db / 20.0)
