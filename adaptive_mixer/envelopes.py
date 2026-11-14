"""Crossfade splines and gain envelopes.

Three crossfade types are provided on the normalized domain ``t in [0, 1]``.
Every outgoing curve ``g_out(t)`` satisfies ``g_out(0)=1, g_out(1)=0`` with a
zero first derivative at both endpoints.  The incoming curve is defined per
crossfade family (``1 - g_out`` for linear/S-curves, ``sin`` partner for the
equal-power pair) so that each family preserves its intended energy law while
remaining C0- and C1-continuous at every splice point.  The smoothstep-based
spline segments are cubic/quintic Hermite pieces with matching endpoint
values and tangents, which is the monotone cubic spline requirement.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

CROSSFADE_TYPES = ("linear", "equal_power", "s_curve")
_EDGE = 0.15


def _s_curve() -> "callable":
    # Quintic smootherstep: zero first and second derivative at both ends.
    def evaluate(t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        return 1.0 - (6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3)

    return evaluate


def _linear_curve() -> "callable":
    # Cubic-spline eased linear ramp: identity in the middle, rounded ends.
    edge = _EDGE

    def evaluate(t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        out = np.empty_like(t)
        left = t < edge
        right = t > 1.0 - edge
        middle = ~(left | right)
        u_left = t[left] / edge
        out[left] = 1.0 - edge * (3.0 * u_left ** 2
                                  - 2.0 * u_left ** 3)
        out[middle] = 1.0 - t[middle]
        u_right = (t[right] - (1.0 - edge)) / edge
        out[right] = edge * (1.0 - (3.0 * u_right ** 2
                                    - 2.0 * u_right ** 3))
        return out

    return evaluate


def _equal_power_curve() -> "callable":
    # Exact constant-power pair with eased angular parameter so that endpoint
    # derivatives are zero: g_out = cos(phi), g_in = sin(phi), with phi
    # running 0 -> pi/2 through a smoothstep mapping.
    def evaluate(t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        return np.cos(0.5 * np.pi * _equal_power_phi(t))

    return evaluate


def _equal_power_in(phi_fn) -> "callable":
    def evaluate(t: np.ndarray) -> np.ndarray:
        return np.sin(0.5 * np.pi * phi_fn(np.clip(
            np.asarray(t, dtype=np.float64), 0.0, 1.0)))

    return evaluate


def _equal_power_phi(t: np.ndarray) -> np.ndarray:
    edge = _EDGE
    phi = np.empty_like(t)
    left = t < edge
    right = t > 1.0 - edge
    middle = ~(left | right)
    u_left = t[left] / edge
    phi[left] = edge * (3.0 * u_left ** 2 - 2.0 * u_left ** 3)
    phi[middle] = t[middle]
    u_right = (t[right] - (1.0 - edge)) / edge
    phi[right] = (1.0 - edge) + edge * (
        3.0 * u_right ** 2 - 2.0 * u_right ** 3)
    return phi


_CURVE_FACTORIES = {
    "linear": _linear_curve,
    "equal_power": _equal_power_curve,
    "s_curve": _s_curve,
}


def crossfade_gains(length: int, curve_type: str) -> Dict[str, np.ndarray]:
    """Return ``{"out": g_out, "in": g_in}`` arrays of given sample length."""
    if curve_type not in _CURVE_FACTORIES:
        raise ValueError(f"unknown crossfade type: {curve_type!r}")
    if length <= 1:
        out = np.ones(max(length, 1), dtype=np.float64)
        return {"out": out, "in": 1.0 - out}
    t = np.arange(length, dtype=np.float64) / (length - 1)
    g_out = _CURVE_FACTORIES[curve_type]()(t)
    g_out = np.clip(g_out, 0.0, 1.0)
    if curve_type == "equal_power":
        g_in = _equal_power_in(_equal_power_phi)(t)
    else:
        g_in = 1.0 - g_out
    return {"out": g_out, "in": np.clip(g_in, 0.0, 1.0)}


def apply_fade_in_out(samples: np.ndarray, fade_samples: int) -> np.ndarray:
    """Symmetric short fade envelope that prevents start/end transients."""
    n = len(samples)
    fade = max(0, min(fade_samples, n // 2 if n else 0))
    envelope = np.ones(max(n, 0), dtype=np.float64)
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    return samples * envelope


def constant_with_fades(length: int, gain: float,
                        fade_samples: int) -> np.ndarray:
    """Constant gain segment with raised-cosine edge fades."""
    envelope = apply_fade_in_out(np.ones(length), fade_samples)
    return envelope * gain
