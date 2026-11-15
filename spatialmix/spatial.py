"""Two-dimensional equal-power spatial gain calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_MIN_GAIN = 0.05
REFERENCE_DISTANCE = 1.0


@dataclass(frozen=True)
class SpatialGains:
    """Spatial parameters and final per-channel gains for one event."""

    distance: float
    azimuth_degrees: float
    distance_gain: float
    left_gain: float
    right_gain: float


def compute_distance_gain(
    distance: float,
    min_gain: float = DEFAULT_MIN_GAIN,
    reference_distance: float = REFERENCE_DISTANCE,
) -> float:
    """Return an inverse-distance gain with an upper and lower bound.

    ``gain = clamp(reference_distance / max(distance, reference_distance),
    min_gain, 1.0)``.  A zero distance therefore has gain 1.0 instead of
    producing an infinite value.
    """

    distance = float(distance)
    min_gain = float(min_gain)
    reference_distance = float(reference_distance)

    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be a finite non-negative number")
    if not math.isfinite(min_gain) or not 0.0 <= min_gain <= 1.0:
        raise ValueError("min_gain must be a finite number in [0, 1]")
    if not math.isfinite(reference_distance) or reference_distance <= 0.0:
        raise ValueError("reference_distance must be a finite positive number")

    inverse_gain = reference_distance / max(distance, reference_distance)
    return min(1.0, max(min_gain, inverse_gain))


def equal_power_pan(pan: float) -> tuple[float, float]:
    """Map a -1 (left) to +1 (right) pan value to equal-power gains."""

    pan = float(pan)
    if not math.isfinite(pan):
        raise ValueError("pan must be finite")
    pan = min(1.0, max(-1.0, pan))
    theta = (pan + 1.0) * math.pi / 4.0
    return math.cos(theta), math.sin(theta)


def compute_spatial_gains(
    listener_x: float,
    listener_y: float,
    source_x: float,
    source_y: float,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> SpatialGains:
    """Compute distance attenuation and equal-power stereo panning.

    Positive Y points forward and positive X points right.  Azimuth is
    ``atan2(dx, dy)`` in degrees.  A source exactly behind the listener has an
    azimuth of +/-180 degrees and is panned to the center, which avoids a
    left/right discontinuity at the rear boundary.
    """

    listener_x = float(listener_x)
    listener_y = float(listener_y)
    source_x = float(source_x)
    source_y = float(source_y)

    values = (listener_x, listener_y, source_x, source_y)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("positions must contain finite numbers")

    dx = source_x - listener_x
    dy = source_y - listener_y
    distance = math.hypot(dx, dy)

    if distance == 0.0:
        azimuth = 0.0
        pan = 0.0
    else:
        azimuth = math.atan2(dx, dy)
        pan = math.sin(azimuth)

    distance_gain = compute_distance_gain(distance, min_gain=min_gain)
    pan_left, pan_right = equal_power_pan(pan)
    return SpatialGains(
        distance=distance,
        azimuth_degrees=math.degrees(azimuth),
        distance_gain=distance_gain,
        left_gain=distance_gain * pan_left,
        right_gain=distance_gain * pan_right,
    )
