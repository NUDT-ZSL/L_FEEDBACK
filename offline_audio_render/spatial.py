"""Listener-relative spatialisation: equal-power stereo and 5.1 panning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAYOUT_MONO = "mono"
LAYOUT_STEREO = "stereo"
LAYOUT_51 = "5.1"

LAYOUT_CHANNELS = {LAYOUT_MONO: 1, LAYOUT_STEREO: 2, LAYOUT_51: 6}

# Standard 5.1 channel order used by WAVE_FORMAT_EXTENSIBLE:
# L, R, C, LFE, Ls, Rs.
CHANNEL_51 = ("L", "R", "C", "LFE", "Ls", "Rs")

# Speaker azimuths in radians, measured clockwise from forward (0),
# negative = listener's left, positive = listener's right.
SPEAKER_AZIMUTH_51 = {
    "L": np.deg2rad(-30.0),
    "R": np.deg2rad(30.0),
    "C": 0.0,
    "Ls": np.deg2rad(-110.0),
    "Rs": np.deg2rad(110.0),
}


@dataclass(frozen=True)
class ListenerPose:
    position: np.ndarray  # shape (3,)
    yaw: float  # radians, 0 looks toward +X, positive rotates toward +Y


def _listener_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return forward, left


def relative_azimuth_distance(
    source_position: np.ndarray, listener: ListenerPose
) -> tuple[float, float]:
    """Return source azimuth (radians, +right) and Euclidean distance."""
    delta = np.asarray(source_position, dtype=np.float64) - listener.position
    forward, left = _listener_axes(listener.yaw)
    along = float(np.dot(delta, forward))
    right_component = float(np.dot(delta, -left))
    # Positive azimuth means the source is to the listener's right.
    azimuth = float(np.arctan2(right_component, along))
    distance = float(np.linalg.norm(delta))
    return azimuth, distance


def distance_gain(distance: float, min_distance: float = 1.0) -> float:
    """Inverse-square distance attenuation with a minimum-distance clamp."""
    effective = max(float(distance), float(min_distance))
    return float((min_distance / effective) ** 2)


def stereo_gains(azimuth: float) -> np.ndarray:
    """Equal-power stereo pair. Azimuth +/- pi/2 maps hard left/right."""
    # Fold rear sources to the front hemisphere.
    angle = _wrap_front(azimuth)
    x = np.clip(angle / (np.pi / 2.0), -1.0, 1.0)
    theta = (x + 1.0) * (np.pi / 4.0)  # 0..pi/2
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)


def five_one_gains(azimuth: float) -> np.ndarray:
    """Constant-power 5.1 gains (L, R, C, LFE, Ls, Rs); LFE is always silent."""
    angle = _wrap_signed_pi(azimuth)
    speakers = ("L", "C", "R", "Rs", "Ls")
    azimuths = np.array(
        [SPEAKER_AZIMUTH_51[name] for name in speakers], dtype=np.float64
    )
    # Sort around the circle so the source falls between two neighbours.
    order = np.argsort(azimuths)
    names = [speakers[i] for i in order]
    az_sorted = azimuths[order]
    wrapped = np.concatenate(
        (az_sorted, az_sorted[:1] + 2.0 * np.pi)
    )
    target = angle if angle >= az_sorted[0] else angle + 2.0 * np.pi
    idx = int(np.searchsorted(wrapped, target, side="right") - 1)
    idx = max(0, min(idx, len(names) - 1))
    a0 = wrapped[idx]
    a1 = wrapped[idx + 1]
    span = a1 - a0
    fraction = 0.0 if span == 0.0 else (target - a0) / span
    left_w = np.cos(fraction * np.pi / 2.0)
    right_w = np.sin(fraction * np.pi / 2.0)

    gains = {name: 0.0 for name in CHANNEL_51}
    gains[names[idx]] = float(left_w)
    gains[names[(idx + 1) % len(names)]] = float(right_w)

    out = np.array(
        [gains["L"], gains["R"], gains["C"], 0.0, gains["Ls"], gains["Rs"]],
        dtype=np.float64,
    )
    # Renormalise to unit power for safety.
    power = float(np.sqrt(np.sum(out * out)))
    if power > 0.0:
        out /= power
    return out


def _wrap_front(azimuth: float) -> float:
    """Fold any azimuth into the front hemisphere [-pi/2, pi/2]."""
    angle = _wrap_signed_pi(azimuth)
    if angle > np.pi / 2.0:
        angle = np.pi - angle
    elif angle < -np.pi / 2.0:
        angle = -np.pi - angle
    return float(np.clip(angle, -np.pi / 2.0, np.pi / 2.0))


def _wrap_signed_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def pan_gains(azimuth: float, layout: str) -> np.ndarray:
    if layout == LAYOUT_MONO:
        return np.ones(1, dtype=np.float64)
    if layout == LAYOUT_STEREO:
        return stereo_gains(azimuth)
    if layout == LAYOUT_51:
        return five_one_gains(azimuth)
    raise ValueError(f"unsupported layout {layout!r}")
