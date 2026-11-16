"""Small vector, quaternion, and AABB geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

EPSILON = 1.0e-12


def normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / np.maximum(norm, EPSILON)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def rotate_world_to_head(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate world vector into head coordinates using inverse(q)."""
    inverse = quat_conjugate(quat)
    pure = np.concatenate(
        (np.zeros(vector.shape[:-1] + (1,)), vector), axis=-1
    )
    rotated = quat_multiply(quat_multiply(inverse, pure), quat)
    return rotated[..., 1:]


def quat_yaw(angle_rad: float) -> np.ndarray:
    """Quaternion rotating +Z toward +X around +Y."""
    return np.array(
        [math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class SpatialCue:
    azimuth: float
    elevation: float
    lateral: float
    itd_seconds: float
    left_ild: float
    right_ild: float
    left_shadow: float
    right_shadow: float


def spatial_cue(
    relative_world: np.ndarray,
    head_quaternion: np.ndarray,
    head_radius: float = 0.0875,
    speed_of_sound: float = 343.0,
) -> SpatialCue:
    """Compute direction, ITD, ILD, and head-shadow amounts for one sample.

    Head-local axes are +X right, +Y up, +Z forward.  Positive lateral values
    therefore mean a source on the right, whose right ear receives the earlier
    wavefront.
    """
    norm = float(np.linalg.norm(relative_world))
    local = rotate_world_to_head(relative_world, head_quaternion)
    if norm < EPSILON:
        return SpatialCue(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)

    lateral = float(np.clip(local[0] / norm, -1.0, 1.0))
    azimuth = math.atan2(float(local[0]), float(local[2]))
    horizontal = math.hypot(float(local[0]), float(local[2]))
    elevation = math.atan2(float(local[1]), max(horizontal, EPSILON))
    max_itd = 2.0 * head_radius / speed_of_sound
    itd = max_itd * lateral
    left_shadow = max(0.0, lateral)
    right_shadow = max(0.0, -lateral)
    left_ild = 1.0 - 0.45 * left_shadow
    right_ild = 1.0 - 0.45 * right_shadow
    return SpatialCue(
        azimuth=azimuth,
        elevation=elevation,
        lateral=lateral,
        itd_seconds=itd,
        left_ild=left_ild,
        right_ild=right_ild,
        left_shadow=left_shadow,
        right_shadow=right_shadow,
    )


def vectorized_spatial_cues(
    relative_world: np.ndarray,
    quaternions: np.ndarray,
    head_radius: float = 0.0875,
    speed_of_sound: float = 343.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized direction and level cues.

    Returns ``(lateral, itd_seconds, left_ild, right_ild, elevation)``.
    """
    local = rotate_world_to_head(relative_world, quaternions)
    norms = np.maximum(np.linalg.norm(relative_world, axis=1), EPSILON)
    lateral = np.clip(local[:, 0] / norms, -1.0, 1.0)
    horizontal = np.hypot(local[:, 0], local[:, 2])
    elevation = np.arctan2(local[:, 1], np.maximum(horizontal, EPSILON))
    max_itd = 2.0 * head_radius / speed_of_sound
    itd = max_itd * lateral
    left_shadow = np.maximum(0.0, lateral)
    right_shadow = np.maximum(0.0, -lateral)
    left_ild = 1.0 - 0.45 * left_shadow
    right_ild = 1.0 - 0.45 * right_shadow
    return lateral, itd, left_ild, right_ild, elevation


def segment_intersects_aabb(
    start: np.ndarray,
    end: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < EPSILON:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        inv = 1.0 / float(direction[axis])
        t1 = (float(minimum[axis]) - float(start[axis])) * inv
        t2 = (float(maximum[axis]) - float(start[axis])) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min <= 1.0


def segment_intersects_aabb_array(
    starts: np.ndarray,
    ends: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask for many listener-to-source segments."""
    directions = ends - starts
    t_min = np.zeros(len(starts), dtype=np.float64)
    t_max = np.ones(len(starts), dtype=np.float64)
    possible = np.ones(len(starts), dtype=bool)
    for axis in range(3):
        delta = directions[:, axis]
        parallel = np.abs(delta) < EPSILON
        outside = (starts[:, axis] < minimum[axis]) | (
            starts[:, axis] > maximum[axis]
        )
        possible &= ~(parallel & outside)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / delta
            t1 = (minimum[axis] - starts[:, axis]) * inv
            t2 = (maximum[axis] - starts[:, axis]) * inv
        low = np.minimum(t1, t2)
        high = np.maximum(t1, t2)
        t_min = np.where(parallel, t_min, np.maximum(t_min, low))
        t_max = np.where(parallel, t_max, np.minimum(t_max, high))
        possible &= ~(parallel & outside)
    return possible & (t_max >= 0.0) & (t_min <= 1.0)
