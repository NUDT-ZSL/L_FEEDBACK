"""Small NumPy-only 3D math helpers.

Matrices use column vectors and the OpenGL clip convention:
clip = projection @ view @ world, with NDC z in [-1, 1].
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


FloatArray = np.ndarray


def normalize(vector: Sequence[float]) -> FloatArray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length <= 1e-15:
        raise ValueError("cannot normalize a zero-length vector")
    return value / length


def look_at(eye: Iterable[float], target: Iterable[float], up: Iterable[float]) -> FloatArray:
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    forward = normalize(target_v - eye_v)
    right = normalize(np.cross(forward, up_v))
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye_v))
    view[1, 3] = -float(np.dot(true_up, eye_v))
    view[2, 3] = float(np.dot(forward, eye_v))
    return view


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> FloatArray:
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2.0 * far * near) / (near - far)
    projection[3, 2] = -1.0
    return projection


def orthographic(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> FloatArray:
    projection = np.eye(4, dtype=np.float64)
    projection[0, 0] = 2.0 / (right - left)
    projection[1, 1] = 2.0 / (top - bottom)
    projection[2, 2] = -2.0 / (far - near)
    projection[0, 3] = -(right + left) / (right - left)
    projection[1, 3] = -(top + bottom) / (top - bottom)
    projection[2, 3] = -(far + near) / (far - near)
    return projection


def transform_points(matrix: FloatArray, points: FloatArray) -> FloatArray:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate(
        (flat, np.ones((flat.shape[0], 1), dtype=np.float64)), axis=1
    )
    transformed = homogeneous @ matrix.T
    return transformed.reshape(*points.shape[:-1], 4)


def orbit_camera_parameters(
    target: Sequence[float],
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
) -> tuple[FloatArray, FloatArray]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    offset = np.array(
        [
            math.cos(el) * math.cos(az),
            math.sin(el),
            math.cos(el) * math.sin(az),
        ],
        dtype=np.float64,
    )
    target_v = np.asarray(target, dtype=np.float64)
    eye = target_v + offset * distance
    return eye, target_v


def frustum_planes(view_projection: FloatArray) -> FloatArray:
    """Return six world-space planes with inside defined as dot(plane, p) >= 0."""

    m = np.asarray(view_projection, dtype=np.float64)
    planes = np.stack(
        (
            m[3] + m[0],
            m[3] - m[0],
            m[3] + m[1],
            m[3] - m[1],
            m[3] + m[2],
            m[3] - m[2],
        )
    )
    lengths = np.linalg.norm(planes[:, :3], axis=1)
    lengths = np.maximum(lengths, 1e-15)
    return planes / lengths[:, None]


def aabb_corners(minimum: Sequence[float], maximum: Sequence[float]) -> FloatArray:
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    return np.array(
        [
            [low[0], low[1], low[2]],
            [high[0], low[1], low[2]],
            [low[0], high[1], low[2]],
            [high[0], high[1], low[2]],
            [low[0], low[1], high[2]],
            [high[0], low[1], high[2]],
            [low[0], high[1], high[2]],
            [high[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )


def aabb_visible(
    minimum: Sequence[float], maximum: Sequence[float], planes: FloatArray
) -> bool:
    corners = aabb_corners(minimum, maximum)
    signed = corners @ planes[:, :3].T + planes[:, 3]
    return bool(np.all(np.any(signed >= 0.0, axis=0)))


def project_to_screen(
    corners_world: FloatArray,
    view_projection: FloatArray,
    width: int,
    height: int,
    backface_fullscreen: bool = True,
) -> tuple[int, int, int, int]:
    clip = transform_points(view_projection, corners_world)
    w = clip[..., 3]
    if np.any(w <= 1e-9):
        if backface_fullscreen:
            return 0, 0, width - 1, height - 1
        valid = w > 1e-9
        if not np.any(valid):
            return 0, 0, -1, -1
        clip = clip[valid]
        w = w[valid]

    ndc = clip[..., :3] / w[..., None]
    x = (ndc[..., 0] * 0.5 + 0.5) * (width - 1)
    y = (0.5 - ndc[..., 1] * 0.5) * (height - 1)
    x0 = int(np.clip(np.floor(np.min(x)), 0, width - 1))
    x1 = int(np.clip(np.ceil(np.max(x)), 0, width - 1))
    y0 = int(np.clip(np.floor(np.min(y)), 0, height - 1))
    y1 = int(np.clip(np.ceil(np.max(y)), 0, height - 1))
    return x0, y0, x1, y1


def direction_from_angles(azimuth_deg: float, elevation_deg: float) -> FloatArray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(el) * math.cos(az),
            math.sin(el),
            math.cos(el) * math.sin(az),
        ],
        dtype=np.float64,
    )

