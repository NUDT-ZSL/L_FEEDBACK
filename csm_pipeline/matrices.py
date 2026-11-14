"""Float64 matrix / vector helpers with a fixed right-handed convention.

* Y is up, the camera looks down its local -Z axis.
* Perspective and orthographic projections map view-space depth to NDC Z in
  the closed interval ``[0, 1]``.
* All matrices are stored row-major as 4x4 arrays and use column vectors,
  i.e. ``clip = M @ point_h``.
"""

from __future__ import annotations

import math

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    length = float(np.linalg.norm(arr))
    if length == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return arr / length


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Perspective projection mapping view depth to NDC Z in [0, 1]."""
    if not (math.isfinite(fovy_rad) and math.isfinite(aspect)):
        raise ValueError("non-finite camera parameters")
    if fovy_rad <= 0.0 or fovy_rad >= math.pi or aspect <= 0.0:
        raise ValueError("invalid fovy or aspect ratio")
    if not (0.0 < near < far):
        raise ValueError("perspective requires 0 < near < far")
    f = 1.0 / math.tan(fovy_rad * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = (near * far) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Orthographic projection mapping view depth to NDC Z in [0, 1]."""
    if not (left < right and bottom < top and near < far):
        raise ValueError("invalid orthographic bounds")
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 2] = 1.0 / (near - far)
    m[2, 3] = near / (near - far)
    m[3, 3] = 1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up_world = normalize(np.asarray(up, dtype=np.float64))
    forward = normalize(target - eye)
    if abs(float(np.dot(forward, up_world))) > 0.999:
        up_world = normalize(np.array((0.0, 0.0, 1.0), dtype=np.float64))
    right = normalize(np.cross(forward, up_world))
    up_vec = np.cross(right, forward)
    m = np.identity(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = up_vec
    m[2, :3] = -forward
    m[0, 3] = -float(np.dot(right, eye))
    m[1, 3] = -float(np.dot(up_vec, eye))
    m[2, 3] = float(np.dot(forward, eye))
    return m


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def transform_points(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform Nx3 points returning Nx3 points after perspective divide."""
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1
    )
    clip = homogeneous @ m.T
    w = clip[:, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / w
    return ndc


def transform_clip(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return raw clip-space coordinates (Nx4) without perspective divide."""
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1
    )
    return homogeneous @ m.T


def frustum_corners(view_proj: np.ndarray) -> np.ndarray:
    """Return the 8 world-space corners of a camera frustum.

    Order: four near-plane corners then four far-plane corners, each set
    traversing (-x,-y), (+x,-y), (+x,+y), (-x,+y) in NDC.
    """
    ndc = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    homog = np.concatenate([ndc, np.ones((8, 1), dtype=np.float64)], axis=1)
    world_h = homog @ np.linalg.inv(view_proj).T
    return world_h[:, :3] / world_h[:, 3:4]


def is_finite_matrix(m: np.ndarray) -> bool:
    arr = np.asarray(m, dtype=np.float64)
    return arr.shape == (4, 4) and bool(np.isfinite(arr).all())

