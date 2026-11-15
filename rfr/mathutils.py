"""Column-major 4x4 matrix and vector helpers.

All scene-graph and camera math is performed in ``float64`` as required by the
specification.  Matrices are stored and returned as ``numpy.ndarray`` with
shape ``(4, 4)`` using *column vector* convention (a point is transformed with
``M @ v``).  Right before rendering, world matrices are cast to ``float32`` so
that the GPU and software paths consume bit-identical inputs.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .errors import RenderError


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translation(v: Sequence[float]) -> np.ndarray:
    m = identity()
    m[0, 3], m[1, 3], m[2, 3] = float(v[0]), float(v[1]), float(v[2])
    return m


def scaling(v: Sequence[float]) -> np.ndarray:
    m = identity()
    m[0, 0], m[1, 1], m[2, 2] = float(v[0]), float(v[1]), float(v[2])
    return m


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = identity()
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = identity()
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    m = identity()
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def compose(
    translate: Sequence[float],
    rotate_euler: Sequence[float],
    scale: Sequence[float],
) -> np.ndarray:
    """Build a local TRS matrix: T * Rz * Ry * Rx * S using float64."""
    rx, ry, rz = (float(a) for a in rotate_euler)
    rot = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
    return translation(translate) @ rot @ scaling(scale)


def perspective(fov_y_radians: float, aspect: float, near: float, far: float) -> np.ndarray:
    """OpenGL-style perspective projection (NDC z in [-1, 1]), float64."""
    if near <= 0.0 or far <= near:
        raise RenderError("E_DIMENSION", "Invalid camera near/far planes.",
                          {"near": near, "far": far})
    f = 1.0 / math.tan(fov_y_radians / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    fn = np.linalg.norm(forward)
    if fn == 0.0:
        raise RenderError("E_DIMENSION", "Camera eye and target coincide.")
    forward /= fn
    right = np.cross(forward, up)
    if np.linalg.norm(right) == 0.0:
        raise RenderError("E_DIMENSION", "Camera up vector is parallel to view direction.")
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    m = identity()
    m[0, 0:3] = right
    m[1, 0:3] = true_up
    m[2, 0:3] = -forward
    m[0, 3] = -np.dot(right, eye)
    m[1, 3] = -np.dot(true_up, eye)
    m[2, 3] = np.dot(forward, eye)
    return m


def normal_matrix(world: np.ndarray) -> np.ndarray:
    """Inverse transpose of the upper 3x3, used to transform normals."""
    upper = np.asarray(world, dtype=np.float64)[:3, :3]
    try:
        inv = np.linalg.inv(upper)
    except np.linalg.LinAlgError as exc:
        raise RenderError("E_DIMENSION", "Singular transform for normals.") from exc
    return inv.T


def to_float32(m: np.ndarray) -> np.ndarray:
    return np.asarray(m, dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr, axis=-1, keepdims=arr.ndim > 1)
    n = np.where(n == 0.0, 1.0, n)
    return arr / n


def transform_point(m: np.ndarray, p: Sequence[float]) -> Tuple[float, float, float]:
    v = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = m @ v
    return float(out[0]), float(out[1]), float(out[2])
