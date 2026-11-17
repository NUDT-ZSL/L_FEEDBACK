"""Small deterministic vector and matrix helpers.

All matrices use column vectors and are applied as ``M @ v``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


FLOAT_EPS = 1e-9


def vec3(value: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(value), dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("expected a 3-component vector")
    return arr


def normalize(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= FLOAT_EPS:
        if fallback is None:
            return np.zeros(3, dtype=np.float64)
        return fallback.copy()
    return value / length


def normalize_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    safe = np.where(lengths > FLOAT_EPS, lengths, 1.0)
    result = values / safe[:, None]
    if fallback is not None:
        bad = lengths <= FLOAT_EPS
        if fallback.ndim == 1:
            result[bad] = fallback
        else:
            result[bad] = fallback[bad]
    return result


def translation(offset: Iterable[float]) -> np.ndarray:
    x, y, z = vec3(offset)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix


def scaling(scale: Iterable[float]) -> np.ndarray:
    x, y, z = vec3(scale)
    return np.diag(np.array([x, y, z, 1.0], dtype=np.float64))


def rotation_x(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_y(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def model_matrix(
    offset: Iterable[float] = (0.0, 0.0, 0.0),
    rotation_degrees: Iterable[float] = (0.0, 0.0, 0.0),
    scale: Iterable[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    rx, ry, rz = vec3(rotation_degrees)
    rotation = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
    return translation(offset) @ rotation @ scaling(scale)


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - position)
    right = normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, position))
    matrix[1, 3] = -float(np.dot(true_up, position))
    matrix[2, 3] = float(np.dot(forward, position))
    return matrix


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def orthographic(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 2.0 / (right - left)
    matrix[1, 1] = 2.0 / (top - bottom)
    matrix[2, 2] = -2.0 / (far - near)
    matrix[0, 3] = -(right + left) / (right - left)
    matrix[1, 3] = -(top + bottom) / (top - bottom)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (matrix @ homogeneous.T).T


def transform_directions(matrix: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ directions.T).T


def normal_matrix(model: np.ndarray) -> np.ndarray:
    return np.linalg.inv(model[:3, :3]).T


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def quantize_srgb(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.clip(linear_to_srgb(clipped), 0.0, 1.0)
    return np.floor(encoded * 255.0 + 0.5).astype(np.uint8)


def screen_triangle_depth(ndc_z: np.ndarray) -> np.ndarray:
    """Map OpenGL-style NDC z in [-1, 1] to raster depth in [0, 1]."""
    return np.clip(ndc_z * 0.5 + 0.5, 0.0, 1.0)
