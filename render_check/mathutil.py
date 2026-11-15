"""Small deterministic matrix helpers for an OpenGL 4.1 renderer."""

from __future__ import annotations

import math

import numpy as np


def _as_f32(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)


def translation(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray(
        (
            (1.0, 0.0, 0.0, x),
            (0.0, 1.0, 0.0, y),
            (0.0, 0.0, 1.0, z),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def scale(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray(
        (
            (x, 0.0, 0.0, 0.0),
            (0.0, y, 0.0, 0.0),
            (0.0, 0.0, z, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, c, -s, 0.0),
            (0.0, s, c, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        (
            (c, 0.0, s, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (-s, 0.0, c, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        (
            (c, -s, 0.0, 0.0),
            (s, c, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float],
) -> np.ndarray:
    eye_vec = np.asarray(eye, dtype=np.float64)
    target_vec = np.asarray(target, dtype=np.float64)
    up_vec = np.asarray(up, dtype=np.float64)
    forward = target_vec - eye_vec
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up_vec)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)
    matrix = np.identity(4, dtype=np.float64)
    matrix[0, :3] = side
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(side, eye_vec))
    matrix[1, 3] = -float(np.dot(true_up, eye_vec))
    matrix[2, 3] = float(np.dot(forward, eye_vec))
    return _as_f32(matrix)


def perspective(
    fov_y_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return _as_f32(matrix)


def model_matrix(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    uniform_scale: float,
) -> np.ndarray:
    transform = translation(*position)
    transform = transform @ rotation_y(rotation[1])
    transform = transform @ rotation_x(rotation[0])
    transform = transform @ rotation_z(rotation[2])
    transform = transform @ scale(uniform_scale, uniform_scale, uniform_scale)
    return np.ascontiguousarray(transform, dtype=np.float32)


def column_major_bytes(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype="<f4").tobytes(order="F")

