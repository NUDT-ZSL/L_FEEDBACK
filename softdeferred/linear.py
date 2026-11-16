from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np


EPS = 1.0e-12


def normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(length, EPS)


def translation(values: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(values, dtype=np.float64)
    return matrix


def scaling(values: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = np.asarray(
        values, dtype=np.float64
    )
    return matrix


def rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def compose_trs(
    translate: Iterable[float] = (0.0, 0.0, 0.0),
    rotate_xyz_deg: Iterable[float] = (0.0, 0.0, 0.0),
    scale: Iterable[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    rx, ry, rz = [float(value) for value in rotate_xyz_deg]
    rotation = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
    return translation(list(translate)) @ rotation @ scaling(list(scale))


def matrix_from_json(value: Any, context: str) -> np.ndarray:
    if value is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(value, dict):
        allowed = {"translate", "rotate_xyz_deg", "scale"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("%s has unsupported transform keys: %s" % (context, sorted(unknown)))
        return compose_trs(
            value.get("translate", (0.0, 0.0, 0.0)),
            value.get("rotate_xyz_deg", (0.0, 0.0, 0.0)),
            value.get("scale", (1.0, 1.0, 1.0)),
        )
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (16,):
        array = array.reshape(4, 4)
    if array.shape != (4, 4):
        raise ValueError("%s must be a 4x4 matrix or a transform object" % context)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s contains non-finite values" % context)
    return array


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    eye_array = np.asarray(eye, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    up_array = normalize(np.asarray(up, dtype=np.float64))
    forward = normalize(target_array - eye_array)
    side = np.cross(forward, up_array)
    if np.linalg.norm(side) < 1.0e-8:
        fallback = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(forward, fallback))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        side = np.cross(forward, fallback)
    side = normalize(side)
    upward = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = side
    matrix[1, :3] = upward
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(side, eye_array))
    matrix[1, 3] = -float(np.dot(upward, eye_array))
    matrix[2, 3] = float(np.dot(forward, eye_array))
    return matrix


def perspective(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
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


def transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    homogeneous = np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    transformed = matrix @ homogeneous
    return transformed[:3] / transformed[3]


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [array, np.ones((array.shape[0], 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ matrix.T
    return transformed[:, :3] / transformed[:, 3:4]


def transform_vectors(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float64) @ matrix[:3, :3].T


def frustum_planes(view_projection: np.ndarray) -> np.ndarray:
    matrix = np.asarray(view_projection, dtype=np.float64)
    rows = matrix[3, :]
    planes = np.stack(
        [
            rows + matrix[0, :],
            rows - matrix[0, :],
            rows + matrix[1, :],
            rows - matrix[1, :],
            rows + matrix[2, :],
            rows - matrix[2, :],
        ]
    )
    lengths = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    return planes / np.maximum(lengths, EPS)


def sphere_outside_frustum(
    center: Sequence[float], radius: float, planes: np.ndarray
) -> bool:
    distances = planes[:, :3] @ np.asarray(center, dtype=np.float64) + planes[:, 3]
    return bool(np.any(distances < -float(radius)))


def bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.zeros(3, dtype=np.float64), 0.0
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.max(np.linalg.norm(array - center, axis=1)))
    return center, radius


def vector3(value: Any, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("%s must contain exactly three numbers" % context)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s contains non-finite values" % context)
    return array


def finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number" % context)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % context)
    return result


def positive_int(value: Any, context: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be an integer" % context)
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer" % context)
    if result != value or result < minimum:
        raise ValueError("%s must be an integer >= %d" % (context, minimum))
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
