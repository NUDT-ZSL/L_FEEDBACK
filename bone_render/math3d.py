from __future__ import annotations

import numpy as np

from .errors import SceneError


F64 = np.float64


def normalize(vector: np.ndarray, path: str = "$", *, code: str = "E_SCHEMA") -> np.ndarray:
    vector = np.asarray(vector, dtype=F64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-12:
        raise SceneError(code, path, "vector must have a finite non-zero length")
    return vector / length


def safe_normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    length = np.linalg.norm(vector, axis=axis, keepdims=True)
    return np.divide(
        vector,
        length,
        out=np.zeros_like(vector),
        where=np.isfinite(length) & (length > 1e-12),
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    right = np.cross(forward, up)
    if np.linalg.norm(right) <= 1e-10:
        raise SceneError("E_CAMERA", "camera.up", "camera up vector is parallel to view direction")
    right = normalize(right)
    true_up = normalize(np.cross(right, forward))
    matrix = np.eye(4, dtype=F64)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye))
    matrix[1, 3] = -float(np.dot(true_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=F64)], axis=1
    )
    transformed = homogeneous @ matrix.T
    return transformed[:, :3]


def project_points(
    view_points: np.ndarray,
    *,
    fov_y_radians: float,
    aspect: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    depths = -view_points[:, 2]
    focal = 1.0 / np.tan(fov_y_radians * 0.5)
    ndc_x = (focal / aspect) * view_points[:, 0] / depths
    ndc_y = focal * view_points[:, 1] / depths
    screen_x = (ndc_x + 1.0) * 0.5 * width
    screen_y = (1.0 - ndc_y) * 0.5 * height
    return np.stack([screen_x, screen_y], axis=1), depths
