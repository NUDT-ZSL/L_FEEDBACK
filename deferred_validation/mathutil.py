"""Small deterministic math helpers shared by scene loading and analysis."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import glm
import numpy as np


def vec3(value: np.ndarray | list[float] | tuple[float, ...]) -> glm.vec3:
    array = np.asarray(value, dtype=np.float64).reshape(3)
    return glm.vec3(float(array[0]), float(array[1]), float(array[2]))


def vec3_array(value: glm.vec3) -> np.ndarray:
    return np.array([value.x, value.y, value.z], dtype=np.float32)


def normalize(value: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        if fallback is None:
            raise ValueError("cannot normalize a zero-length vector")
        return np.asarray(fallback, dtype=np.float64)
    return value / length


def euler_deg_matrix(rotation_deg: np.ndarray) -> glm.mat4:
    """Intrinsic X, then Y, then Z rotation, matching translate * Rz * Ry * Rx * scale."""
    rx, ry, rz = [float(v) for v in np.asarray(rotation_deg).reshape(3)]
    matrix = glm.mat4(1.0)
    matrix = glm.rotate(matrix, glm.radians(rz), glm.vec3(0.0, 0.0, 1.0))
    matrix = glm.rotate(matrix, glm.radians(ry), glm.vec3(0.0, 1.0, 0.0))
    matrix = glm.rotate(matrix, glm.radians(rx), glm.vec3(1.0, 0.0, 0.0))
    return matrix


def node_matrix(translation: np.ndarray, rotation_deg: np.ndarray, scale: np.ndarray) -> glm.mat4:
    matrix = glm.translate(glm.mat4(1.0), vec3(translation))
    matrix = matrix * euler_deg_matrix(rotation_deg)
    matrix = glm.scale(matrix, vec3(scale))
    return matrix


def glm_to_numpy(matrix: glm.mat4, dtype=np.float32) -> np.ndarray:
    return np.asarray(matrix, dtype=dtype)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ matrix.astype(np.float64).T
    return (transformed[:, :3] / transformed[:, 3:4]).astype(np.float32)


def cosine_degrees(value: float) -> float:
    return float(np.cos(np.deg2rad(value)))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def downsample_box(array: np.ndarray, factor: int) -> np.ndarray:
    """Downsample HxWxC by an integer box factor."""
    if factor == 1:
        return np.asarray(array)
    height, width = array.shape[:2]
    if height % factor or width % factor:
        raise ValueError("array dimensions are not divisible by SSAA factor")
    return array.reshape(height // factor, factor, width // factor, factor, *array.shape[2:]).mean(
        axis=(1, 3)
    )


def tone_map(linear_rgb: np.ndarray, exposure: float, operator: str, gamma: float = 2.2) -> np.ndarray:
    """CPU reference implementation matching the GLSL tone mapper."""
    rgb = np.maximum(np.asarray(linear_rgb, dtype=np.float64) * float(exposure), 0.0)
    if operator == "aces":
        mapped = (rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14)
    elif operator == "reinhard":
        mapped = rgb / (1.0 + rgb)
    else:
        raise ValueError(f"unsupported tone mapper: {operator}")
    mapped = np.clip(mapped, 0.0, 1.0)
    return np.power(mapped, 1.0 / float(gamma))
