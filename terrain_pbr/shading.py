"""Height-map conversion and deterministic tangent-space normal baking."""

from __future__ import annotations

import numpy as np


DEFAULT_NORMAL_RGB = (128, 128, 255)


def coverage_to_height(coverage: np.ndarray) -> np.ndarray:
    """Encode geometric raster coverage into an 8-bit height field."""

    clamped = np.clip(coverage, 0.0, 1.0)
    return np.floor(clamped * 255.0 + 0.5).astype(np.uint8)


def bake_tangent_normal(
    height: np.ndarray,
    *,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bake a 3x3 Sobel normal map. Output is RGB8, +Y follows image rows."""

    if height.ndim != 2:
        raise ValueError("height must be a 2D array")
    if strength <= 0.0:
        raise ValueError("strength must be > 0")

    height_f = height.astype(np.float64)
    padded = np.pad(height_f, ((1, 1), (1, 1)), mode="edge")

    gradient_x = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    ) / 8.0
    gradient_y = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    ) / 8.0

    zero_gradient = (gradient_x == 0.0) & (gradient_y == 0.0)
    normal_x = -gradient_x * strength
    normal_y = -gradient_y * strength
    normal_z = np.ones_like(height_f)
    length = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    normal_x /= length
    normal_y /= length
    normal_z /= length

    encoded = np.empty((*height.shape, 3), dtype=np.uint8)
    encoded[:, :, 0] = np.floor((normal_x * 0.5 + 0.5) * 255.0 + 0.5).astype(
        np.uint8
    )
    encoded[:, :, 1] = np.floor((normal_y * 0.5 + 0.5) * 255.0 + 0.5).astype(
        np.uint8
    )
    encoded[:, :, 2] = np.floor((normal_z * 0.5 + 0.5) * 255.0 + 0.5).astype(
        np.uint8
    )
    encoded[zero_gradient] = DEFAULT_NORMAL_RGB
    return encoded, gradient_x, gradient_y
