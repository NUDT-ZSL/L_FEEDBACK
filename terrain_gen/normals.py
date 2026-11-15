"""Tangent-space normal map generation from heightfields."""

from __future__ import annotations

import numpy as np


def _periodic_roll(array: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    return np.roll(np.roll(array, row_shift, axis=0), col_shift, axis=1)


def sobel_gradients(heightfield: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Sobel ``(dz_dx, dz_dy)`` on a periodically wrapping grid."""
    # Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] / 8
    dz_dx = (
        _periodic_roll(heightfield, -1, 1)
        + 2.0 * _periodic_roll(heightfield, 0, 1)
        + _periodic_roll(heightfield, 1, 1)
        - _periodic_roll(heightfield, -1, -1)
        - 2.0 * _periodic_roll(heightfield, 0, -1)
        - _periodic_roll(heightfield, 1, -1)
    ) / 8.0
    # Sobel Y kernel: transpose of X.
    dz_dy = (
        _periodic_roll(heightfield, 1, -1)
        + 2.0 * _periodic_roll(heightfield, 1, 0)
        + _periodic_roll(heightfield, 1, 1)
        - _periodic_roll(heightfield, -1, -1)
        - 2.0 * _periodic_roll(heightfield, -1, 0)
        - _periodic_roll(heightfield, -1, 1)
    ) / 8.0
    return dz_dx, dz_dy


def compute_normals(heightfield: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Compute unit tangent-space normals as an ``(H, W, 3)`` float64 array.

    The normal points along ``(-dz/dx, -dz/dy, +strength)`` (OpenGL convention,
    ``+Z`` out of the surface) and is normalized per pixel.
    """
    dz_dx, dz_dy = sobel_gradients(heightfield)
    normals = np.empty((*heightfield.shape, 3), dtype=np.float64)
    normals[..., 0] = -dz_dx * strength
    normals[..., 1] = -dz_dy * strength
    normals[..., 2] = strength
    length = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / length


def encode_normals_rgb(normals: np.ndarray) -> np.ndarray:
    """Map unit vectors from ``[-1, 1]`` to 8-bit ``[0, 255]`` channels."""
    encoded = np.rint((normals * 0.5 + 0.5) * 255.0).clip(0, 255)
    return encoded.astype(np.uint8)
