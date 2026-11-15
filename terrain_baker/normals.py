"""Tangent-space normal map baking from a height field.

Gradients use Sobel (bilinear-weighted) central differences. The caller
supplies the height field with a 1-pixel wrap-around halo so edges and
chunk seams are continuous. Z is always non-negative.
"""

from __future__ import annotations

import numpy as np

HALO = 1  # Sobel needs a 1-pixel border


def normals_from_height(haloed_height: np.ndarray,
                        strength: float) -> np.ndarray:
    """Compute unit tangent-space normals from a haloed height field.

    ``haloed_height`` has shape (H + 2, W + 2); the returned normals have
    shape (H, W, 3) with z >= 0.
    """
    h = haloed_height
    gx = ((h[:-2, 2:] + 2.0 * h[1:-1, 2:] + h[2:, 2:])
          - (h[:-2, :-2] + 2.0 * h[1:-1, :-2] + h[2:, :-2]))
    gy = ((h[2:, :-2] + 2.0 * h[2:, 1:-1] + h[2:, 2:])
          - (h[:-2, :-2] + 2.0 * h[:-2, 1:-1] + h[:-2, 2:]))
    nx = -gx * strength
    ny = -gy * strength
    nz = np.full_like(nx, 8.0)  # Sobel kernel sum normalization
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normals = np.stack([nx, ny, nz], axis=-1) / length[:, :, None]
    return normals


def encode_normals(normals: np.ndarray) -> np.ndarray:
    """Encode unit normals to 8-bit RGB (0.5 offset, z stays >= 128)."""
    encoded = np.round((normals * 0.5 + 0.5) * 255.0)
    return np.clip(encoded, 0, 255).astype(np.uint8)


def decode_normals(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) / 255.0 * 2.0 - 1.0


def normalization_correction(rgb: np.ndarray) -> float:
    """Global correction factor for 8-bit quantization drift.

    The engine multiplies sampled (decoded) normals by this factor before
    renormalizing. Recorded in the manifest.
    """
    decoded = decode_normals(rgb)
    lengths = np.sqrt((decoded ** 2).sum(axis=-1))
    mean_length = float(lengths.mean())
    if mean_length <= 0.0:
        return 1.0
    return 1.0 / mean_length
