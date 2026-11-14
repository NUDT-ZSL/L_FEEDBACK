"""World-space normal map baker.

Normals are computed with the exact stencil that matches area-weighted
accumulation of the terrain triangle face normals produced by
`lod_mesh.build_chunk_mesh` (fixed diagonal, SW->NE).  Border derivatives
use clamped (edge-replicated) samples, so the normal map agrees with the
mesh geometry at chunk boundaries to floating-point precision.

Encoding: RGB8, Z-up, n -> round((n * 0.5 + 0.5) * 255).  Image row 0
corresponds to grid y index 0 (south edge).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .heightfield import HeightField

# Stencil of the fixed-diagonal triangulation (see lod_mesh).  Applied to
# edge-padded heights and scaled by the sample spacing.
_KX = np.array([[1.0, -1.0, 0.0],
                [2.0, 0.0, -2.0],
                [0.0, 1.0, -1.0]])
_KY = np.array([[1.0, 2.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, -2.0, -1.0]])


def normals_from_heights(heights: np.ndarray, spacing: float) -> np.ndarray:
    """Compute unit world-space normals for a regular height grid.

    heights: (n, n) array, row = y.  Returns (n, n, 3) unit normals.
    Uses clamped sampling at the borders (edge padding).
    """
    hp = np.pad(heights, 1, mode="edge")
    n = heights.shape[0]
    nx = np.zeros_like(heights)
    ny = np.zeros_like(heights)
    for dj in range(3):
        for di in range(3):
            window = hp[dj:dj + n, di:di + n]
            if _KX[dj, di]:
                nx += window * _KX[dj, di]
            if _KY[dj, di]:
                ny += window * _KY[dj, di]
    nx *= spacing
    ny *= spacing
    nz = np.full_like(heights, 6.0 * spacing * spacing)
    normals = np.stack([nx, ny, nz], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def encode_rgb8(normals: np.ndarray) -> np.ndarray:
    return np.round(normals * 127.5 + 127.5).astype(np.uint8)


def decode_rgb8(rgb: np.ndarray) -> np.ndarray:
    n = rgb.astype(np.float64) / 127.5 - 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def bake_normal_map(field: HeightField, cx: int, cy: int, step: int = 1) -> Image.Image:
    """Bake the normal map for chunk (cx, cy) at the given sample step.

    The sampling grid (step, clamping) is identical to the LOD mesh built
    with the same step, per the shared sampling rule.
    """
    _, _, heights = field.chunk_grid(cx, cy, step)
    normals = normals_from_heights(heights, field.spacing * step)
    rgb = encode_rgb8(normals)
    return Image.fromarray(rgb, mode="RGB")
