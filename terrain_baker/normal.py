"""Tangent-space normal map baking with clamped edge sampling and tiling.

Normals use central differences with an edge-clamped one-texel halo, the
same operator as the mesh normals, so map and geometry agree. Gradients
are divided by the LOD0 world spacing so they represent true world slopes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .memory import MemoryBudget


@dataclass
class TileStat:
    """Per-tile timing information."""

    name: str
    row: int
    col: int
    rows: int
    cols: int
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "row": int(self.row),
            "col": int(self.col),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "elapsed_ms": float(self.elapsed_ms),
        }


def _central_gradients(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients for every interior texel.

    The patch carries a one-texel clamped halo, so boundary texels use the
    same edge-clamped samples as the mesh normal computation. This is the
    identical operator to :func:`terrain_baker.lod.tangent_normals_from_height`,
    guaranteeing geometric consistency between map and mesh.
    """
    p = patch.astype(np.float32, copy=False)
    half = np.float32(0.5)
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) * half
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * half
    return gx, gy


def normals_from_height(
    height_patch: np.ndarray, spacing_world: float = 1.0
) -> np.ndarray:
    """Compute unit tangent-space normals for a clamped height patch.

    The patch must already include the one-texel clamped halo; the output
    covers only the interior, i.e. ``patch.shape[0]-2`` by ``...-2``.
    Gradients are divided by ``spacing_world`` so normals represent true
    world-space slopes, matching the mesh normal convention.
    """
    gx, gy = _central_gradients(height_patch)
    rows = height_patch.shape[0] - 2
    cols = height_patch.shape[1] - 2
    scale = np.float32(1.0 / spacing_world)
    normals = np.empty((rows, cols, 3), dtype=np.float32)
    normals[:, :, 0] = -gx * scale
    normals[:, :, 1] = -gy * scale
    normals[:, :, 2] = np.float32(1.0)
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    np.divide(normals, lengths, out=normals)
    return normals


def encode_rgb8(normals: np.ndarray) -> np.ndarray:
    """Map unit normals in [-1, 1] to RGB8 bytes in [0, 255]."""
    scaled = np.clip(normals, -1.0, 1.0) * np.float32(0.5) + np.float32(0.5)
    scaled = np.rint(scaled * np.float32(255.0)).astype(np.uint8)
    return scaled


def decode_rgb8(image: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_rgb8`, returning non-normalized float normals."""
    return image.astype(np.float32) * np.float32(2.0 / 255.0) - np.float32(1.0)


def _clamped_patch(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    n = grid.shape[0]
    rows = np.clip(np.arange(r0 - 1, r1 + 1), 0, n - 1)
    cols = np.clip(np.arange(c0 - 1, c1 + 1), 0, n - 1)
    return grid[np.ix_(rows, cols)]


def bake_normal_map(
    grid: np.ndarray,
    budget: MemoryBudget,
    tile: int = 512,
    spacing_world: float = 1.0,
) -> tuple[np.ndarray, list[TileStat]]:
    """Bake an RGB8 tangent-space normal map using fixed square tiles.

    Each tile reads a one-texel clamped halo, so boundary tiles reuse the
    exact edge values and no seams are introduced between neighbouring tiles.
    """
    n = grid.shape[0]
    if tile < 1:
        raise ValueError("tile size must be positive")
    out = np.empty((n, n, 3), dtype=np.uint8)
    stats: list[TileStat] = []
    tile_elems = (min(tile, n) + 2) ** 2
    working_bytes = tile_elems * grid.dtype.itemsize + (min(tile, n) ** 2) * (4 * 3 + 1)
    budget.acquire(working_bytes)
    try:
        for r0 in range(0, n, tile):
            r1 = min(r0 + tile, n)
            for c0 in range(0, n, tile):
                c1 = min(c0 + tile, n)
                start = time.perf_counter()
                patch = _clamped_patch(grid, r0, r1, c0, c1)
                normals = normals_from_height(patch, spacing_world)
                out[r0:r1, c0:c1, :] = encode_rgb8(normals)
                stats.append(
                    TileStat(
                        name="normal",
                        row=r0,
                        col=c0,
                        rows=r1 - r0,
                        cols=c1 - c0,
                        elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    )
                )
    finally:
        budget.release(working_bytes)
    return out, stats


def edge_mask(n: int) -> np.ndarray:
    """Boolean mask marking pixels whose gradients used clamped sampling."""
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def normal_length_error(image: np.ndarray) -> dict:
    """Report length error of decoded normals before and after normalization."""
    decoded = decode_rgb8(image)
    raw_length = np.linalg.norm(decoded, axis=2)
    normalized = decoded / np.maximum(raw_length[..., None], np.float32(1e-12))
    normalized_length = np.linalg.norm(normalized, axis=2)
    return {
        "max_raw_length_error": float(np.max(np.abs(raw_length - 1.0))),
        "max_normalized_length_error": float(
            np.max(np.abs(normalized_length - 1.0))
        ),
        "mean_raw_length_error": float(np.mean(np.abs(raw_length - 1.0))),
    }
