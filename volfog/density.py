"""Deterministic procedural density fields and external .npy import."""

from __future__ import annotations

import numpy as np

from .config import normalize_density, validate_density_array
from .errors import RenderError


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _value_noise_3d(
    grid: np.ndarray,
    perm: np.ndarray,
    lattice: int,
) -> np.ndarray:
    """Trilinearly interpolated lattice value noise, deterministic given perm."""
    x0 = np.floor(grid[..., 0]).astype(np.int64)
    y0 = np.floor(grid[..., 1]).astype(np.int64)
    z0 = np.floor(grid[..., 2]).astype(np.int64)
    tx = _smoothstep(grid[..., 0] - x0)
    ty = _smoothstep(grid[..., 1] - y0)
    tz = _smoothstep(grid[..., 2] - z0)

    def lattice_hash(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        h = perm[(ix % lattice) & 255]
        h = perm[(h + (iy % lattice)) & 255]
        h = perm[(h + (iz % lattice)) & 255]
        return h.astype(np.float64) / 255.0

    c000 = lattice_hash(x0, y0, z0)
    c100 = lattice_hash(x0 + 1, y0, z0)
    c010 = lattice_hash(x0, y0 + 1, z0)
    c110 = lattice_hash(x0 + 1, y0 + 1, z0)
    c001 = lattice_hash(x0, y0, z0 + 1)
    c101 = lattice_hash(x0 + 1, y0, z0 + 1)
    c011 = lattice_hash(x0, y0 + 1, z0 + 1)
    c111 = lattice_hash(x0 + 1, y0 + 1, z0 + 1)

    cx00 = c000 + (c100 - c000) * tx
    cx10 = c010 + (c110 - c010) * tx
    cx01 = c001 + (c101 - c001) * tx
    cx11 = c011 + (c111 - c011) * tx
    cy0 = cx00 + (cx10 - cx00) * ty
    cy1 = cx01 + (cx11 - cx01) * ty
    return cy0 + (cy1 - cy0) * tz


def generate_density_field(resolution: int = 64, seed: int = 1234) -> np.ndarray:
    """Build a deterministic 3D FBM density field normalized to [0, 1].

    A seeded ``np.random.default_rng`` permutation table drives value-noise
    octaves, so identical (resolution, seed) always yields identical voxels.
    """
    if not isinstance(resolution, (int, np.integer)) or resolution < 2:
        raise RenderError("voxel_res", "voxel_res must be an integer >= 2")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(256).astype(np.int64)

    axis = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([gx, gy, gz], axis=-1)

    field = np.zeros((resolution,) * 3, dtype=np.float64)
    amplitude = 1.0
    amplitude_sum = 0.0
    frequency = 3.0
    for _ in range(4):
        lattice = max(2, int(round(frequency)))
        field += amplitude * _value_noise_3d(points * frequency, perm, lattice)
        amplitude_sum += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    field /= amplitude_sum

    # Gentle radial envelope keeps the fog inside a cloud-like volume.
    centered = points - 0.5
    radius = np.sqrt(np.sum(centered * centered, axis=-1)) * 2.0
    envelope = np.clip(1.0 - radius * radius, 0.0, 1.0)
    field = field * (0.35 + 0.65 * envelope)
    return normalize_density(field)


def load_density_field(path: str) -> np.ndarray:
    """Load an external 3D density field from a .npy file and normalize it."""
    try:
        density = np.load(path)
    except FileNotFoundError:
        raise RenderError("density_file", f"file not found: {path}")
    except Exception as exc:  # malformed / unreadable .npy
        raise RenderError("density_file", f"cannot load numpy file: {exc}")
    density = validate_density_array(density, param="density_file")
    return normalize_density(density)
