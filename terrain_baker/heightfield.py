"""Deterministic procedural heightfield stored as a memory-mapped file.

The heightfield is written row-block by row-block into a raw little-endian
float32 ``.dat`` file via :class:`numpy.memmap`. The full grid is therefore
never resident in RAM; later stages stream rows/halos from the same file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .errors import ParameterError
from .memory import MemoryBudget
from .noise import fbm_2d

MIN_RESOLUTION = 257
MAX_RESOLUTION = 4097

# Rough upper bound on simultaneous float64 temporaries inside the fBm path.
_FBM_TEMP_ARRAY_FACTOR = 16


@dataclass(frozen=True)
class HeightfieldSpec:
    """Inputs describing one deterministic heightfield."""

    seed: int
    resolution: int
    height_scale: float
    world_size: float = 1000.0
    octaves: int = 5
    frequency: float = 2.0

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "resolution": int(self.resolution),
            "height_scale": float(self.height_scale),
            "world_size": float(self.world_size),
            "octaves": int(self.octaves),
            "frequency": float(self.frequency),
        }


def validate_resolution(resolution: int) -> None:
    if not isinstance(resolution, int) or isinstance(resolution, bool):
        raise ParameterError("resolution must be an integer")
    if resolution < MIN_RESOLUTION or resolution > MAX_RESOLUTION:
        raise ParameterError(
            f"resolution must be within [{MIN_RESOLUTION}, {MAX_RESOLUTION}], "
            f"got {resolution}"
        )
    if resolution % 2 == 0:
        raise ParameterError(f"resolution must be odd, got {resolution}")


def validate_spec(spec: HeightfieldSpec) -> None:
    validate_resolution(spec.resolution)
    if not np.isfinite(spec.seed):
        raise ParameterError("seed must be a finite integer")
    if not np.isfinite(spec.height_scale) or spec.height_scale < 0:
        raise ParameterError("height_scale must be a non-negative finite number")
    if spec.world_size <= 0 or not np.isfinite(spec.world_size):
        raise ParameterError("world_size must be positive and finite")
    if spec.octaves < 1:
        raise ParameterError("octaves must be >= 1")
    if spec.frequency <= 0 or not np.isfinite(spec.frequency):
        raise ParameterError("frequency must be positive and finite")


def height_values_for_rows(rows: np.ndarray, spec: HeightfieldSpec) -> np.ndarray:
    """Compute float32 heights for an array of row indices (1D or 2D grid)."""
    n = spec.resolution
    jj = rows.astype(np.float64)
    ii = np.arange(n, dtype=np.float64)
    v = jj / np.float64(n - 1)
    u = ii / np.float64(n - 1)
    x = u * np.float64(spec.frequency)
    y = v * np.float64(spec.frequency)
    x_grid, y_grid = np.meshgrid(x, y, copy=True)
    field = fbm_2d(x_grid, y_grid, seed=int(spec.seed), octaves=spec.octaves)
    return (field * np.float64(spec.height_scale)).astype(np.float32)


def open_heightfield(path: str, resolution: int, mode: str = "r") -> np.memmap:
    """Open a previously baked raw float32 heightfield as a memmap."""
    return np.memmap(path, dtype="<f4", mode=mode, shape=(resolution, resolution))


def build_heightfield(
    path: str,
    spec: HeightfieldSpec,
    budget: MemoryBudget,
    block_rows: int = 64,
) -> np.memmap:
    """Generate the heightfield into ``path`` and return a read/write memmap."""
    validate_spec(spec)
    if block_rows < 1:
        raise ParameterError("block_rows must be >= 1")
    n = spec.resolution
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    grid = np.memmap(path, dtype="<f4", mode="w+", shape=(n, n))
    block_elems = min(block_rows, n) * n
    working_bytes = block_elems * 8 * _FBM_TEMP_ARRAY_FACTOR
    budget.acquire(working_bytes)
    try:
        for start in range(0, n, block_rows):
            end = min(start + block_rows, n)
            row_index = np.arange(start, end, dtype=np.int64)
            grid[start:end, :] = height_values_for_rows(row_index, spec)
            grid.flush()
    finally:
        budget.release(working_bytes)
    grid.flush()
    return grid
