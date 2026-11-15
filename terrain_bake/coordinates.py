"""Shared coordinate-system constants and helpers.

The entire pipeline uses a right-handed world frame:

* X runs from 0 to 1 across heightfield columns.
* Z runs from 0 to 1 across heightfield rows.
* Y is up and stores ``heightfield * HEIGHT_SCALE``.

For the lowest LOD mesh, the tangent basis is T=+X, N=+Y and
B=cross(N,T)=-Z.  A tangent-space normal is encoded as (r, g, b) where
r is along T, g is along B, and b is along N.
"""

from __future__ import annotations

import numpy as np


WORLD_EXTENT_X = 1.0
WORLD_EXTENT_Z = 1.0
HEIGHT_SCALE = 0.32
BASE_TANGENT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)

MATERIAL_NAMES = ("grass", "rock", "snow")
MATERIAL_COLORS = np.array(
    [
        [0.22, 0.50, 0.16],
        [0.43, 0.40, 0.35],
        [0.92, 0.94, 0.98],
    ],
    dtype=np.float64,
)


def world_axis_step(samples: int, extent: float = 1.0) -> float:
    if samples < 2:
        raise ValueError("at least two samples are required")
    return float(extent) / float(samples - 1)


def heightfield_grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X/Z coordinate matrices matching heightfield [row, col] storage."""

    x = np.linspace(0.0, WORLD_EXTENT_X, cols, dtype=np.float64)
    z = np.linspace(0.0, WORLD_EXTENT_Z, rows, dtype=np.float64)
    return np.meshgrid(x, z)

