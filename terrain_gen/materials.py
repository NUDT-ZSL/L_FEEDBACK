"""Material region classification from elevation and slope."""

from __future__ import annotations

import numpy as np

from .normals import sobel_gradients

MATERIAL_NAMES = ("grass", "rock", "sand", "snow")
GRASS, ROCK, SAND, SNOW = 0, 1, 2, 3


def slope_field(heightfield: np.ndarray) -> np.ndarray:
    """Gradient magnitude derived from the same Sobel operator as normals."""
    dz_dx, dz_dy = sobel_gradients(heightfield)
    return np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)


def classify_materials(
    heightfield: np.ndarray,
    sand_level: float = 0.25,
    snow_level: float = 0.72,
    rock_slope: float = 0.03,
) -> np.ndarray:
    """Assign a deterministic material ID to every terrain cell.

    Decision order (first match wins, giving stable tie handling):
      1. ``sand`` below ``sand_level``;
      2. ``rock`` on slopes steeper than ``rock_slope``;
      3. ``snow`` above ``snow_level`` on gentle slopes;
      4. ``grass`` everywhere else.
    """
    slope = slope_field(heightfield)
    material = np.full(heightfield.shape, GRASS, dtype=np.uint16)
    material[(heightfield >= sand_level) & (slope >= rock_slope)] = ROCK
    material[heightfield < sand_level] = SAND
    material[(heightfield >= snow_level) & (slope < rock_slope)] = SNOW
    return material


def material_statistics(material: np.ndarray) -> dict[str, float]:
    """Return the pixel fraction of each material plus the raw pixel counts."""
    total = float(material.size)
    counts = np.bincount(material.ravel(), minlength=len(MATERIAL_NAMES))
    fractions = {
        name: float(counts[index]) / total
        for index, name in enumerate(MATERIAL_NAMES)
    }
    return {
        "pixel_counts": {
            name: int(counts[index]) for index, name in enumerate(MATERIAL_NAMES)
        },
        "fractions": fractions,
        "fractions_sum": float(sum(fractions.values())),
        "total_pixels": int(total),
    }
