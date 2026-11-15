"""Deterministic, tileable fractal value-noise heightfield generation.

All randomness flows from a single :class:`numpy.random.Generator` seeded from
the configuration, so identical configurations reproduce byte-identical output.
Noise lattices wrap around both axes, making the resulting heightfield periodic
and therefore free of seams when tiled.
"""

from __future__ import annotations

import numpy as np

from .config import NoiseConfig


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _periodic_value_noise(
    width: int,
    height: int,
    lattice_x: int,
    lattice_y: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a bilinearly interpolated, periodically tiling lattice."""
    grid = rng.random((lattice_y, lattice_x), dtype=np.float64)

    x = np.arange(width, dtype=np.float64) * lattice_x / width
    y = np.arange(height, dtype=np.float64) * lattice_y / height
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = _smoothstep(x - x0)
    ty = _smoothstep(y - y0)
    x1 = (x0 + 1) % lattice_x
    y1 = (y0 + 1) % lattice_y

    top_left = grid[np.ix_(y0, x0)]
    top_right = grid[np.ix_(y0, x1)]
    bottom_left = grid[np.ix_(y1, x0)]
    bottom_right = grid[np.ix_(y1, x1)]

    top = top_left + (top_right - top_left) * tx[np.newaxis, :]
    bottom = bottom_left + (bottom_right - bottom_left) * tx[np.newaxis, :]
    return top + (bottom - top) * ty[:, np.newaxis]


def generate_heightfield(width: int, height: int, noise_cfg: NoiseConfig) -> np.ndarray:
    """Generate a normalized fractal heightfield with values in ``[0, 1]``.

    Output is a ``(height, width)`` float64 array. Normalization is min-max,
    so the field always contains exactly one minimum of 0 and one maximum of 1.
    """
    rng = np.random.default_rng(noise_cfg.seed)
    field = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    amplitude_sum = 0.0

    for octave in range(noise_cfg.octaves):
        lattice_x = max(1, noise_cfg.base_frequency * int(round(noise_cfg.lacunarity ** octave)))
        lattice_y = lattice_x
        # Dense lattices add no information beyond the output resolution.
        lattice_x = min(lattice_x, width)
        lattice_y = min(lattice_y, height)
        field += amplitude * _periodic_value_noise(width, height, lattice_x, lattice_y, rng)
        amplitude_sum += amplitude
        amplitude *= noise_cfg.persistence

    field /= amplitude_sum
    minimum = float(field.min())
    maximum = float(field.max())
    if maximum > minimum:
        field = (field - minimum) / (maximum - minimum)
    else:
        field = np.zeros_like(field)
    np.clip(field, 0.0, 1.0, out=field)
    return field
