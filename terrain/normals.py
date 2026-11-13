"""Neighbour-difference normals, Lambert shading and sRGB gamma correction.

Normals use finite differences over the heightfield.  Border samples need one
ring outside the tile, which is taken from the global heightfield sampler, so
every vertex (including tile borders) gets a central-difference normal.
"""

from __future__ import annotations

import numpy as np

from .config import LightConfig, PipelineConfig
from .heightfield import Heightfield
from .quadtree import TileKey


def ring_expanded_grid(
    key: TileKey, heightfield: Heightfield, config: PipelineConfig
) -> tuple[np.ndarray, float, float, float]:
    """Return a ``(R+2) x (R+2)`` grid including the one-ring border."""
    x0, z0, x1, z1 = key.bounds(config.root_min, config.root_size)
    resolution = config.tile_resolution
    x0, z0, x1, z1 = key.bounds(config.root_min, config.root_size)
    spacing = (x1 - x0) / (resolution - 1)
    x_values = x0 + (np.arange(resolution) - 0) * spacing
    z_values = z0 + (np.arange(resolution) - 0) * spacing
    xs = np.concatenate(
        [[x_values[0] - spacing], x_values, [x_values[-1] + spacing]]
    )
    zs = np.concatenate(
        [[z_values[0] - spacing], z_values, [z_values[-1] + spacing]]
    )
    xx, zz = np.meshgrid(xs, zs)
    expanded = heightfield.sample_array(xx, zz)
    return expanded, float(spacing), float(x0), float(z0)


def tile_normals(
    key: TileKey, heightfield: Heightfield, config: PipelineConfig
) -> np.ndarray:
    """Unit float64 normals ``(R, R, 3)`` via central neighbour differences."""
    expanded, spacing, _, _ = ring_expanded_grid(key, heightfield, config)
    dzdx = (expanded[1:-1, 2:] - expanded[1:-1, :-2]) / (2.0 * spacing)
    dzdy = (expanded[2:, 1:-1] - expanded[:-2, 1:-1]) / (2.0 * spacing)
    normals = np.stack(
        [-dzdx, np.ones_like(dzdx), -dzdy], axis=-1
    )
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / lengths
    if np.any(~np.isfinite(normals)):
        raise ValueError("non-finite normal produced")
    return normals


def shade_vertex_colors(
    normals: np.ndarray, light: LightConfig
) -> np.ndarray:
    """Lambert reflectance in linear RGB, one channel, range [0, 1]."""
    direction = np.asarray(light.direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    diffuse = np.clip(normals @ direction, 0.0, 1.0)
    linear = np.clip(
        light.ambient + light.diffuse_albedo * diffuse, 0.0, 1.0
    )
    return linear


def gamma_encode(linear: np.ndarray, gamma: float) -> np.ndarray:
    """Linear float [0,1] -> sRGB-style gamma-encoded uint8."""
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.power(clipped, 1.0 / float(gamma))
    return np.rint(encoded * 255.0).astype(np.uint8)
