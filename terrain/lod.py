"""Conservative frustum culling and screen-space-error LOD selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import Frustum
from .config import PipelineConfig
from .heightfield import Heightfield
from .quadtree import TileKey, root_key


@dataclass(frozen=True)
class CullCounters:
    tested_per_level: tuple[int, ...]
    visible_per_level: tuple[int, ...]
    culled_per_level: tuple[int, ...]

    @property
    def total_culled(self) -> int:
        return sum(self.culled_per_level)


def tile_aabb(
    key: TileKey,
    config: PipelineConfig,
    height_bounds: dict,
) -> tuple[float, float, float, float, float, float]:
    """World AABB of a tile using measured height bounds (float64)."""
    x0, z0, x1, z1 = key.bounds(config.root_min, config.root_size)
    ymin, ymax = height_bounds[key]
    return x0, ymin, z0, x1, ymax, z1


class HeightBoundsCache:
    """Lazily measured tile height bounds (float64), computed on first use."""

    def __init__(self, config: PipelineConfig, heightfield: Heightfield):
        self._config = config
        self._heightfield = heightfield
        self._bounds: dict[TileKey, tuple[float, float]] = {}
        self._grids: dict = {}

    def aabb(self, key: TileKey):
        if key not in self._bounds:
            x0, z0, x1, z1 = key.bounds(
                self._config.root_min, self._config.root_size
            )
            grid = self._heightfield.tile_grid(
                x0, z0, x1, z1, self._config.tile_resolution
            )
            self._bounds[key] = (float(grid.min()), float(grid.max()))
            self._grids[key] = grid
        return tile_aabb(key, self._config, self._bounds)

    def get(self, key: TileKey) -> tuple[float, float]:
        if key not in self._bounds:
            self.aabb(key)
        return self._bounds[key]

    def grid(self, key: TileKey) -> np.ndarray:
        if key not in self._grids:
            self.aabb(key)
        return self._grids[key]


def _closest_distance(
    aabb, eye: np.ndarray
) -> float:
    """Distance from the eye to the nearest point of an AABB."""
    x0, y0, z0, x1, y1, z1 = aabb
    closest = np.array(
        [
            min(max(eye[0], x0), x1),
            min(max(eye[1], y0), y1),
            min(max(eye[2], z0), z1),
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(closest - eye))


def screen_space_error(
    key: TileKey,
    aabb,
    config: PipelineConfig,
) -> float:
    """Classic SSE: projected world-space geometric error in pixels."""
    eye = np.asarray(config.camera.eye, dtype=np.float64)
    distance = max(_closest_distance(aabb, eye), config.camera.near)
    focal = 1.0 / math.tan(math.radians(config.camera.fov_y_deg) * 0.5)
    geometric_error = config.root_size / (1 << key.level)
    return (
        geometric_error
        * focal
        * config.image_height
        / distance
    )


def sse_accepts(sse: float, config: PipelineConfig) -> bool:
    """SSE within threshold using a scale-dependent relative tolerance.

    The threshold itself is the scale: acceptance band is
    ``sse <= threshold * (1 + tol)``, keeping comparisons stable under
    floating-point reordering while remaining proportional to the metric.
    """
    threshold = float(config.sse_threshold)
    return sse <= threshold * (1.0 + float(config.sse_relative_tolerance))


def select_leaves(
    config: PipelineConfig,
    height_bounds: HeightBoundsCache,
) -> tuple[frozenset[TileKey], CullCounters]:
    """Deterministically cull and refine the quadtree into leaf tiles.

    Tiles are visited in a fixed (level, ix, iz) order; every tested tile is
    counted per level, culled subtrees never contribute descendants, and
    refinement stops on SSE acceptance or at ``max_depth`` so the result is a
    valid quadtree leaf set.
    """
    frustum = Frustum(config.camera)
    tested = [0] * (config.max_depth + 1)
    visible = [0] * (config.max_depth + 1)
    culled = [0] * (config.max_depth + 1)
    leaves: set[TileKey] = set()
    pending = [root_key()]
    while pending:
        key = pending.pop()
        tested[key.level] += 1
        aabb = height_bounds.aabb(key)
        if not frustum.intersects_aabb(aabb):
            culled[key.level] += 1
            continue
        visible[key.level] += 1
        if key.level >= config.max_depth or sse_accepts(
            screen_space_error(key, aabb, config), config
        ):
            leaves.add(key)
            continue
        pending.extend(reversed(key.children()))
    counters = CullCounters(
        tested_per_level=tuple(tested),
        visible_per_level=tuple(visible),
        culled_per_level=tuple(culled),
    )
    return frozenset(leaves), counters


def enforce_restricted_quadtree(
    leaves: frozenset[TileKey], max_depth: int
) -> frozenset[TileKey]:
    """Split coarse neighbours until adjacent leaves differ by at most one level.

    Splitting a leaf replaces it by its four children; the fixed key order and
    fixed-point iteration make the output deterministic regardless of input
    ordering.
    """
    current = set(leaves)
    while True:
        ordered = sorted(current, key=lambda tile: (tile.level, tile.ix, tile.iz))
        changed = False
        for tile in ordered:
            for direction in ("N", "S", "W", "E"):
                from .quadtree import adjacent_leaves

                neighbours = adjacent_leaves(frozenset(current), tile, direction)
                if not neighbours:
                    continue
                finest = max(leaf.level for leaf in neighbours)
                if tile.level - finest >= 2:
                    current.discard(tile)
                    current.update(tile.children())
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return frozenset(current)


def recount_leaves(
    leaves: frozenset[TileKey], max_depth: int
) -> tuple[int, ...]:
    """Per-level visible counts for a final (restricted) leaf set."""
    counts = [0] * (max_depth + 1)
    for leaf in leaves:
        counts[leaf.level] += 1
    return tuple(counts)
