"""Version-based incremental tile redraw with strict cache invalidation.

Scene and camera each carry a content-hash version id. A tile is
recomputed only when the scene or camera version actually changed AND the
tile may be affected (union of the previous frame's touched tiles and the
new projections). Unchanged tiles reuse the previous frame's results.

Disk cache records are keyed by an explicit snapshot generation plus both
version ids. When the generation or versions no longer match, the whole
cache is discarded -- stale file offsets are never reused as a "hit".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

from .arch import segment_aabb
from .camera import Camera
from .raster import RenderResult, render
from .scene import Scene
from .tiles import cross_check, empty_manifest, make_tiles, recount_tile

CACHE_FILE = "tile_cache.json"


@dataclass
class FrameResult:
    generation: int
    scene_version: str
    camera_version: str
    color: np.ndarray
    depth: np.ndarray
    ids: np.ndarray
    manifest: List[dict]
    dirty_tiles: List[int]
    reused_tiles: List[int]
    manifest_diffs: List[dict] = field(default_factory=list)


class IncrementalRenderer:
    def __init__(self, width: int, height: int, tile_width: int):
        self.width = width
        self.height = height
        self.tile_width = tile_width
        self.tiles = make_tiles(width, tile_width)
        self.generation = 0
        self._last_scene_version: Optional[str] = None
        self._last_camera_version: Optional[str] = None
        self._last_touched: Set[int] = set()
        self._frame: Optional[RenderResult] = None
        self._manifest: List[dict] = empty_manifest(self.tiles)

    def _touched_tiles(self, scene: Scene, camera: Camera) -> Set[int]:
        touched: Set[int] = set()
        for seg in scene.segments:
            mins, maxs = segment_aabb(seg)
            if not camera.aabb_visible(mins, maxs):
                continue
            x0, _y0, x1, _y1 = camera.screen_bbox(
                mins, maxs, self.width, self.height
            )
            for i, (tx0, tx1) in enumerate(self.tiles):
                if tx0 < x1 and x0 < tx1:
                    touched.add(i)
        return touched

    def render_frame(self, scene: Scene, camera: Camera) -> FrameResult:
        sv, cv = scene.version, camera.version
        new_touched = self._touched_tiles(scene, camera)

        if self._frame is None:
            dirty = set(range(len(self.tiles)))
        elif sv == self._last_scene_version and cv == self._last_camera_version:
            dirty = set()  # no version change: reuse everything
        else:
            # Recompute tiles that may gain content (new projections) or
            # must erase stale content (previously touched tiles).
            dirty = self._last_touched | new_touched

        if self._frame is None:
            self._frame = RenderResult(
                color=np.empty((self.height, self.width, 4), dtype=np.uint8),
                depth=np.full((self.height, self.width), np.inf, dtype=np.float32),
                ids=np.full((self.height, self.width), -1, dtype=np.int32),
            )
            self._frame.color[:, :] = np.array((12, 12, 20, 255), dtype=np.uint8)

        for ti in sorted(dirty):
            x0, x1 = self.tiles[ti]
            # Clear the tile region, then re-render only that band.
            self._frame.color[:, x0:x1] = np.array((12, 12, 20, 255), dtype=np.uint8)
            self._frame.depth[:, x0:x1] = np.inf
            self._frame.ids[:, x0:x1] = -1
            render(
                scene,
                camera,
                self.width,
                self.height,
                x_range=(x0, x1),
                into=self._frame,
            )
            # Manifest records the final committed pixels of this tile.
            self._manifest[ti] = recount_tile(self._frame.ids, ti, x0, x1)

        self.generation += 1
        self._last_scene_version = sv
        self._last_camera_version = cv
        self._last_touched = new_touched

        diffs = cross_check(self._manifest, self._frame.ids, self.tiles)
        return FrameResult(
            generation=self.generation,
            scene_version=sv,
            camera_version=cv,
            color=self._frame.color,
            depth=self._frame.depth,
            ids=self._frame.ids,
            manifest=[dict(m) for m in self._manifest],
            dirty_tiles=sorted(dirty),
            reused_tiles=sorted(set(range(len(self.tiles))) - dirty),
            manifest_diffs=diffs,
        )

    def cache_meta(self) -> dict:
        return {
            "generation": self.generation,
            "scene_version": self._last_scene_version,
            "camera_version": self._last_camera_version,
            "width": self.width,
            "height": self.height,
            "tile_width": self.tile_width,
        }

    def save_cache(self, cache_dir: str) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        payload = dict(self.cache_meta())
        payload["tiles"] = self._manifest
        path = os.path.join(cache_dir, CACHE_FILE)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        return path


def load_cache(cache_dir: str, expected_meta: dict) -> Optional[dict]:
    """Load a tile cache only if every header field matches the expected
    snapshot generation and versions. Any mismatch discards the cache
    entirely (returns None); stale offsets are never consulted."""
    path = os.path.join(cache_dir, CACHE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    for key, value in expected_meta.items():
        if payload.get(key) != value:
            return None
    return payload
