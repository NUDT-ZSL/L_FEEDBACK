from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _hash_u32(values: np.ndarray, seed: int, salt: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=False)
    x ^= np.uint64((seed * 0x9E3779B1 + salt * 0x85EBCA77) & 0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x *= np.uint64(0x7FEB352D)
    x ^= x >> np.uint64(15)
    x *= np.uint64(0x846CA68B)
    x ^= x >> np.uint64(16)
    return (x & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0


@dataclass(frozen=True)
class GrainTexture:
    normals: np.ndarray
    active: np.ndarray
    width: int
    height: int
    density: float
    seed: int
    mode: str
    digest: str
    generation_ms: float

    def sample(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.mod(uv[..., 0], 1.0)
        v = np.mod(uv[..., 1], 1.0)
        x = np.minimum(self.width - 1, (u * self.width).astype(np.int64))
        y = np.minimum(self.height - 1, (v * self.height).astype(np.int64))
        return self.normals[y, x], self.active[y, x]


def build_grain_texture(
    density: float,
    seed: int,
    mode: str,
    width: int = 1024,
    height: int = 1024,
    cell_size: int = 2,
) -> GrainTexture:
    started = time.perf_counter()
    width = max(512, int(width))
    height = max(512, int(height))
    cell_size = max(1, int(cell_size))
    cx = int(math.ceil(width / cell_size))
    cy = int(math.ceil(height / cell_size))
    yy, xx = np.meshgrid(
        np.arange(cy, dtype=np.uint64),
        np.arange(cx, dtype=np.uint64),
        indexing="ij",
    )
    cell = (yy * np.uint64(cx) + xx).astype(np.uint64)
    active_roll = _hash_u32(cell, seed, 1)
    density = min(1.0, max(0.0, float(density)))
    active_cells = active_roll < density * 0.55

    radius_hash = _hash_u32(cell, seed, 2)
    angle_hash = _hash_u32(cell, seed, 3)
    max_tilt = 0.36 if mode == "pearl" else 0.24
    radius = np.sqrt(radius_hash) * max_tilt
    angle = angle_hash * (2.0 * math.pi)
    nx = np.cos(angle) * radius
    ny = np.sin(angle) * radius
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny))
    normals = np.stack((nx, ny, nz), axis=-1)
    normals[~active_cells] = 0.0
    active = active_cells & (density > 0.0)

    full_normals = np.zeros((cy * cell_size, cx * cell_size, 3), dtype=np.float32)
    full_active = np.zeros((cy * cell_size, cx * cell_size), dtype=bool)
    for dy in range(cell_size):
        for dx in range(cell_size):
            full_normals[dy::cell_size, dx::cell_size] = normals.astype(np.float32)
            full_active[dy::cell_size, dx::cell_size] = active

    full_normals = full_normals[:height, :width]
    full_active = full_active[:height, :width]
    digest = hashlib.sha256()
    digest.update(full_normals.tobytes())
    digest.update(full_active.tobytes())
    elapsed = (time.perf_counter() - started) * 1000.0
    return GrainTexture(
        normals=full_normals,
        active=full_active,
        width=width,
        height=height,
        density=density,
        seed=int(seed),
        mode=mode,
        digest=digest.hexdigest(),
        generation_ms=elapsed,
    )


class GrainCache:
    def __init__(self) -> None:
        self._textures: Dict[Tuple[float, int, str], GrainTexture] = {}
        self.generation_count = 0
        self.last_generation_ms = 0.0
        self._lock = threading.Lock()

    def get(self, density: float, seed: int, mode: str) -> GrainTexture:
        key = (round(float(density), 4), int(seed), str(mode))
        with self._lock:
            if key in self._textures:
                return self._textures[key]
            texture = build_grain_texture(*key)
            self._textures[key] = texture
            self.generation_count += 1
            self.last_generation_ms = texture.generation_ms
            if len(self._textures) > 6:
                oldest = next(iter(self._textures))
                if oldest != key:
                    del self._textures[oldest]
            return texture

    def clear(self) -> None:
        with self._lock:
            self._textures.clear()

