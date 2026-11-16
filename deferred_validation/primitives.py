"""Procedural triangle meshes and normal-map generation for self-contained scenes."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _mesh(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
        "uvs": np.asarray(uvs, dtype=np.float32),
        "indices": np.asarray(indices, dtype=np.uint32),
    }


def _quad(a, b, c, d, normal, uv_scale: float = 1.0):
    positions = np.asarray([a, b, c, d], dtype=np.float32)
    normals = np.tile(np.asarray(normal, dtype=np.float32), (4, 1))
    uvs = np.asarray(
        [[0.0, 0.0], [uv_scale, 0.0], [uv_scale, uv_scale], [0.0, uv_scale]], dtype=np.float32
    )
    indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return positions, normals, uvs, indices


def make_plane(size: float = 1.0, uv_scale: float = 2.0) -> Dict[str, np.ndarray]:
    s = float(size)
    positions = np.asarray(
        [[-s, 0.0, -s], [-s, 0.0, s], [s, 0.0, s], [s, 0.0, -s]], dtype=np.float32
    )
    normals = np.tile(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (4, 1))
    uvs = np.asarray(
        [[0.0, 0.0], [0.0, uv_scale], [uv_scale, uv_scale], [uv_scale, 0.0]], dtype=np.float32
    )
    indices = np.asarray([0, 2, 1, 0, 3, 2], dtype=np.uint32)
    return _mesh(positions, normals, uvs, indices)


def make_box(size: float = 1.0) -> Dict[str, np.ndarray]:
    s = float(size) * 0.5
    faces = [
        ((s, -s, -s), (s, s, -s), (s, s, s), (s, -s, s), (1, 0, 0)),
        ((-s, -s, s), (-s, s, s), (-s, s, -s), (-s, -s, -s), (-1, 0, 0)),
        ((-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s), (0, 1, 0)),
        ((-s, -s, s), (-s, -s, -s), (s, -s, -s), (s, -s, s), (0, -1, 0)),
        ((-s, -s, -s), (-s, s, -s), (s, s, -s), (s, -s, -s), (0, 0, -1)),
        ((s, -s, s), (s, s, s), (-s, s, s), (-s, -s, s), (0, 0, 1)),
    ]
    positions = []
    normals = []
    uvs = []
    indices = []
    for base, (a, b, c, d, normal) in enumerate(faces):
        p, n, uv, idx = _quad(a, b, c, d, normal, 1.0)
        positions.append(p)
        normals.append(n)
        uvs.append(uv)
        indices.append(idx + np.uint32(base * 4))
    return _mesh(
        np.concatenate(positions),
        np.concatenate(normals),
        np.concatenate(uvs),
        np.concatenate(indices),
    )


def make_sphere(radius: float = 1.0, rings: int = 18, segments: int = 28) -> Dict[str, np.ndarray]:
    r = float(radius)
    positions = []
    normals = []
    uvs = []
    for iy in range(rings + 1):
        v = iy / rings
        theta = v * math.pi
        for ix in range(segments + 1):
            u = ix / segments
            phi = u * math.tau
            nx = math.sin(theta) * math.cos(phi)
            ny = math.cos(theta)
            nz = math.sin(theta) * math.sin(phi)
            positions.append((r * nx, r * ny, r * nz))
            normals.append((nx, ny, nz))
            uvs.append((u, 1.0 - v))
    indices = []
    row = segments + 1
    for iy in range(rings):
        for ix in range(segments):
            a = iy * row + ix
            b = a + 1
            c = a + row
            d = c + 1
            indices.extend((a, c, b, b, c, d))
    return _mesh(
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def make_tiling_normal_map(width: int = 32, height: int = 32, bump: float = 0.18) -> np.ndarray:
    """Return an RGBA8 tangent-space normal map with two sine wave ridges."""
    xs = np.linspace(0.0, 1.0, width, endpoint=False)
    ys = np.linspace(0.0, 1.0, height, endpoint=False)
    u, v = np.meshgrid(xs, ys)
    height_field = np.sin(u * math.tau * 2.0) * np.cos(v * math.tau * 3.0)
    du = np.cos(u * math.tau * 2.0) * math.tau * 2.0 * np.cos(v * math.tau * 3.0)
    dv = -np.sin(u * math.tau * 2.0) * np.sin(v * math.tau * 3.0) * math.tau * 3.0
    nx = -du * bump
    ny = -dv * bump
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normals = np.stack([nx / length, ny / length, nz / length], axis=-1)
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(normals * 0.5 + 0.5, 0.0, 1.0)
    del height_field
    return np.round(rgba * 255.0).astype(np.uint8)


def mesh_to_json(mesh: Dict[str, np.ndarray], precision: int = 6) -> Dict[str, list]:
    return {
        "positions": np.round(mesh["positions"], precision).reshape(-1).tolist(),
        "normals": np.round(mesh["normals"], precision).reshape(-1).tolist(),
        "uvs": np.round(mesh["uvs"], precision).reshape(-1).tolist(),
        "indices": mesh["indices"].astype(int).tolist(),
    }
