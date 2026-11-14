"""Deterministic procedural triangle meshes.

Every mesh stores flat float64 triangle positions/colors and per-vertex
outward normals.  Generation order is fixed and does not depend on hashing or
on any unordered iteration, so identical inputs always produce bit-identical
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Mesh:
    name: str
    positions: np.ndarray  # (N, 3, 3): triangle x vertex x xyz
    normals: np.ndarray  # (N, 3, 3): vertex normals, outward facing

    @property
    def triangle_count(self) -> int:
        return int(self.positions.shape[0])


def _push_triangle(
    pos_list: list,
    nrm_list: list,
    points: tuple,
    normal: np.ndarray,
) -> None:
    pos_list.append(np.asarray(points, dtype=np.float64))
    nrm_list.append(
        np.repeat(np.asarray(normal, dtype=np.float64)[None, :], 3, axis=0)
    )


def box(name: str, center: np.ndarray, size: np.ndarray) -> Mesh:
    """Axis-aligned box with 12 outward-wound triangles."""
    center = np.asarray(center, dtype=np.float64)
    hx, hy, hz = (np.asarray(size, dtype=np.float64) * 0.5).tolist()
    c = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    ) + center
    faces = [
        (0, 3, 2, 0, 2, 1, (0, 0, -1)),  # -Z
        (4, 5, 6, 4, 6, 7, (0, 0, 1)),  # +Z
        (0, 1, 5, 0, 5, 4, (0, -1, 0)),  # -Y
        (2, 3, 7, 2, 7, 6, (0, 1, 0)),  # +Y
        (0, 4, 7, 0, 7, 3, (-1, 0, 0)),  # -X
        (1, 2, 6, 1, 6, 5, (1, 0, 0)),  # +X
    ]
    pos_list: list = []
    nrm_list: list = []
    for a, b, cc, d, e, f, n in faces:
        normal = np.asarray(n, dtype=np.float64)
        _push_triangle(pos_list, nrm_list, (c[a], c[b], c[cc]), normal)
        _push_triangle(pos_list, nrm_list, (c[d], c[e], c[f]), normal)
    return Mesh(name, np.asarray(pos_list), np.asarray(nrm_list))


def plane(
    name: str,
    center: np.ndarray,
    width: float,
    depth: float,
    subdivisions: int = 1,
) -> Mesh:
    """Horizontal XZ plane with outward +Y normals, wound CCW when viewed +Y."""
    if subdivisions < 1:
        raise ValueError("plane subdivisions must be >= 1")
    center = np.asarray(center, dtype=np.float64)
    step_x = width / subdivisions
    step_z = depth / subdivisions
    x0 = center[0] - width * 0.5
    z0 = center[2] - depth * 0.5
    y = center[1]
    up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    pos_list: list = []
    nrm_list: list = []
    for iz in range(subdivisions):
        for ix in range(subdivisions):
            x = x0 + ix * step_x
            z = z0 + iz * step_z
            p00 = np.array((x, y, z), dtype=np.float64)
            p10 = np.array((x + step_x, y, z), dtype=np.float64)
            p11 = np.array((x + step_x, y, z + step_z), dtype=np.float64)
            p01 = np.array((x, y, z + step_z), dtype=np.float64)
            _push_triangle(pos_list, nrm_list, (p00, p10, p11), up)
            _push_triangle(pos_list, nrm_list, (p00, p11, p01), up)
    return Mesh(name, np.asarray(pos_list), np.asarray(nrm_list))


def sphere(
    name: str,
    center: np.ndarray,
    radius: float,
    segments: int = 12,
    rings: int = 8,
) -> Mesh:
    """UV sphere generated with a fixed ring/segment traversal order."""
    if segments < 3 or rings < 2 or radius <= 0.0:
        raise ValueError("invalid sphere parameters")
    center = np.asarray(center, dtype=np.float64)
    positions: list = []
    normals: list = []

    def grid_vertex(i_ring: int, i_seg: int) -> np.ndarray:
        theta = math.pi * i_ring / rings
        phi = 2.0 * math.pi * i_seg / segments
        normal = np.array(
            (
                math.sin(theta) * math.cos(phi),
                math.cos(theta),
                math.sin(theta) * math.sin(phi),
            ),
            dtype=np.float64,
        )
        return normal, center + radius * normal

    for i in range(rings):
        for j in range(segments):
            j_next = (j + 1) % segments
            n00, p00 = grid_vertex(i, j)
            n01, p01 = grid_vertex(i, j_next)
            n11, p11 = grid_vertex(i + 1, j_next)
            n10, p10 = grid_vertex(i + 1, j)
            if i > 0:
                _push_triangle(positions, normals, (p00, p01, p11), n00)
                normals[-1][0] = n00
                normals[-1][1] = n01
                normals[-1][2] = n11
            if i < rings - 1:
                _push_triangle(positions, normals, (p00, p11, p10), n00)
                normals[-1][0] = n00
                normals[-1][1] = n11
                normals[-1][2] = n10
    return Mesh(name, np.asarray(positions), np.asarray(normals))


def merge_meshes(meshes) -> Mesh:
    names = [m.name for m in meshes]
    positions = np.concatenate([m.positions for m in meshes], axis=0)
    normals = np.concatenate([m.normals for m in meshes], axis=0)
    return Mesh("merged:" + ",".join(names), positions, normals)

