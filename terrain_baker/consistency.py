"""Normal map / mesh consistency validation.

Geometric vertex normals are recomputed from the mesh triangles by
area-weighted face-normal accumulation.  At chunk borders, virtual "clamp
quads" (ghost vertices replicating the edge heights, per the shared
clamping rule) complete the stencil so the result matches the normal map
baker exactly.  Each mesh vertex is then compared against the normal map
texel at the same grid position; angular deviations above the configured
threshold mark the chunk as erroneous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lod_mesh import ChunkMesh
from .normalmap import decode_rgb8


def _accumulate(vertices: np.ndarray, tris, normals: np.ndarray) -> None:
    for a, b, c in tris:
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        n = np.cross(pb - pa, pc - pa)
        normals[a] += n
        normals[b] += n
        normals[c] += n


def geometric_vertex_normals(mesh: ChunkMesh) -> np.ndarray:
    """Area-weighted vertex normals, completed with clamp ghost quads."""
    n = mesh.grid_size
    spacing = mesh.spacing
    verts = mesh.vertices
    normals = np.zeros_like(verts)
    _accumulate(verts, mesh.indices, normals)

    heights = verts[:, 2].reshape(n, n)

    def vid(i: int, j: int) -> int:
        return j * n + i

    ghost_verts = []
    ghost_tris = []

    def corner_ref(x: int, y: int) -> int:
        if 0 <= x < n and 0 <= y < n:
            return vid(x, y)
        cx = min(max(x, 0), n - 1)
        cy = min(max(y, 0), n - 1)
        ghost_verts.append(np.array([x * spacing, y * spacing, heights[cy, cx]]))
        return len(verts) + len(ghost_verts) - 1

    def ghost_quad(x0: int, y0: int, x1: int, y1: int) -> None:
        a = corner_ref(x0, y0)
        b = corner_ref(x1, y0)
        c = corner_ref(x1, y1)
        d = corner_ref(x0, y1)
        ghost_tris.append((a, b, c))
        ghost_tris.append((a, c, d))

    for i in range(n - 1):
        ghost_quad(i, -1, i + 1, 0)        # south
        ghost_quad(i, n - 1, i + 1, n)     # north
    for j in range(n - 1):
        ghost_quad(-1, j, 0, j + 1)        # west
        ghost_quad(n - 1, j, n, j + 1)     # east
    ghost_quad(-1, -1, 0, 0)
    ghost_quad(n - 1, -1, n, 0)
    ghost_quad(n - 1, n - 1, n, n)
    ghost_quad(-1, n - 1, 0, n)

    if ghost_verts:
        # ghost vertices are built in the chunk's local frame; shift to the
        # mesh's world frame before accumulating.  Accumulate into an
        # extended buffer since ghost triangles reference ghost indices.
        origin = verts[0].copy()
        origin[2] = 0.0
        ghost_arr = np.array(ghost_verts) + origin
        all_verts = np.concatenate([verts, ghost_arr], axis=0)
        extended = np.zeros((len(all_verts), 3), dtype=np.float64)
        _accumulate(all_verts, ghost_tris, extended)
        normals += extended[:len(verts)]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths


@dataclass
class ConsistencyResult:
    max_angle_deg: float
    mean_angle_deg: float
    bad_vertices: int
    threshold_deg: float

    @property
    def ok(self) -> bool:
        return self.bad_vertices == 0


def validate_chunk(mesh: ChunkMesh, normal_rgb: np.ndarray,
                   threshold_deg: float) -> ConsistencyResult:
    """Compare mesh face-normal-derived vertex normals against the normal map.

    normal_rgb: (n, n, 3) uint8 texels aligned with the mesh grid (row = y).
    """
    n = mesh.grid_size
    if normal_rgb.shape[0] != n or normal_rgb.shape[1] != n:
        raise ValueError(
            f"normal map {normal_rgb.shape[:2]} does not match mesh grid {n}")
    geo = geometric_vertex_normals(mesh).reshape(n, n, 3)
    sampled = decode_rgb8(normal_rgb)
    dot = np.clip(np.sum(geo * sampled, axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    bad = int(np.count_nonzero(angles > threshold_deg))
    return ConsistencyResult(
        max_angle_deg=float(angles.max()),
        mean_angle_deg=float(angles.mean()),
        bad_vertices=bad,
        threshold_deg=threshold_deg,
    )
