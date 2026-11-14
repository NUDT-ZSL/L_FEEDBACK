"""LOD chunk mesh generation with crack-free edge stitching.

Each chunk mesh is a regular grid of (cells/2^lod + 1)^2 vertices whose
world positions coincide exactly with the vertices of the finer LODs, so
same-LOD neighbours share identical border vertex positions.  Between
different LODs, T-junctions are removed by a stitch strip generated from
the two edges' vertex indices (`build_stitch_strip`): every fine-edge
vertex is referenced, which guarantees a watertight transition.

Triangulation uses a fixed diagonal (SW->NE) which pairs with the normal
map stencil in `normalmap`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heightfield import HeightField


@dataclass
class ChunkMesh:
    cx: int
    cy: int
    lod: int
    vertices: np.ndarray      # (n*n, 3) float64, row-major (y rows, x cols)
    indices: np.ndarray       # (m, 3) uint32, CCW from +Z
    grid_size: int            # vertices per side (n)
    spacing: float            # world spacing between adjacent grid vertices
    edges: dict               # name -> ordered vertex indices along the edge

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            cx=np.int64(self.cx), cy=np.int64(self.cy), lod=np.int64(self.lod),
            vertices=self.vertices, indices=self.indices,
            grid_size=np.int64(self.grid_size), spacing=np.float64(self.spacing),
            edge_north=self.edges["north"], edge_south=self.edges["south"],
            edge_west=self.edges["west"], edge_east=self.edges["east"],
        )

    @staticmethod
    def load(path) -> "ChunkMesh":
        d = np.load(path)
        return ChunkMesh(
            cx=int(d["cx"]), cy=int(d["cy"]), lod=int(d["lod"]),
            vertices=d["vertices"], indices=d["indices"],
            grid_size=int(d["grid_size"]), spacing=float(d["spacing"]),
            edges={"north": d["edge_north"], "south": d["edge_south"],
                   "west": d["edge_west"], "east": d["edge_east"]},
        )


def build_chunk_mesh(field: HeightField, cx: int, cy: int, lod: int) -> ChunkMesh:
    step = 1 << lod
    xs, ys, heights = field.chunk_grid(cx, cy, step)
    n = heights.shape[0]
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.stack([gx.ravel(), gy.ravel(), heights.ravel()], axis=1)

    def vid(i: int, j: int) -> int:
        return j * n + i

    quads = []
    for j in range(n - 1):
        for i in range(n - 1):
            quads.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            quads.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    indices = np.array(quads, dtype=np.uint32)
    edges = {
        "south": np.array([vid(i, 0) for i in range(n)], dtype=np.uint32),
        "north": np.array([vid(i, n - 1) for i in range(n)], dtype=np.uint32),
        "west": np.array([vid(0, j) for j in range(n)], dtype=np.uint32),
        "east": np.array([vid(n - 1, j) for j in range(n)], dtype=np.uint32),
    }
    return ChunkMesh(cx=cx, cy=cy, lod=lod, vertices=vertices, indices=indices,
                     grid_size=n, spacing=field.spacing * step, edges=edges)


def build_stitch_strip(fine_edge: np.ndarray, coarse_edge: np.ndarray) -> np.ndarray:
    """Triangles bridging a fine edge and a coarser edge of the same segment.

    Both index arrays run in the same direction along the shared segment and
    their endpoints coincide.  The fine edge must have
    ``(len(coarse_edge) - 1) * ratio + 1`` vertices with ``ratio`` a positive
    integer (typically 2**dlod).  Every fine vertex is referenced exactly
    once as a strip vertex, eliminating T-junction cracks.

    Returns an (m, 3) uint32 array.  Winding assumes the fine mesh is on the
    "left" of the edge direction and the coarse mesh on the "right"; pass
    reversed edges to flip orientation.
    """
    fine_edge = np.asarray(fine_edge, dtype=np.uint32)
    coarse_edge = np.asarray(coarse_edge, dtype=np.uint32)
    n_coarse = len(coarse_edge)
    if n_coarse < 2 or len(fine_edge) < 2:
        raise ValueError("edges need at least two vertices")
    ratio, rem = divmod(len(fine_edge) - 1, n_coarse - 1)
    if rem or ratio < 1:
        raise ValueError("fine edge length incompatible with coarse edge")
    tris = []
    fi = 0
    for ci in range(n_coarse - 1):
        c0 = int(coarse_edge[ci])
        c1 = int(coarse_edge[ci + 1])
        for _ in range(ratio):
            tris.append((c0, int(fine_edge[fi]), int(fine_edge[fi + 1])))
            fi += 1
        tris.append((c0, int(fine_edge[fi]), c1))
    return np.array(tris, dtype=np.uint32)


def stitch_meshes(fine: ChunkMesh, coarse: ChunkMesh, edge: str) -> np.ndarray:
    """Build the stitch strip between two adjacent chunks of different LOD.

    ``edge`` is the side of the *fine* chunk that touches the coarse chunk.
    The coarse chunk must be the neighbour across that edge.
    """
    fine_edge = fine.edges[edge]
    opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
    coarse_edge = coarse.edges[opposite[edge]]
    return build_stitch_strip(fine_edge, coarse_edge)


def select_lod_for_distance(distance: float, chunk_world_size: float, max_lod: int,
                            screen_error_threshold_px: float = 2.0,
                            viewport_height_px: int = 1080,
                            fov_y_deg: float = 60.0) -> int:
    """Pick the coarsest LOD whose estimated screen-space error fits the budget.

    Geometric error model: lod L subsamples the chunk grid by 2^L, so the
    worst-case height deviation is bounded by ``chunk_world_size * 2^L / 64``
    (a conservative slope-based heuristic).  Screen error projects that to
    pixels with a simple pinhole model.
    """
    import math

    distance = max(distance, 1e-6)
    proj = viewport_height_px / (2.0 * distance * math.tan(math.radians(fov_y_deg) * 0.5))
    # coarsest LOD whose projected geometric error still fits the budget
    for lod in range(max_lod, -1, -1):
        geometric_error = chunk_world_size * (1 << lod) / 64.0
        if geometric_error * proj <= screen_error_threshold_px:
            return lod
    return 0
