"""Multi-level LOD terrain grids, low-pass decimation and binary meshes.

Each level halves the grid spacing. Down-sampling applies a separable
tent (1-2-1) low-pass filter before 2x decimation, which attenuates the
frequencies that would otherwise alias into the coarser grid. End points
are preserved exactly so all levels share the same outer footprint and
texture-coordinate domain.

Meshes are written as one explicit little-endian binary file per level
containing a vertex block followed by an index block, plus JSON metadata
with absolute byte offsets so a render pipeline can load without parsing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .memory import MemoryBudget
from .normal import TileStat

VERTEX_DTYPE = np.dtype(
    [
        ("position", "<f4", 3),
        ("normal", "<f4", 3),
        ("uv", "<f4", 2),
    ]
)
INDEX_DTYPE = np.dtype("<u4")
VERTEX_STRIDE = VERTEX_DTYPE.itemsize  # 32 bytes


@dataclass
class BoundingBox:
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    def to_dict(self) -> dict:
        return {"min": list(self.min_xyz), "max": list(self.max_xyz)}


@dataclass
class LodLevel:
    level: int
    resolution: int
    spacing: int
    vertex_count: int
    triangle_count: int
    index_count: int
    binary_path: str
    vertex_offset: int
    index_offset: int
    bbox: BoundingBox
    tile_stats: list[TileStat] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "resolution": self.resolution,
            "spacing_texels": self.spacing,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "index_count": self.index_count,
            "binary_path": self.binary_path,
            "vertex_offset_bytes": self.vertex_offset,
            "index_offset_bytes": self.index_offset,
            "vertex_stride_bytes": VERTEX_STRIDE,
            "index_type": "uint32",
            "endianness": "little",
            "bounding_box": self.bbox.to_dict(),
            "tile_stats": [stat.to_dict() for stat in self.tile_stats],
        }


def lod_resolution(base_resolution: int, level: int) -> int:
    """Resolution for a level, requiring an odd result with shared corners."""
    if level < 0:
        raise ValueError("LOD level must be >= 0")
    factor = 1 << level
    if (base_resolution - 1) % factor != 0:
        raise ValueError(
            f"level {level} requires (resolution-1) divisible by {factor}"
        )
    return (base_resolution - 1) // factor + 1


def _tent_1d(values: np.ndarray, axis: int) -> np.ndarray:
    """Apply a clamped 1-2-1 tent filter along one axis, same shape out."""
    moved = np.moveaxis(values, axis, 0)
    up = np.empty_like(moved)
    down = np.empty_like(moved)
    up[:-1] = moved[1:]
    up[-1] = moved[-1]
    down[1:] = moved[:-1]
    down[0] = moved[0]
    filtered = (
        down.astype(np.float32)
        + np.float32(2.0) * moved.astype(np.float32)
        + up.astype(np.float32)
    ) * np.float32(0.25)
    return np.moveaxis(filtered, 0, axis)


def downsample_heightfield(
    src: np.ndarray,
    budget: MemoryBudget,
    out_path: str | None = None,
    block_rows: int = 64,
) -> np.ndarray:
    """Low-pass filter then 2x decimate an odd grid into the next LOD.

    Returns a memmap when ``out_path`` is provided, otherwise an in-memory
    array used by tests. Corner samples are copied exactly.
    """
    n = src.shape[0]
    if (n - 1) % 2 != 0:
        raise ValueError("source resolution must be odd")
    coarse = (n - 1) // 2 + 1
    block = min(block_rows, coarse)
    working_bytes = (block + 1) * n * 4 * 4
    budget.acquire(working_bytes)
    try:
        if out_path is None:
            dst = np.empty((coarse, coarse), dtype=np.float32)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            dst = np.memmap(out_path, dtype="<f4", mode="w+", shape=(coarse, coarse))
        for dst_r0 in range(0, coarse, block):
            dst_r1 = min(dst_r0 + block, coarse)
            # Include one clamped source-row halo so the vertical tent
            # filter sees the neighbour across block boundaries.
            src_r0 = max(dst_r0 * 2 - 1, 0)
            src_r1 = min(dst_r1 * 2 + 2, n)
            band = np.asarray(src[src_r0:src_r1, :], dtype=np.float32)
            vertical = _tent_1d(band, axis=0)
            full = _tent_1d(vertical, axis=1)
            local_rows = np.arange(dst_r0, dst_r1) * 2 - src_r0
            cols = np.arange(0, n, 2, dtype=np.int64)
            sub = full[np.ix_(local_rows, cols)]
            dst[dst_r0:dst_r1, :] = sub
        # Preserve the four corners exactly across levels.
        dst[0, 0] = src[0, 0]
        dst[0, -1] = src[0, -1]
        dst[-1, 0] = src[-1, 0]
        dst[-1, -1] = src[-1, -1]
        if hasattr(dst, "flush"):
            dst.flush()
    finally:
        budget.release(working_bytes)
    return dst


def tangent_normals_from_height(
    height: np.ndarray,
    spacing_world: float,
) -> np.ndarray:
    """Tangent-space unit normals at every vertex for a given world spacing.

    Uses central differences with edge clamping. The same convention as the
    normal map is used (X along U, Y along V downward, Z toward the viewer),
    but gradients are divided by world spacing so they represent true slopes.
    """
    h = height.astype(np.float32, copy=False)
    left = np.empty_like(h)
    right = np.empty_like(h)
    up = np.empty_like(h)
    down = np.empty_like(h)
    left[:, 1:] = h[:, :-1]
    left[:, 0] = h[:, 0]
    right[:, :-1] = h[:, 1:]
    right[:, -1] = h[:, -1]
    up[1:, :] = h[:-1, :]
    up[0, :] = h[0, :]
    down[:-1, :] = h[1:, :]
    down[-1, :] = h[-1, :]
    dx = (right - left) * np.float32(0.5 / spacing_world)
    dy = (down - up) * np.float32(0.5 / spacing_world)
    normals = np.empty((h.shape[0], h.shape[1], 3), dtype=np.float32)
    normals[:, :, 0] = -dx
    normals[:, :, 1] = -dy
    normals[:, :, 2] = np.float32(1.0)
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    np.divide(normals, lengths, out=normals)
    return normals


def _write_mesh_band(
    handle,
    index_handle,
    height: np.ndarray,
    normals: np.ndarray,
    dst_r0: int,
    dst_r1: int,
    coarse: int,
    world_size: float,
    spacing_world: float,
) -> BoundingBox:
    """Append one row band of vertices and its triangle indices."""
    band_rows = dst_r1 - dst_r0
    u = (np.arange(coarse, dtype=np.float32) / np.float32(coarse - 1))
    v_band = (
        np.arange(dst_r0, dst_r1, dtype=np.float32) / np.float32(coarse - 1)
    )
    x = u * np.float32(world_size)
    y = v_band * np.float32(world_size)
    xx, yy = np.meshgrid(x, y, copy=True)
    vertices = np.empty(band_rows * coarse, dtype=VERTEX_DTYPE)
    vertices["position"][:, 0] = xx.reshape(-1)
    vertices["position"][:, 1] = yy.reshape(-1)
    vertices["position"][:, 2] = height.reshape(-1)
    vertices["normal"] = normals.reshape(-1, 3)
    uu, vv = np.meshgrid(u, v_band, copy=True)
    vertices["uv"][:, 0] = uu.reshape(-1)
    vertices["uv"][:, 1] = vv.reshape(-1)
    handle.write(vertices.tobytes())

    mins = [
        float(vertices["position"][:, 0].min()),
        float(vertices["position"][:, 1].min()),
        float(vertices["position"][:, 2].min()),
    ]
    maxs = [
        float(vertices["position"][:, 0].max()),
        float(vertices["position"][:, 1].max()),
        float(vertices["position"][:, 2].max()),
    ]

    if dst_r1 > 1:
        top = max(dst_r0 - 1, 0)
        quad_r = np.arange(top, dst_r1 - 1, dtype=np.uint32)
        quad_c = np.arange(coarse - 1, dtype=np.uint32)
        rr, cc = np.meshgrid(quad_r, quad_c, indexing="ij")
        v00 = rr * coarse + cc
        v01 = v00 + np.uint32(1)
        v10 = v00 + np.uint32(coarse)
        v11 = v10 + np.uint32(1)
        # Winding is counter-clockwise when viewed from +Z.
        indices = np.empty((rr.size, 6), dtype=INDEX_DTYPE)
        indices[:, 0] = v00.reshape(-1)
        indices[:, 1] = v10.reshape(-1)
        indices[:, 2] = v01.reshape(-1)
        indices[:, 3] = v01.reshape(-1)
        indices[:, 4] = v10.reshape(-1)
        indices[:, 5] = v11.reshape(-1)
        index_handle.write(indices.tobytes())

    return BoundingBox(tuple(mins), tuple(maxs))


def _merge_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    mins = [min(box.min_xyz[i] for box in boxes) for i in range(3)]
    maxs = [max(box.max_xyz[i] for box in boxes) for i in range(3)]
    return BoundingBox(tuple(mins), tuple(maxs))


def write_lod_mesh(
    path: str,
    height: np.ndarray,
    level: int,
    spacing_texels: int,
    world_size: float,
    budget: MemoryBudget,
    relative_path: str,
    block_rows: int = 64,
) -> LodLevel:
    """Stream a LOD mesh to disk in row bands under the memory budget."""
    coarse = height.shape[0]
    spacing_world = world_size * spacing_texels / (coarse - 1)
    vertex_offset = 0
    index_offset = coarse * coarse * VERTEX_STRIDE
    band = min(block_rows, coarse)
    working_bytes = band * coarse * (VERTEX_STRIDE + 12 + 8)
    budget.acquire(working_bytes)
    stats: list[TileStat] = []
    boxes: list[BoundingBox] = []
    try:
        # Reserve the full vertex region, then stream indices from a second
        # handle positioned at the index block, keeping both offsets exact.
        with open(path, "wb") as handle, open(path, "r+b") as index_handle:
            handle.seek(index_offset - 1)
            handle.write(b"\x00")
            handle.seek(vertex_offset)
            index_handle.seek(index_offset)
            for dst_r0 in range(0, coarse, band):
                start = time.perf_counter()
                dst_r1 = min(dst_r0 + band, coarse)
                band_height = np.asarray(height[dst_r0:dst_r1, :], dtype=np.float32)
                # Recompute normals for the band with one clamped halo row.
                r0 = max(dst_r0 - 1, 0)
                r1 = min(dst_r1 + 1, coarse)
                halo = np.asarray(height[r0:r1, :], dtype=np.float32)
                halo_normals = tangent_normals_from_height(halo, spacing_world)
                local0 = dst_r0 - r0
                local1 = dst_r1 - r0
                band_normals = halo_normals[local0:local1, :, :]
                box = _write_mesh_band(
                    handle,
                    index_handle,
                    band_height,
                    band_normals,
                    dst_r0,
                    dst_r1,
                    coarse,
                    world_size,
                    spacing_world,
                )
                boxes.append(box)
                stats.append(
                    TileStat(
                        name=f"lod{level}",
                        row=dst_r0,
                        col=0,
                        rows=dst_r1 - dst_r0,
                        cols=coarse,
                        elapsed_ms=(time.perf_counter() - start) * 1000.0,
                    )
                )
    finally:
        budget.release(working_bytes)

    vertex_count = coarse * coarse
    triangle_count = (coarse - 1) * (coarse - 1) * 2
    level_info = LodLevel(
        level=level,
        resolution=coarse,
        spacing=spacing_texels,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        index_count=triangle_count * 3,
        binary_path=relative_path,
        vertex_offset=vertex_offset,
        index_offset=index_offset,
        bbox=_merge_boxes(boxes),
        tile_stats=stats,
    )
    return level_info


def build_lod_chain(
    base_grid: np.ndarray,
    levels: list[int],
    world_size: float,
    budget: MemoryBudget,
    temp_dir: str,
    mesh_dir: str,
    relative_mesh_dir: str,
    block_rows: int = 64,
) -> tuple[list[LodLevel], list[str]]:
    """Build every requested LOD and return metadata plus temp memmap paths."""
    if not levels:
        raise ValueError("at least one LOD level is required")
    ordered = sorted(set(int(level) for level in levels))
    if ordered[0] < 0:
        raise ValueError("LOD levels must be non-negative")

    os.makedirs(mesh_dir, exist_ok=True)
    current = base_grid
    current_level = 0
    current_spacing = 1
    infos: list[LodLevel] = []
    temp_grids: list[str] = []

    for level in ordered:
        while current_level < level:
            next_path = os.path.join(temp_dir, f"heightfield_lod{current_level + 1}.dat")
            current = downsample_heightfield(
                current, budget, out_path=next_path, block_rows=block_rows
            )
            temp_grids.append(next_path)
            current_level += 1
            current_spacing *= 2
        mesh_name = f"lod{level}.bin"
        info = write_lod_mesh(
            path=os.path.join(mesh_dir, mesh_name),
            height=current,
            level=level,
            spacing_texels=current_spacing,
            world_size=world_size,
            budget=budget,
            relative_path=os.path.join(relative_mesh_dir, mesh_name),
            block_rows=block_rows,
        )
        infos.append(info)
    return infos, temp_grids
