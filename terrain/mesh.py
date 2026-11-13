"""Tile mesh construction with T-junction (stitch) elimination.

Every tile is a fixed ``tile_resolution x tile_resolution`` heightfield grid.
When a neighbouring leaf is one level finer, the vertices on its shared edge
are copied verbatim into this tile's boundary cells.  The affected cells are
triangulated as convex fans so both sides of the edge traverse the exact same
world-space points: no cracks and no T-junctions remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig
from .errors import E_MESH_PATCH, PipelineError
from .heightfield import Heightfield
from .quadtree import TileKey, adjacent_leaves

EDGE_DIRECTIONS = ("N", "E", "S", "W")


@dataclass(frozen=True)
class TileMesh:
    key: TileKey
    vertices: np.ndarray
    triangles: np.ndarray
    stitched_edges: tuple[str, ...]
    height_grid: np.ndarray

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


def tile_axis_coordinates(
    key: TileKey, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Regular per-tile sample coordinates ``(x_values, z_values)``."""
    x0, z0, x1, z1 = key.bounds(config.root_min, config.root_size)
    resolution = config.tile_resolution
    values = np.arange(resolution, dtype=np.float64)
    x_values = x0 + (x1 - x0) * values / (resolution - 1)
    z_values = z0 + (z1 - z0) * values / (resolution - 1)
    return x_values, z_values


def edge_world_points(
    x_values: np.ndarray,
    z_values: np.ndarray,
    direction: str,
) -> np.ndarray:
    """``(K, 2)`` world (x, z) points of one tile edge in traversal order."""
    if direction in ("N", "S"):
        z = z_values[0] if direction == "N" else z_values[-1]
        return np.stack(
            [x_values, np.full_like(x_values, z)], axis=1
        )
    x = x_values[0] if direction == "W" else x_values[-1]
    return np.stack(
        [np.full_like(z_values, x), z_values], axis=1
    )


def _opposite(direction: str) -> str:
    return {"N": "S", "S": "N", "W": "E", "E": "W"}[direction]


def build_tile_mesh(
    key: TileKey,
    leaves: frozenset[TileKey],
    heightfield: Heightfield,
    config: PipelineConfig,
    height_grid: np.ndarray | None = None,
) -> TileMesh:
    """Construct one stitched mesh; raises ``PipelineError`` on invalid patch."""
    resolution = config.tile_resolution
    x0, z0, x1, z1 = key.bounds(config.root_min, config.root_size)
    x_values, z_values = tile_axis_coordinates(key, config)
    if height_grid is None:
        xx, zz = np.meshgrid(x_values, z_values)
        height_grid = heightfield.sample_array(xx, zz)

    base_count = resolution * resolution
    vertices = np.zeros((base_count, 3), dtype=np.float64)
    index_of = {}
    for row in range(resolution):
        for col in range(resolution):
            vertex_id = row * resolution + col
            vertices[vertex_id] = (
                x_values[col],
                height_grid[row, col],
                z_values[row],
            )
            index_of[(row, col)] = vertex_id

    edge_points: dict[str, np.ndarray] = {}
    stitched_edges: list[str] = []
    if config.enable_stitching:
        for direction in EDGE_DIRECTIONS:
            neighbours = adjacent_leaves(leaves, key, direction)
            if not neighbours:
                continue
            finest = max(tile.level for tile in neighbours)
            if finest <= key.level:
                continue
            if finest - key.level > 1:
                raise PipelineError(
                    E_MESH_PATCH,
                    f"{key}: non-restricted neighbour level {finest} "
                    f"on {direction}",
                )
            stitched_edges.append(direction)
            edge_points[direction] = _collect_fine_edge_points(
                key, direction, neighbours, leaves, heightfield, config
            )

    extras = (
        np.concatenate(
            [edge_points[d] for d in EDGE_DIRECTIONS if d in edge_points],
            axis=0,
        )
        if edge_points
        else np.zeros((0, 3), dtype=np.float64)
    )
    if extras.size:
        vertices = np.concatenate([vertices, extras], axis=0)

    inserts = {
        direction: _index_edge_points(
            edge_points.get(direction), vertices, x0, z0, x1, z1, direction
        )
        for direction in EDGE_DIRECTIONS
    }

    def base_id(row: int, col: int) -> int:
        return index_of[(row, col)]

    def edge_extras(direction: str, low: float, high: float) -> list[int]:
        ids = [
            vertex_id
            for parameter, vertex_id in inserts[direction].items()
            if low < parameter < high
        ]
        ids.sort(
            key=lambda vertex_id: vertices[
                vertex_id, 0 if direction in ("N", "S") else 2
            ]
        )
        return ids

    def cell_polygon(row: int, col: int) -> list[int]:
        # CCW viewed from +y: SW -> NW -> NE -> SE.
        sw = base_id(row, col)
        nw = base_id(row + 1, col)
        ne = base_id(row + 1, col + 1)
        se = base_id(row, col + 1)
        span = resolution - 1
        polygon = [sw]
        if col == 0:
            polygon.extend(reversed(edge_extras("W", row / span, (row + 1) / span)))
        polygon.append(nw)
        if row == 0:
            polygon.extend(edge_extras("N", col / span, (col + 1) / span))
        polygon.append(ne)
        if col == resolution - 2:
            polygon.extend(edge_extras("E", row / span, (row + 1) / span))
        polygon.append(se)
        if row == resolution - 2:
            polygon.extend(
                reversed(edge_extras("S", col / span, (col + 1) / span))
            )
        return polygon

    triangles: list[tuple[int, int, int]] = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            polygon = cell_polygon(row, col)
            origin = polygon[0]
            for k in range(1, len(polygon) - 1):
                triangles.append((origin, polygon[k], polygon[k + 1]))

    mesh = TileMesh(
        key=key,
        vertices=vertices,
        triangles=np.asarray(triangles, dtype=np.int64),
        stitched_edges=tuple(stitched_edges),
        height_grid=height_grid,
    )
    validate_mesh(mesh)
    return mesh


def _collect_fine_edge_points(
    key, direction, neighbours, leaves, heightfield, config
) -> np.ndarray:
    """Gather the finer neighbours' edge points lying on this tile's edge."""
    x0, z0, x1, z1 = key.bounds(config.root_min, config.root_size)
    is_horizontal = direction in ("N", "S")
    eps = 1e-9
    own_x, own_z = tile_axis_coordinates(key, config)
    own_axis_values = own_x if is_horizontal else own_z
    collected: list[np.ndarray] = []
    for neighbour in neighbours:
        nx, nz = tile_axis_coordinates(neighbour, config)
        fine_edge = edge_world_points(nx, nz, _opposite(direction))
        if is_horizontal:
            mask = (fine_edge[:, 0] > x0 + eps) & (
                fine_edge[:, 0] < x1 - eps
            )
        else:
            mask = (fine_edge[:, 1] > z0 + eps) & (
                fine_edge[:, 1] < z1 - eps
            )
        interior = fine_edge[mask]
        along = interior[:, 0] if is_horizontal else interior[:, 1]
        novel = np.ones(interior.shape[0], dtype=bool)
        for value in own_axis_values:
            novel &= np.abs(along - value) > eps
        interior = interior[novel]
        if interior.size == 0:
            continue
        heights = heightfield.sample_array(
            interior[:, 0], interior[:, 1]
        )
        collected.append(
            np.stack(
                [interior[:, 0], heights, interior[:, 1]], axis=1
            )
        )
    points = np.concatenate(collected, axis=0)
    axis = 0 if is_horizontal else 2
    order = np.argsort(points[:, axis])
    points = points[order]
    coordinate = points[:, axis]
    keep = np.concatenate(([True], np.diff(coordinate) > 1e-9))
    return points[keep]


def _index_edge_points(points, vertices, x0, z0, x1, z1, direction):
    result = {}
    if points is None or points.size == 0:
        return result
    for x, _y, z in points:
        parameter = (
            (x - x0) / (x1 - x0)
            if direction in ("N", "S")
            else (z - z0) / (z1 - z0)
        )
        matches = np.flatnonzero(
            np.isclose(vertices[:, 0], x) & np.isclose(vertices[:, 2], z)
        )
        if matches.size != 1:
            raise PipelineError(
                E_MESH_PATCH,
                f"edge point ({x:.4f},{z:.4f}) matched {matches.size} vertices",
            )
        result[round(float(parameter), 12)] = int(matches[0])
    return result


def validate_mesh(mesh: TileMesh) -> None:
    """Reject half-patched or degenerate meshes before they reach a backend."""
    vertex_count = mesh.vertices.shape[0]
    if mesh.triangles.size == 0:
        raise PipelineError(E_MESH_PATCH, f"{mesh.key}: produced no triangles")
    if np.any(mesh.triangles < 0) or np.any(mesh.triangles >= vertex_count):
        raise PipelineError(E_MESH_PATCH, f"{mesh.key}: triangle index out of range")
    for triangle in mesh.triangles:
        if (
            triangle[0] == triangle[1]
            or triangle[1] == triangle[2]
            or triangle[0] == triangle[2]
        ):
            raise PipelineError(E_MESH_PATCH, f"{mesh.key}: degenerate triangle")
    if np.any(~np.isfinite(mesh.vertices)):
        raise PipelineError(E_MESH_PATCH, f"{mesh.key}: non-finite vertex")
    used = np.unique(mesh.triangles)
    if used.shape[0] != vertex_count:
        raise PipelineError(
            E_MESH_PATCH,
            f"{mesh.key}: {vertex_count - used.shape[0]} dangling vertices",
        )


def build_tile_mesh_safe(
    key: TileKey,
    leaves: frozenset[TileKey],
    heightfield: Heightfield,
    config: PipelineConfig,
    height_grid: np.ndarray | None = None,
) -> TileMesh:
    """Build with stitching, then roll back to a plain grid on patch failure."""
    try:
        return build_tile_mesh(
            key, leaves, heightfield, config, height_grid
        )
    except PipelineError:
        rollback = config.with_overrides(enable_stitching=False)
        plain = build_tile_mesh(
            key, leaves, heightfield, rollback, height_grid
        )
        return TileMesh(
            key=plain.key,
            vertices=plain.vertices,
            triangles=plain.triangles,
            stitched_edges=("ROLLBACK",),
            height_grid=plain.height_grid,
        )
