"""Camera setup, world-space terrain tiles, culling, and mesh extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .math3d import (
    aabb_corners,
    aabb_visible,
    frustum_planes,
    look_at,
    orbit_camera_parameters,
    perspective,
    project_to_screen,
)
from .terrain import TerrainData


@dataclass(frozen=True)
class OrbitCamera:
    eye: np.ndarray
    target: np.ndarray
    view: np.ndarray
    projection: np.ndarray
    view_projection: np.ndarray
    planes: np.ndarray


@dataclass(frozen=True)
class WorldTile:
    row: int
    col: int
    vertex_row0: int
    vertex_row1: int
    vertex_col0: int
    vertex_col1: int
    x0: float
    x1: float
    z0: float
    z1: float
    y0: float
    y1: float
    screen_bbox: tuple[int, int, int, int]

    @property
    def tile_id(self) -> str:
        return f"r{self.row}c{self.col}"

    @property
    def minimum(self) -> np.ndarray:
        return np.array((self.x0, self.y0, self.z0), dtype=np.float64)

    @property
    def maximum(self) -> np.ndarray:
        return np.array((self.x1, self.y1, self.z1), dtype=np.float64)

    def bbox_dict(self) -> dict[str, float]:
        return {
            "x0": self.x0,
            "x1": self.x1,
            "y0": self.y0,
            "y1": self.y1,
            "z0": self.z0,
            "z1": self.z1,
        }


@dataclass(frozen=True)
class VisibilityResult:
    all_tiles: tuple[WorldTile, ...]
    visible: tuple[WorldTile, ...]
    culled_frustum: int
    culled_backface: int


@dataclass(frozen=True)
class TileMesh:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray


def setup_camera(config: AppConfig) -> OrbitCamera:
    terrain = config.terrain
    camera_config = config.camera
    target = np.array(
        (
            camera_config.target_fraction[0] * terrain.world_width,
            camera_config.target_fraction[1] * terrain.height_scale,
            camera_config.target_fraction[2] * terrain.world_depth,
        ),
        dtype=np.float64,
    )
    eye, target = orbit_camera_parameters(
        target,
        camera_config.azimuth_deg,
        camera_config.elevation_deg,
        camera_config.distance,
    )
    view = look_at(eye, target, (0.0, 1.0, 0.0))
    aspect = config.render.width / config.render.height
    projection = perspective(
        camera_config.fov_y_deg,
        aspect,
        camera_config.near,
        camera_config.far,
    )
    view_projection = projection @ view
    return OrbitCamera(
        eye=eye,
        target=target,
        view=view,
        projection=projection,
        view_projection=view_projection,
        planes=frustum_planes(view_projection),
    )


def _tile_backfaces(
    camera: OrbitCamera,
    normals: np.ndarray,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    y0: float,
    y1: float,
) -> bool:
    """True only when every tile corner faces away from the camera."""

    corners = np.array(
        (
            (x0, y0, z0),
            (x1, y0, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x0, y1, z0),
            (x1, y1, z0),
            (x0, y1, z1),
            (x1, y1, z1),
        ),
        dtype=np.float64,
    )
    to_camera = camera.eye[None, :] - corners
    lengths = np.linalg.norm(to_camera, axis=1, keepdims=True)
    to_camera /= np.maximum(lengths, 1e-15)
    sample_normals = np.array(
        (
            normals[0, 0],
            normals[0, -1],
            normals[-1, 0],
            normals[-1, -1],
            normals[len(normals) // 2, len(normals[0]) // 2],
        ),
        dtype=np.float64,
    )
    sample_dirs = to_camera[:4]
    facing = np.sum(sample_normals[:4] * sample_dirs, axis=1)
    return bool(np.max(facing) <= 0.0)


def build_world_tiles(
    config: AppConfig, terrain: TerrainData, camera: OrbitCamera
) -> VisibilityResult:
    terrain_config = config.terrain
    rows, cols = terrain.heights.shape
    tile_cells = terrain_config.tile_cells
    cell_rows = rows - 1
    cell_cols = cols - 1
    tile_row_count = math.ceil(cell_rows / tile_cells)
    tile_col_count = math.ceil(cell_cols / tile_cells)

    all_tiles: list[WorldTile] = []
    visible_tiles: list[WorldTile] = []
    culled_frustum = 0
    culled_backface = 0
    width_scale = terrain_config.world_width / cell_cols
    depth_scale = terrain_config.world_depth / cell_rows

    for tile_row in range(tile_row_count):
        vr0 = tile_row * tile_cells
        vr1 = min(vr0 + tile_cells, cell_rows)
        for tile_col in range(tile_col_count):
            vc0 = tile_col * tile_cells
            vc1 = min(vc0 + tile_cells, cell_cols)
            x0 = vc0 * width_scale
            x1 = vc1 * width_scale
            z0 = vr0 * depth_scale
            z1 = vr1 * depth_scale
            heights = terrain.heights[vr0 : vr1 + 1, vc0 : vc1 + 1]
            y0 = float(np.min(heights)) * terrain_config.height_scale
            y1 = float(np.max(heights)) * terrain_config.height_scale
            tile_normals = terrain.normals[vr0 : vr1 + 1, vc0 : vc1 + 1]
            corners = aabb_corners((x0, y0, z0), (x1, y1, z1))
            screen_bbox = project_to_screen(
                corners,
                camera.view_projection,
                config.render.width,
                config.render.height,
            )
            tile = WorldTile(
                row=tile_row,
                col=tile_col,
                vertex_row0=vr0,
                vertex_row1=vr1,
                vertex_col0=vc0,
                vertex_col1=vc1,
                x0=x0,
                x1=x1,
                z0=z0,
                z1=z1,
                y0=y0,
                y1=y1,
                screen_bbox=screen_bbox,
            )
            all_tiles.append(tile)

            if not aabb_visible(tile.minimum, tile.maximum, camera.planes):
                culled_frustum += 1
                continue
            if _tile_backfaces(
                camera, tile_normals, x0, x1, z0, z1, y0, y1
            ):
                culled_backface += 1
                continue
            visible_tiles.append(tile)

    visible_tiles.sort(key=lambda item: (item.row, item.col))
    return VisibilityResult(
        all_tiles=tuple(all_tiles),
        visible=tuple(visible_tiles),
        culled_frustum=culled_frustum,
        culled_backface=culled_backface,
    )


def extract_tile_mesh(
    config: AppConfig,
    terrain: TerrainData,
    tile: WorldTile,
    *,
    with_normals: bool = True,
) -> TileMesh:
    terrain_config = config.terrain
    rows, cols = terrain.heights.shape
    row_indices = np.arange(tile.vertex_row0, tile.vertex_row1 + 1)
    col_indices = np.arange(tile.vertex_col0, tile.vertex_col1 + 1)
    x = col_indices.astype(np.float64) * (
        terrain_config.world_width / (cols - 1)
    )
    z = row_indices.astype(np.float64) * (
        terrain_config.world_depth / (rows - 1)
    )
    height = (
        terrain.heights[
            tile.vertex_row0 : tile.vertex_row1 + 1,
            tile.vertex_col0 : tile.vertex_col1 + 1,
        ].astype(np.float64)
        * terrain_config.height_scale
    )
    positions = np.stack(
        np.broadcast_arrays(x[None, :], height, z[:, None]), axis=-1
    ).reshape(-1, 3)
    normals = terrain.normals[
        tile.vertex_row0 : tile.vertex_row1 + 1,
        tile.vertex_col0 : tile.vertex_col1 + 1,
    ].reshape(-1, 3)

    local_rows = tile.vertex_row1 - tile.vertex_row0
    local_cols = tile.vertex_col1 - tile.vertex_col0
    rr, cc = np.meshgrid(
        np.arange(local_rows), np.arange(local_cols), indexing="ij"
    )
    a = rr * (local_cols + 1) + cc
    b = a + 1
    c = a + (local_cols + 1)
    d = c + 1
    triangles = np.concatenate(
        (
            np.stack((a.reshape(-1), b.reshape(-1), d.reshape(-1)), axis=1),
            np.stack((a.reshape(-1), d.reshape(-1), c.reshape(-1)), axis=1),
        ),
        axis=0,
    ).astype(np.int32)
    if not with_normals:
        normals = np.empty((0, 3), dtype=np.float32)
    return TileMesh(positions=positions, normals=normals, triangles=triangles)

