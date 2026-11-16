"""Directional light fitting, shadow depth maps, and fixed-count PCF."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .math3d import (
    aabb_corners,
    direction_from_angles,
    look_at,
    orthographic,
    transform_points,
)
from .raster import rasterize_tile_shadow
from .scene import WorldTile, extract_tile_mesh
from .terrain import TerrainData


@dataclass(frozen=True)
class LightCamera:
    direction_to_light: np.ndarray
    view: np.ndarray
    projection: np.ndarray
    view_projection: np.ndarray
    near: float
    far: float
    extents: float


@dataclass(frozen=True)
class LightingResources:
    light_camera: LightCamera | None
    depth_map: np.ndarray | None
    pcf_offsets: np.ndarray


def fit_light_camera(config: AppConfig, terrain: TerrainData) -> LightCamera:
    terrain_config = config.terrain
    light_direction = direction_from_angles(
        config.light.azimuth_deg, config.light.elevation_deg
    )
    low = np.array(
        (
            0.0,
            float(np.min(terrain.heights)) * terrain_config.height_scale,
            0.0,
        ),
        dtype=np.float64,
    )
    high = np.array(
        (
            terrain_config.world_width,
            float(np.max(terrain.heights)) * terrain_config.height_scale,
            terrain_config.world_depth,
        ),
        dtype=np.float64,
    )
    center = (low + high) * 0.5
    extent_hint = 0.5 * float(np.linalg.norm(high - low))
    distance = max(extent_hint * 2.5, 2.0)
    eye = center + light_direction * distance
    up = (0.0, 1.0, 0.0)
    if abs(float(np.dot(light_direction, np.asarray(up)))) > 0.98:
        up = (0.0, 0.0, 1.0)
    view = look_at(eye, center, up)

    corners = aabb_corners(low, high)
    view_corners = transform_points(view, corners)
    extents = max(
        float(np.max(np.abs(view_corners[:, 0]))),
        float(np.max(np.abs(view_corners[:, 1]))),
    )
    padding = max(extents * 0.03, 0.5)
    extents += padding
    near = max(0.01, -float(np.max(view_corners[:, 2])) - padding)
    far = -float(np.min(view_corners[:, 2])) + padding
    if far <= near:
        far = near + 1.0
    projection = orthographic(-extents, extents, -extents, extents, near, far)
    return LightCamera(
        direction_to_light=light_direction,
        view=view,
        projection=projection,
        view_projection=projection @ view,
        near=near,
        far=far,
        extents=extents,
    )


def render_shadow_map(
    config: AppConfig,
    terrain: TerrainData,
    tiles: tuple[WorldTile, ...],
    light_camera: LightCamera,
) -> np.ndarray:
    resolution = config.light.shadow.resolution
    depth = np.full((resolution, resolution), np.inf, dtype=np.float32)
    for tile in tiles:
        mesh = extract_tile_mesh(config, terrain, tile, with_normals=False)
        rasterize_tile_shadow(mesh, light_camera.view_projection, depth)
    finite = np.isfinite(depth)
    depth[finite] = depth[finite] * 0.5 + 0.5
    return depth


def pcf_disk_offsets(sample_count: int, radius: float) -> np.ndarray:
    if sample_count <= 1:
        return np.zeros((1, 2), dtype=np.float64)
    indices = np.arange(sample_count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    radial = np.sqrt((indices + 0.5) / sample_count) * radius
    angle = indices * golden_angle
    return np.stack((np.cos(angle) * radial, np.sin(angle) * radial), axis=1)


def build_lighting_resources(
    config: AppConfig,
    terrain: TerrainData,
    tiles: tuple[WorldTile, ...],
    depth_map: np.ndarray | None = None,
) -> LightingResources:
    if not config.light.shadow.enabled:
        return LightingResources(light_camera=None, depth_map=None, pcf_offsets=np.zeros((1, 2)))
    light_camera = fit_light_camera(config, terrain)
    if depth_map is None:
        depth_map = render_shadow_map(config, terrain, tiles, light_camera)
    expected = (config.light.shadow.resolution, config.light.shadow.resolution)
    if depth_map.shape != expected:
        raise ValueError(
            f"shadow map shape {depth_map.shape} does not match configured {expected}"
        )
    offsets = pcf_disk_offsets(
        config.light.shadow.pcf_samples, config.light.shadow.filter_radius
    )
    return LightingResources(
        light_camera=light_camera,
        depth_map=depth_map,
        pcf_offsets=offsets,
    )


def pcf_visibility(
    positions: np.ndarray,
    normals: np.ndarray,
    config: AppConfig,
    resources: LightingResources,
) -> np.ndarray:
    count = positions.shape[0]
    if (
        not config.light.shadow.enabled
        or resources.light_camera is None
        or resources.depth_map is None
    ):
        return np.ones(count, dtype=np.float64)

    light = resources.light_camera
    shadow = config.light.shadow
    shifted = positions.astype(np.float64) + normals.astype(np.float64) * shadow.normal_offset
    homogeneous = np.concatenate(
        (shifted, np.ones((shifted.shape[0], 1), dtype=np.float64)), axis=1
    )
    clip = homogeneous @ light.view_projection.T
    widths = clip[:, 3]
    valid = widths > 1e-12
    ndc = np.zeros((count, 3), dtype=np.float64)
    ndc[valid] = clip[valid, :3] / widths[valid, None]
    u = ndc[:, 0] * 0.5 + 0.5
    v = ndc[:, 1] * 0.5 + 0.5
    current_depth = ndc[:, 2] * 0.5 + 0.5
    result = np.ones(count, dtype=np.float64)
    outside = (
        (~valid)
        | (u < 0.0)
        | (u > 1.0)
        | (v < 0.0)
        | (v > 1.0)
        | (current_depth < 0.0)
        | (current_depth > 1.0)
    )
    candidates = np.flatnonzero(~outside)
    if candidates.size == 0:
        return result

    resolution = resources.depth_map.shape[0]
    bias = shadow.depth_bias
    current_u = u[candidates]
    current_v = v[candidates]
    current_z = current_depth[candidates]
    visibility = np.zeros(candidates.size, dtype=np.float64)
    for offset_x, offset_y in resources.pcf_offsets:
        columns = np.floor(current_u * resolution + offset_x).astype(np.int64)
        rows = np.floor((1.0 - current_v) * resolution + offset_y).astype(np.int64)
        in_bounds = (
            (columns >= 0)
            & (columns < resolution)
            & (rows >= 0)
            & (rows < resolution)
        )
        sampled = np.full(candidates.size, np.inf, dtype=np.float64)
        sampled[in_bounds] = resources.depth_map[
            rows[in_bounds], columns[in_bounds]
        ]
        visibility += (current_z <= sampled + bias).astype(np.float64)
    visibility /= max(resources.pcf_offsets.shape[0], 1)
    result[candidates] = visibility
    return result

