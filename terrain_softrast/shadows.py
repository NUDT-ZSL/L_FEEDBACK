from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import Camera, CascadeSettings, DirectionalLight, RenderSettings
from .geometry import TerrainGeometry
from .math3d import (
    clip_to_frustum,
    cross2,
    frustum_corners,
    look_at,
    normalize,
    orthographic,
    select_up,
    to_homogeneous,
)


@dataclass
class Cascade:
    index: int
    split_near: float
    split_far: float
    light_view: np.ndarray
    light_projection: np.ndarray
    light_view_projection: np.ndarray
    ortho_near: float
    ortho_far: float
    half_extent: float
    texel_world: float
    shadow_size: int
    depth_map: np.ndarray
    light_direction: np.ndarray

    def light_clip(self, world: np.ndarray) -> np.ndarray:
        world = np.asarray(world, dtype=np.float64)
        return to_homogeneous(world) @ self.light_view_projection.T

    def light_distance(self, world: np.ndarray) -> np.ndarray:
        view = to_homogeneous(np.asarray(world, dtype=np.float64)) @ self.light_view.T
        return -view[:, 2]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "split_near": self.split_near,
            "split_far": self.split_far,
            "shadow_size": [self.shadow_size, self.shadow_size],
            "ortho_near": self.ortho_near,
            "ortho_far": self.ortho_far,
            "half_extent": self.half_extent,
            "texel_world": self.texel_world,
        }


def cascade_splits(camera: Camera, settings: CascadeSettings) -> list[float]:
    splits: list[float] = []
    near = camera.near
    far = camera.far
    for index in range(1, settings.count + 1):
        fraction = index / settings.count
        uniform = near + (far - near) * fraction
        practical = near * (far / near) ** fraction
        value = settings.split_lambda * practical + (1.0 - settings.split_lambda) * uniform
        splits.append(float(min(max(value, near + 1e-6), far)))
    splits[-1] = far
    for index in range(1, len(splits)):
        if splits[index] <= splits[index - 1]:
            splits[index] = min(far, splits[index - 1] + 1e-5)
    return splits


def build_cascades(
    camera: Camera,
    light: DirectionalLight,
    width: int,
    height: int,
    settings: CascadeSettings,
) -> list[Cascade]:
    camera = camera.normalized(width, height)
    light = light.normalized()
    splits = cascade_splits(camera, settings)
    direction = normalize(np.asarray(light.direction, dtype=np.float64))
    up = select_up(direction)
    cascades: list[Cascade] = []
    start = camera.near
    for index, end in enumerate(splits):
        corners = frustum_corners(camera, width, height, start, end)
        center = (np.min(corners, axis=0) + np.max(corners, axis=0)) * 0.5
        radius = float(np.max(np.linalg.norm(corners - center, axis=1)))
        distance = max(radius * 1.25, 0.5)
        eye = center - direction * distance
        view = look_at(eye, center, up)
        light_corners_view = to_homogeneous(corners) @ view.T
        distances = -light_corners_view[:, 2]
        near_distance = max(0.01, float(np.min(distances)) - radius * 0.05)
        far_distance = max(near_distance + 0.01, float(np.max(distances)) + radius * 0.05)
        half_extent = max(radius * 1.02, 0.02)
        margin = half_extent * 1e-4
        projection = orthographic(
            -half_extent,
            half_extent,
            -half_extent,
            half_extent,
            near_distance - margin,
            far_distance + margin,
        )
        view_projection = projection @ view
        texel_world = 2.0 * half_extent / settings.shadow_size
        cascades.append(
            Cascade(
                index=index,
                split_near=float(start),
                split_far=float(end),
                light_view=view,
                light_projection=projection,
                light_view_projection=view_projection,
                ortho_near=near_distance - margin,
                ortho_far=far_distance + margin,
                half_extent=half_extent,
                texel_world=float(texel_world),
                shadow_size=settings.shadow_size,
                depth_map=np.full((settings.shadow_size, settings.shadow_size), np.inf, dtype=np.float32),
                light_direction=direction,
            )
        )
        start = end
    return cascades


def _top_left(edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    dx = float(edge_end[0] - edge_start[0])
    dy = float(edge_end[1] - edge_start[1])
    return (dy > 0.0) | ((dy == 0.0) & (dx < 0.0))


def _rasterize_depth_triangle(
    points: np.ndarray,
    depths: np.ndarray,
    depth_map: np.ndarray,
    scissor: tuple[int, int, int, int],
) -> int:
    area = float(cross2(points[1, :2] - points[0, :2], points[2, :2] - points[0, :2]))
    if abs(area) <= 1e-12:
        return 0
    if area < 0.0:
        points = points[[0, 2, 1]]
        depths = depths[[0, 2, 1]]
        area = -area
    sx0, sy0, sx1, sy1 = scissor
    min_x = max(sx0, int(math.floor(float(np.min(points[:, 0])))))
    max_x = min(sx1, int(math.ceil(float(np.max(points[:, 0])))))
    min_y = max(sy0, int(math.floor(float(np.min(points[:, 1])))))
    max_y = min(sy1, int(math.ceil(float(np.max(points[:, 1])))))
    if min_x >= max_x or min_y >= max_y:
        return 0
    xs = np.arange(min_x, max_x, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y, dtype=np.float64) + 0.5
    px, py = np.meshgrid(xs, ys)
    p = np.stack((px, py), axis=-1)
    e0 = cross2(points[2, :2] - points[1, :2], p - points[1, :2])
    e1 = cross2(points[0, :2] - points[2, :2], p - points[2, :2])
    e2 = cross2(points[1, :2] - points[0, :2], p - points[0, :2])
    eps = max(1e-10, abs(area) * 1e-12)
    tl0 = _top_left(points[1], points[2])
    tl1 = _top_left(points[2], points[0])
    tl2 = _top_left(points[0], points[1])
    mask = (
        ((e0 > eps) | ((np.abs(e0) <= eps) & tl0))
        & ((e1 > eps) | ((np.abs(e1) <= eps) & tl1))
        & ((e2 > eps) | ((np.abs(e2) <= eps) & tl2))
    )
    if not np.any(mask):
        return 0
    l0 = e0[mask] / area
    l1 = e1[mask] / area
    l2 = e2[mask] / area
    candidate_depth = l0 * depths[0] + l1 * depths[1] + l2 * depths[2]
    rows, columns = np.nonzero(mask)
    rows = rows + min_y
    columns = columns + min_x
    current = depth_map[rows, columns]
    wins = candidate_depth < current
    if not np.any(wins):
        return 0
    depth_map[rows[wins], columns[wins]] = np.asarray(candidate_depth[wins], dtype=np.float32)
    return int(np.count_nonzero(wins))


def rasterize_cascade_depth(
    cascade: Cascade,
    geometry: TerrainGeometry,
    scissor: tuple[int, int, int, int] | None = None,
) -> dict:
    size = cascade.shadow_size
    if scissor is None:
        scissor = (0, 0, size, size)
    sx0, sy0, sx1, sy1 = scissor
    cascade.depth_map[sy0:sy1, sx0:sx1] = np.inf
    vertices = geometry.vertices.reshape(-1, 3)
    triangles = geometry.triangles
    clip_vertices = to_homogeneous(vertices) @ cascade.light_view_projection.T
    light_distances = cascade.light_distance(vertices)
    written = 0
    for indices in triangles:
        clip = clip_vertices[indices]
        distances = light_distances[indices]
        clipped, clipped_distances = clip_to_frustum(clip, distances[:, None])
        if clipped.shape[0] < 3 or clipped_distances is None:
            continue
        ndc = clipped[:, :3] / clipped[:, 3:4]
        screen = np.empty((clipped.shape[0], 3), dtype=np.float64)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * size
        screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * size
        screen[:, 2] = ndc[:, 2]
        for fan in range(1, clipped.shape[0] - 1):
            tri_screen = screen[[0, fan, fan + 1]]
            tri_depths = clipped_distances[[0, fan, fan + 1], 0]
            tri_min_x = float(np.min(tri_screen[:, 0]))
            tri_max_x = float(np.max(tri_screen[:, 0]))
            tri_min_y = float(np.min(tri_screen[:, 1]))
            tri_max_y = float(np.max(tri_screen[:, 1]))
            if tri_max_x < sx0 or tri_min_x > sx1 or tri_max_y < sy0 or tri_min_y > sy1:
                continue
            written += _rasterize_depth_triangle(tri_screen, tri_depths, cascade.depth_map, scissor)
    return {"writes": written, "scissor": scissor}


def project_shadow_region(
    cascade: Cascade,
    world_min: np.ndarray,
    world_max: np.ndarray,
    padding: int = 4,
) -> tuple[int, int, int, int] | None:
    corners = np.asarray(
        [
            (x, y, z)
            for x in (world_min[0], world_max[0])
            for y in (world_min[1], world_max[1])
            for z in (world_min[2], world_max[2])
        ],
        dtype=np.float64,
    )
    clip = to_homogeneous(corners) @ cascade.light_view_projection.T
    ndc = clip[:, :3] / clip[:, 3:4]
    u = (ndc[:, 0] * 0.5 + 0.5) * cascade.shadow_size
    v = (0.5 - ndc[:, 1] * 0.5) * cascade.shadow_size
    x0 = max(0, int(math.floor(float(np.min(u)))) - padding)
    x1 = min(cascade.shadow_size, int(math.ceil(float(np.max(u)))) + padding)
    y0 = max(0, int(math.floor(float(np.min(v)))) - padding)
    y1 = min(cascade.shadow_size, int(math.ceil(float(np.max(v)))) + padding)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def update_cascade_depths(
    cascades: list[Cascade],
    geometry: TerrainGeometry,
    world_min: np.ndarray,
    world_max: np.ndarray,
) -> list[dict | None]:
    updates: list[dict | None] = []
    for cascade in cascades:
        scissor = project_shadow_region(cascade, world_min, world_max)
        if scissor is None:
            updates.append(None)
            continue
        result = rasterize_cascade_depth(cascade, geometry, scissor)
        result["dirty_texels"] = (scissor[2] - scissor[0]) * (scissor[3] - scissor[1])
        updates.append(result)
    return updates


def cascade_weights(depths: np.ndarray, cascades: list[Cascade], overlap_fraction: float) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float64)
    count = len(cascades)
    weights = np.zeros((depths.size, count), dtype=np.float64)
    boundaries = np.asarray([cascade.split_far for cascade in cascades], dtype=np.float64)
    for index in range(count):
        start = cascades[index].split_near
        end = cascades[index].split_far
        inside = (depths >= start) & (depths < end)
        if index == count - 1:
            inside |= depths >= end
        weights[inside, index] = 1.0
    base = np.searchsorted(boundaries, depths, side="right")
    base = np.clip(base, 0, count - 1)
    for index in range(count - 1):
        length = cascades[index].split_far - cascades[index].split_near
        overlap = max(length * overlap_fraction, 1e-9)
        threshold = cascades[index].split_far - overlap
        mask = (depths >= threshold) & (depths < cascades[index].split_far)
        if not np.any(mask):
            continue
        t = np.clip((depths[mask] - threshold) / overlap, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        weights[mask, :] = 0.0
        weights[mask, index] = 1.0 - smooth
        weights[mask, index + 1] = smooth
    total = np.sum(weights, axis=1, keepdims=True)
    zero = total[:, 0] <= 0.0
    if np.any(zero):
        weights[zero, -1] = 1.0
        total = np.sum(weights, axis=1, keepdims=True)
    return weights / np.maximum(total, 1e-15)


def _sample_shadow_cascade(
    cascade: Cascade,
    world: np.ndarray,
    normals: np.ndarray,
    settings: RenderSettings,
) -> np.ndarray:
    size = cascade.shadow_size
    offset_world = world + normals * (cascade.texel_world * settings.shadow_normal_offset)
    clip = cascade.light_clip(offset_world)
    w = clip[:, 3]
    valid = (w > 1e-9) & np.all(np.isfinite(clip), axis=1)
    safe_w = np.where(valid, w, 1.0)
    ndc = clip[:, :3] / safe_w[:, None]
    u = (ndc[:, 0] * 0.5 + 0.5) * size
    v = (0.5 - ndc[:, 1] * 0.5) * size
    valid &= (u >= 0.0) & (u <= size - 1.0) & (v >= 0.0) & (v <= size - 1.0)
    receiver_depth = cascade.light_distance(offset_world)
    facing = np.maximum(np.abs(np.sum(normals * (-cascade.light_direction)[None, :], axis=1)), 0.08)
    tan_theta = np.sqrt(np.maximum(1.0 - facing * facing, 0.0)) / facing
    bias = settings.shadow_constant_bias + settings.shadow_slope_bias * cascade.texel_world * tan_theta
    lit_count = np.zeros(world.shape[0], dtype=np.float64)
    count = 0
    radius = settings.shadow_pcf_radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sample_u = np.clip(np.floor(u + 0.5) + dx, 0, size - 1).astype(np.int64)
            sample_v = np.clip(np.floor(v + 0.5) + dy, 0, size - 1).astype(np.int64)
            stored = cascade.depth_map[sample_v, sample_u].astype(np.float64)
            lit = receiver_depth - bias <= stored + 1e-5
            lit_count += lit.astype(np.float64)
            count += 1
    result = np.ones(world.shape[0], dtype=np.float64)
    result[valid] = lit_count[valid] / count
    return result


def sample_visibility(
    cascades: list[Cascade],
    world: np.ndarray,
    normals: np.ndarray,
    linear_depths: np.ndarray,
    settings: RenderSettings,
) -> np.ndarray:
    weights = cascade_weights(linear_depths, cascades, settings.cascades.overlap_fraction)
    visibility = np.zeros(world.shape[0], dtype=np.float64)
    for index, cascade in enumerate(cascades):
        active = weights[:, index] > 1e-12
        if not np.any(active):
            continue
        cascade_lit = _sample_shadow_cascade(cascade, world[active], normals[active], settings)
        visibility[active] += weights[active, index] * cascade_lit
    return np.clip(visibility, 0.0, 1.0)
