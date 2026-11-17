from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import Camera, DirectionalLight, Material, PointLight, RenderSettings
from .geometry import TerrainGeometry
from .math3d import clip_to_frustum, cross2, to_homogeneous
from .shadows import Cascade, sample_visibility
from .shade import shade_fragments


BACKGROUND_LINEAR = np.asarray((0.018, 0.022, 0.030), dtype=np.float32)


@dataclass
class ProjectedTriangle:
    screen: np.ndarray
    inv_w: np.ndarray
    attributes: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class ProjectionStats:
    source_triangles: int
    clipped_triangles: int
    backface_culled: int
    degenerate_rejected: int
    behind_near_rejected: int


def _top_left(edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    dx = float(edge_end[0] - edge_start[0])
    dy = float(edge_end[1] - edge_start[1])
    return dy > 0.0 or (dy == 0.0 and dx < 0.0)


def project_geometry(
    geometry: TerrainGeometry,
    camera: Camera,
    settings: RenderSettings,
) -> tuple[list[ProjectedTriangle], ProjectionStats]:
    camera = camera.normalized(settings.width, settings.height)
    from .math3d import camera_matrices

    _, _, view_projection = camera_matrices(camera, settings.width, settings.height)
    vertices = geometry.vertices.reshape(-1, 3)
    normals = geometry.normals.reshape(-1, 3)
    weights = geometry.weights.reshape(vertices.shape[0], -1)
    clip_vertices = to_homogeneous(vertices) @ view_projection.T
    attributes = np.concatenate((vertices, normals, weights), axis=1)
    projected: list[ProjectedTriangle] = []
    clipped_count = 0
    culled_count = 0
    degenerate_count = 0
    behind_count = 0
    for triangle in geometry.triangles:
        clip = clip_vertices[triangle]
        attrs = attributes[triangle]
        if np.all(clip[:, 3] <= 1e-7):
            behind_count += 1
            continue
        clipped, clipped_attrs = clip_to_frustum(clip, attrs)
        if clipped.shape[0] < 3 or clipped_attrs is None:
            behind_count += 1
            continue
        clipped_count += 1
        w = clipped[:, 3]
        if np.any(w <= 1e-9) or not np.all(np.isfinite(clipped)):
            behind_count += 1
            continue
        ndc = clipped[:, :3] / w[:, None]
        screen = np.empty((clipped.shape[0], 3), dtype=np.float64)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * settings.width
        screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * settings.height
        screen[:, 2] = ndc[:, 2]
        for fan in range(1, clipped.shape[0] - 1):
            tri_screen = screen[[0, fan, fan + 1], :2]
            tri_w = w[[0, fan, fan + 1]]
            tri_attrs = clipped_attrs[[0, fan, fan + 1]]
            area = float(cross2(tri_screen[1] - tri_screen[0], tri_screen[2] - tri_screen[0]))
            if settings.backface_culling and area >= -1e-12:
                culled_count += 1
                continue
            if abs(area) <= 1e-12:
                degenerate_count += 1
                continue
            if area < 0.0:
                tri_screen = tri_screen[[0, 2, 1]]
                tri_w = tri_w[[0, 2, 1]]
                tri_attrs = tri_attrs[[0, 2, 1]]
                area = -area
            min_x = max(0, int(math.floor(float(np.min(tri_screen[:, 0])))))
            max_x = min(settings.width, int(math.ceil(float(np.max(tri_screen[:, 0])))))
            min_y = max(0, int(math.floor(float(np.min(tri_screen[:, 1])))))
            max_y = min(settings.height, int(math.ceil(float(np.max(tri_screen[:, 1])))))
            if min_x >= max_x or min_y >= max_y:
                continue
            projected.append(
                ProjectedTriangle(
                    screen=tri_screen,
                    inv_w=1.0 / tri_w,
                    attributes=tri_attrs,
                    bbox=(min_x, min_y, max_x, max_y),
                )
            )
    stats = ProjectionStats(
        source_triangles=geometry.triangle_count,
        clipped_triangles=clipped_count,
        backface_culled=culled_count,
        degenerate_rejected=degenerate_count,
        behind_near_rejected=behind_count,
    )
    return projected, stats


def _rasterize_triangle(
    triangle: ProjectedTriangle,
    tile: tuple[int, int, int, int],
    color: np.ndarray,
    depth: np.ndarray,
    visibility: np.ndarray,
    geometry: TerrainGeometry,
    materials: tuple[Material, ...],
    camera: Camera,
    directional_light: DirectionalLight,
    point_lights: tuple[PointLight, ...],
    settings: RenderSettings,
    cascades: list[Cascade],
) -> int:
    x0 = max(tile[0], triangle.bbox[0])
    y0 = max(tile[1], triangle.bbox[1])
    x1 = min(tile[2], triangle.bbox[2])
    y1 = min(tile[3], triangle.bbox[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    points = triangle.screen
    area = float(cross2(points[1] - points[0], points[2] - points[0]))
    if area <= 0.0:
        return 0
    xs = np.arange(x0, x1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1, dtype=np.float64) + 0.5
    px, py = np.meshgrid(xs, ys)
    sample = np.stack((px, py), axis=-1)
    e0 = cross2(points[2] - points[1], sample - points[1])
    e1 = cross2(points[0] - points[2], sample - points[2])
    e2 = cross2(points[1] - points[0], sample - points[0])
    l0 = e0 / area
    l1 = e1 / area
    l2 = e2 / area
    eps = 1e-12
    tl0 = _top_left(points[1], points[2])
    tl1 = _top_left(points[2], points[0])
    tl2 = _top_left(points[0], points[1])
    inside0 = (l0 > eps) | ((np.abs(l0) <= eps) & tl0)
    inside1 = (l1 > eps) | ((np.abs(l1) <= eps) & tl1)
    inside2 = (l2 > eps) | ((np.abs(l2) <= eps) & tl2)
    mask = inside0 & inside1 & inside2
    if not np.any(mask):
        return 0
    denominator = l0 * triangle.inv_w[0] + l1 * triangle.inv_w[1] + l2 * triangle.inv_w[2]
    valid = mask & (denominator > 1e-15) & np.isfinite(denominator)
    if not np.any(valid):
        return 0
    linear_depth = 1.0 / denominator
    rows, columns = np.nonzero(valid)
    rows = rows + y0
    columns = columns + x0
    candidate_depth = linear_depth[valid]
    current_depth = depth[rows, columns]
    wins = (candidate_depth >= 0.0) & (candidate_depth < current_depth)
    if not np.any(wins):
        return 0
    rows = rows[wins]
    columns = columns[wins]
    candidate_depth = candidate_depth[wins]
    l0v = l0[valid][wins]
    l1v = l1[valid][wins]
    l2v = l2[valid][wins]
    invw = triangle.inv_w
    denom = 1.0 / candidate_depth
    attributes = (
        (l0v * invw[0])[:, None] * triangle.attributes[0][None, :]
        + (l1v * invw[1])[:, None] * triangle.attributes[1][None, :]
        + (l2v * invw[2])[:, None] * triangle.attributes[2][None, :]
    ) / denom[:, None]
    normal_count = 3
    weight_count = geometry.weights.shape[-1]
    world = attributes[:, :3]
    normals = attributes[:, 3 : 3 + normal_count]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    weights = attributes[:, 3 + normal_count : 3 + normal_count + weight_count]
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-15)
    vis = sample_visibility(cascades, world, normals, candidate_depth, settings)
    shaded = shade_fragments(
        world,
        normals,
        weights,
        vis,
        materials,
        camera,
        directional_light,
        point_lights,
        settings,
    )
    depth[rows, columns] = np.asarray(candidate_depth, dtype=np.float32)
    visibility[rows, columns] = np.asarray(vis, dtype=np.float32)
    color[rows, columns] = np.asarray(shaded, dtype=np.float32)
    return int(rows.size)


def tile_grid(width: int, height: int, tile_size: int) -> tuple[int, int]:
    return (width + tile_size - 1) // tile_size, (height + tile_size - 1) // tile_size


def tile_rect(columns: int, tile_size: int, tile_id: int, width: int, height: int) -> tuple[int, int, int, int]:
    tx = tile_id % columns
    ty = tile_id // columns
    return (
        tx * tile_size,
        ty * tile_size,
        min(width, (tx + 1) * tile_size),
        min(height, (ty + 1) * tile_size),
    )


def _bin_triangles(
    triangles: list[ProjectedTriangle],
    columns: int,
    tile_size: int,
    width: int,
    height: int,
) -> dict[int, list[int]]:
    rows = (height + tile_size - 1) // tile_size
    bins: dict[int, list[int]] = {}
    for index, triangle in enumerate(triangles):
        x0, y0, x1, y1 = triangle.bbox
        tx0 = x0 // tile_size
        tx1 = min(columns - 1, (x1 - 1) // tile_size)
        ty0 = y0 // tile_size
        ty1 = min(rows - 1, (y1 - 1) // tile_size)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                bins.setdefault(ty * columns + tx, []).append(index)
    return bins


def rasterize_selected_tiles(
    geometry: TerrainGeometry,
    camera: Camera,
    directional_light: DirectionalLight,
    point_lights: tuple[PointLight, ...],
    materials: tuple[Material, ...],
    settings: RenderSettings,
    cascades: list[Cascade],
    color: np.ndarray,
    depth: np.ndarray,
    visibility: np.ndarray,
    selected_tiles: set[int],
) -> dict:
    projected, projection_stats = project_geometry(geometry, camera, settings)
    columns, rows = tile_grid(settings.width, settings.height, settings.tile_size)
    bins = _bin_triangles(projected, columns, settings.tile_size, settings.width, settings.height)
    fragments = 0
    rendered_triangles = 0
    for tile_id in sorted(selected_tiles):
        rect = tile_rect(columns, settings.tile_size, tile_id, settings.width, settings.height)
        x0, y0, x1, y1 = rect
        color[y0:y1, x0:x1] = BACKGROUND_LINEAR
        depth[y0:y1, x0:x1] = np.inf
        visibility[y0:y1, x0:x1] = 0.0
        for triangle_index in bins.get(tile_id, ()):
            triangle = projected[triangle_index]
            written = _rasterize_triangle(
                triangle,
                rect,
                color,
                depth,
                visibility,
                geometry,
                materials,
                camera,
                directional_light,
                point_lights,
                settings,
                cascades,
            )
            fragments += written
            if written:
                rendered_triangles += 1
    return {
        "fragments": fragments,
        "rendered_triangles": rendered_triangles,
        "projected_triangles": len(projected),
        "total_tiles": columns * rows,
        "rendered_tiles": len(selected_tiles),
        "projection": projection_stats.__dict__,
    }
