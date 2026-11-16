"""CPU triangle rasterization for camera G-buffers and light depth maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import TileMesh


_INSIDE_EPS = 1e-7


@dataclass
class CameraGBuffer:
    depth: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    tile_indices: np.ndarray
    covered: np.ndarray


def allocate_gbuffer(width: int, height: int) -> CameraGBuffer:
    return CameraGBuffer(
        depth=np.zeros((height, width), dtype=np.float32),
        positions=np.zeros((height, width, 3), dtype=np.float32),
        normals=np.zeros((height, width, 3), dtype=np.float32),
        tile_indices=np.full((height, width), -1, dtype=np.int32),
        covered=np.zeros((height, width), dtype=bool),
    )


def _edge(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _clip_triangle_near(
    clip: np.ndarray, world: np.ndarray, normals: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Clip one triangle against OpenGL's near plane z + w >= 0."""

    attributes = np.concatenate((clip, world, normals), axis=1)
    plane_values = clip[:, 2] + clip[:, 3]
    polygon: list[np.ndarray] = []
    for index in range(3):
        current = attributes[index]
        following = attributes[(index + 1) % 3]
        current_plane = plane_values[index]
        following_plane = plane_values[(index + 1) % 3]
        current_inside = current_plane >= 0.0
        following_inside = following_plane >= 0.0
        if current_inside:
            polygon.append(current)
        if current_inside != following_inside:
            denominator = current_plane - following_plane
            if abs(denominator) <= 1e-15:
                continue
            t = current_plane / denominator
            polygon.append(current + (following - current) * t)

    if len(polygon) < 3:
        return []
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(1, len(polygon) - 1):
        values = np.stack((polygon[0], polygon[index], polygon[index + 1]))
        triangles.append((values[:, :4], values[:, 4:7], values[:, 7:10]))
    return triangles


def rasterize_tile_camera(
    mesh: TileMesh,
    view_projection: np.ndarray,
    gbuffer: CameraGBuffer,
    tile_index: int,
    *,
    image_width: int,
    image_height: int,
    x_offset: int,
    y_offset: int,
) -> None:
    positions = mesh.positions.astype(np.float64)
    normals = mesh.normals.astype(np.float64)
    homogeneous = np.concatenate(
        (positions, np.ones((positions.shape[0], 1), dtype=np.float64)), axis=1
    )
    clip_vertices = homogeneous @ view_projection.T
    strip_height = gbuffer.depth.shape[0]
    strip_width = gbuffer.depth.shape[1]

    for triangle in mesh.triangles:
        triangle_clip = clip_vertices[triangle]
        triangle_world = positions[triangle]
        triangle_normals = normals[triangle]
        polygons = _clip_triangle_near(triangle_clip, triangle_world, triangle_normals)
        for tri_clip, tri_world, tri_normals in polygons:
            widths = tri_clip[:, 3]
            if np.any(widths <= 1e-9):
                continue
            ndc = tri_clip[:, :3] / widths[:, None]
            sx = (ndc[:, 0] * 0.5 + 0.5) * image_width - x_offset
            sy = (0.5 - ndc[:, 1] * 0.5) * image_height - y_offset
            local_min_y = max(0, int(np.floor(np.min(sy))))
            local_max_y = min(strip_height - 1, int(np.ceil(np.max(sy))))
            min_x = max(0, int(np.floor(np.min(sx))))
            max_x = min(strip_width - 1, int(np.ceil(np.max(sx))))
            if local_min_y > local_max_y or min_x > max_x:
                continue

            xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
            ys = np.arange(local_min_y, local_max_y + 1, dtype=np.float64) + 0.5
            px, py = np.meshgrid(xs, ys)
            area = _edge(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2])
            if abs(area) <= 1e-12:
                continue
            b0 = _edge(sx[1], sy[1], sx[2], sy[2], px, py) / area
            b1 = _edge(sx[2], sy[2], sx[0], sy[0], px, py) / area
            b2 = 1.0 - b0 - b1
            inside = (b0 >= -_INSIDE_EPS) & (b1 >= -_INSIDE_EPS) & (b2 >= -_INSIDE_EPS)
            if not np.any(inside):
                continue

            inverse_w = 1.0 / widths
            inverse_interpolated = (
                b0 * inverse_w[0] + b1 * inverse_w[1] + b2 * inverse_w[2]
            )
            valid = inside & (inverse_interpolated > 1e-12)
            if not np.any(valid):
                continue
            depth_view = gbuffer.depth[local_min_y : local_max_y + 1, min_x : max_x + 1]
            update = valid & (inverse_interpolated > depth_view)
            if not np.any(update):
                continue

            inverse_safe = np.maximum(inverse_interpolated, 1e-15)
            weights = np.stack((b0, b1, b2), axis=-1)
            world_values = (
                np.sum(
                    weights[..., None]
                    * (tri_world * inverse_w[:, None])[None, None, :, :],
                    axis=2,
                )
                / inverse_safe[..., None]
            )
            normal_values = (
                np.sum(
                    weights[..., None]
                    * (tri_normals * inverse_w[:, None])[None, None, :, :],
                    axis=2,
                )
                / inverse_safe[..., None]
            )

            pos_region = gbuffer.positions[local_min_y : local_max_y + 1, min_x : max_x + 1]
            norm_region = gbuffer.normals[local_min_y : local_max_y + 1, min_x : max_x + 1]
            tile_region = gbuffer.tile_indices[local_min_y : local_max_y + 1, min_x : max_x + 1]
            cover_region = gbuffer.covered[local_min_y : local_max_y + 1, min_x : max_x + 1]
            depth_view[update] = inverse_interpolated[update]
            pos_region[update] = world_values[update]
            norm_region[update] = normal_values[update]
            tile_region[update] = tile_index
            cover_region[update] = True


def rasterize_tile_shadow(
    mesh: TileMesh,
    light_view_projection: np.ndarray,
    depth_buffer: np.ndarray,
) -> None:
    """Rasterize a tile mesh into a light-space NDC depth map."""

    resolution_y, resolution_x = depth_buffer.shape
    positions = mesh.positions.astype(np.float64)
    homogeneous = np.concatenate(
        (positions, np.ones((positions.shape[0], 1), dtype=np.float64)), axis=1
    )
    clip_vertices = homogeneous @ light_view_projection.T
    widths = clip_vertices[:, 3]
    if not np.all(np.abs(widths) > 1e-12):
        return
    ndc_vertices = clip_vertices[:, :3] / widths[:, None]
    sx_vertices = (ndc_vertices[:, 0] * 0.5 + 0.5) * resolution_x
    sy_vertices = (0.5 - ndc_vertices[:, 1] * 0.5) * resolution_y
    z_vertices = ndc_vertices[:, 2]

    for triangle in mesh.triangles:
        sx = sx_vertices[triangle]
        sy = sy_vertices[triangle]
        sz = z_vertices[triangle]
        min_x = max(0, int(np.floor(np.min(sx))))
        max_x = min(resolution_x - 1, int(np.ceil(np.max(sx))))
        min_y = max(0, int(np.floor(np.min(sy))))
        max_y = min(resolution_y - 1, int(np.ceil(np.max(sy))))
        if min_x > max_x or min_y > max_y:
            continue
        if np.all(sz < -1.5) or np.all(sz > 1.5):
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        px, py = np.meshgrid(xs, ys)
        area = _edge(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2])
        if abs(area) <= 1e-12:
            continue
        b0 = _edge(sx[1], sy[1], sx[2], sy[2], px, py) / area
        b1 = _edge(sx[2], sy[2], sx[0], sy[0], px, py) / area
        b2 = 1.0 - b0 - b1
        inside = (
            (b0 >= -_INSIDE_EPS)
            & (b1 >= -_INSIDE_EPS)
            & (b2 >= -_INSIDE_EPS)
        )
        if not np.any(inside):
            continue
        z_values = b0 * sz[0] + b1 * sz[1] + b2 * sz[2]
        clipped_z = (z_values >= -1.0) & (z_values <= 1.0)
        region = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & clipped_z & (z_values < region)
        region[update] = z_values[update]

