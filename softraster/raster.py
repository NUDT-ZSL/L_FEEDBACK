"""Near-plane clipping and deterministic NumPy triangle rasterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .math3d import (
    EPS,
    OrthographicCamera,
    PerspectiveCamera,
    clip_polygon_far,
    clip_polygon_near,
)
from .scene import Mesh


Camera = PerspectiveCamera | OrthographicCamera
MAX_RASTER_PIXELS_PER_CHUNK = 1_000_000


@dataclass
class GBuffer:
    depth: np.ndarray
    mesh_index: np.ndarray
    world_position: np.ndarray
    normal: np.ndarray

    @property
    def hit_mask(self) -> np.ndarray:
        return self.mesh_index >= 0


def rasterize_gbuffer(
    camera: PerspectiveCamera,
    meshes: tuple[Mesh, ...],
    visible_meshes: np.ndarray,
) -> GBuffer:
    height = camera.height
    width = camera.width
    depth = np.full((height, width), np.inf, dtype=np.float64)
    mesh_index = np.full((height, width), -1, dtype=np.int32)
    world_position = np.zeros((height, width, 3), dtype=np.float64)
    normal = np.zeros((height, width, 3), dtype=np.float64)

    for index, mesh in enumerate(meshes):
        if not bool(visible_meshes[index]):
            continue
        cull_backface = not mesh.double_sided
        for fragments in iter_mesh_fragments(camera, mesh, cull_backface):
            pixel_y, pixel_x, fragment_depth, position, fragment_normal = fragments
            old_depth = depth[pixel_y, pixel_x]
            update = fragment_depth < old_depth
            if not np.any(update):
                continue
            ys = pixel_y[update]
            xs = pixel_x[update]
            depth[ys, xs] = fragment_depth[update]
            mesh_index[ys, xs] = index
            world_position[ys, xs] = position[update]
            normal[ys, xs] = fragment_normal[update]

    return GBuffer(
        depth=depth,
        mesh_index=mesh_index,
        world_position=world_position,
        normal=normal,
    )


def rasterize_depth_map(camera: Camera, meshes: tuple[Mesh, ...]) -> np.ndarray:
    """Rasterize all scene meshes into a light depth map.

    Shadow casters are deliberately not filtered by camera frustum visibility.
    Shadow-map triangles are treated as two-sided so thin and open caster
    geometry still contributes depth.
    """
    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    for mesh in meshes:
        for fragments in iter_mesh_fragments(camera, mesh, False):
            pixel_y, pixel_x, fragment_depth, _, _ = fragments
            old_depth = depth[pixel_y, pixel_x]
            update = fragment_depth < old_depth
            if not np.any(update):
                continue
            depth[pixel_y[update], pixel_x[update]] = fragment_depth[update]
    return depth


def mesh_frustum_visibility(
    camera: PerspectiveCamera, meshes: tuple[Mesh, ...]
) -> np.ndarray:
    return np.asarray(
        [camera.mesh_in_frustum(mesh.center, mesh.radius) for mesh in meshes],
        dtype=bool,
    )


def iter_mesh_fragments(
    camera: Camera,
    mesh: Mesh,
    cull_backface: bool,
) -> Iterator[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Yield covered fragment arrays for one mesh.

    Each yielded tuple contains pixel y/x indices, depth, perspective-correct
    world positions and normalized normals. The clipping and triangulation are
    performed before any pixel test, so interpolation never uses vertices that
    lie behind the near plane.
    """
    for face in mesh.indices:
        world = mesh.vertices[face]
        vertex_normals = mesh.normals[face]
        face_cross = np.cross(world[1] - world[0], world[2] - world[0])
        cross_length = float(np.linalg.norm(face_cross))
        if cross_length <= 1.0e-12:
            continue
        if cull_backface:
            facing = float(np.dot(face_cross, camera.eye - world[0]))
            if facing <= 0.0:
                continue
        fallback_normal = face_cross / cross_length

        view = camera.to_view(world)
        clipped_view, clipped_world, clipped_normals = clip_polygon_near(
            view, world, vertex_normals, camera.near
        )
        if clipped_view.shape[0] < 3:
            continue
        clipped_view, clipped_world, clipped_normals = clip_polygon_far(
            clipped_view, clipped_world, clipped_normals, camera.far
        )
        if clipped_view.shape[0] < 3:
            continue

        screen_x, screen_y, inv_depth = camera.project(clipped_view)
        for vertex in range(1, clipped_view.shape[0] - 1):
            triangle_indices = np.asarray([0, vertex, vertex + 1])
            px = screen_x[triangle_indices]
            py = screen_y[triangle_indices]
            area = _edge(px[0], py[0], px[1], py[1], px[2], py[2])
            if abs(float(area)) <= 1.0e-12:
                continue

            width = camera.width
            height = camera.height
            x0_float = max(0.0, float(np.floor(np.min(px))))
            x1_float = min(float(width - 1), float(np.ceil(np.max(px))))
            y0_float = max(0.0, float(np.floor(np.min(py))))
            y1_float = min(float(height - 1), float(np.ceil(np.max(py))))
            if x1_float < x0_float or y1_float < y0_float:
                continue
            x0, x1 = int(x0_float), int(x1_float)
            y0, y1 = int(y0_float), int(y1_float)
            span_width = x1 - x0 + 1
            rows_per_chunk = max(
                1, MAX_RASTER_PIXELS_PER_CHUNK // max(span_width, 1)
            )
            xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5

            for row_start in range(y0, y1 + 1, rows_per_chunk):
                row_end = min(y1 + 1, row_start + rows_per_chunk)
                ys = np.arange(row_start, row_end, dtype=np.float64) + 0.5
                sample_x = xs[None, :]
                sample_y = ys[:, None]
                edge0 = _edge(
                    px[1], py[1], px[2], py[2], sample_x, sample_y
                )
                edge1 = _edge(
                    px[2], py[2], px[0], py[0], sample_x, sample_y
                )
                edge2 = _edge(
                    px[0], py[0], px[1], py[1], sample_x, sample_y
                )
                bary0 = edge0 / area
                bary1 = edge1 / area
                bary2 = edge2 / area
                inside = (
                    (bary0 >= -1.0e-10)
                    & (bary1 >= -1.0e-10)
                    & (bary2 >= -1.0e-10)
                )
                if not np.any(inside):
                    continue

                local_y, local_x = np.nonzero(inside)
                weights = np.stack(
                    (
                        bary0[local_y, local_x],
                        bary1[local_y, local_x],
                        bary2[local_y, local_x],
                    ),
                    axis=1,
                )
                tri_inv_depth = inv_depth[triangle_indices]
                denominator = weights @ tri_inv_depth
                valid = denominator > EPS
                if not np.any(valid):
                    continue
                weights = weights[valid]
                denominator = denominator[valid]
                local_x = local_x[valid]
                local_y = local_y[valid]

                triangle_world = clipped_world[triangle_indices]
                triangle_normals = clipped_normals[triangle_indices]
                weighted_depth = weights * tri_inv_depth[None, :]
                position = (
                    weighted_depth @ triangle_world
                ) / denominator[:, None]
                interpolated_normal = (
                    weighted_depth @ triangle_normals
                ) / denominator[:, None]
                normal_length = np.linalg.norm(
                    interpolated_normal, axis=1, keepdims=True
                )
                interpolated_normal = np.where(
                    normal_length > 1.0e-10,
                    interpolated_normal / np.maximum(normal_length, EPS),
                    fallback_normal[None, :],
                )
                pixel_x = x0 + local_x
                pixel_y = row_start + local_y
                fragment_depth = camera.depth_metric(position)
                yield (
                    pixel_y,
                    pixel_x,
                    fragment_depth,
                    position,
                    interpolated_normal,
                )


def _edge(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    px: np.ndarray | float,
    py: np.ndarray | float,
) -> np.ndarray:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
