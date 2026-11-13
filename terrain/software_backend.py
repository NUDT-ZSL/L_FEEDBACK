"""Deterministic NumPy software rasteriser (single-threaded, float32 framebuffer).

The backend receives fully built meshes plus per-vertex colors.  Geometry is
transformed in float64 and quantised to float32 only at rasterisation time.
Back-face culling uses geometric cross products, depth uses a strict less-than
test so equal-depth ties keep the first (deterministically ordered) writer.
"""

from __future__ import annotations

import math

import numpy as np

from .camera import look_at, perspective
from .config import PipelineConfig
from .mesh import TileMesh

BACKEND_NAME = "numpy-software"


def _clip_near(
    camera_vertices: np.ndarray, triangles: np.ndarray, near: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clip triangles against the camera-space near plane (z = -near)."""
    clipped_vertices = list(camera_vertices)
    clipped_triangles: list[tuple[int, int, int]] = []

    def interpolate(vertex_a, vertex_b):
        distance_a = -vertex_a[2] - near
        distance_b = -vertex_b[2] - near
        factor = distance_a / (distance_a - distance_b)
        new_vertex = vertex_a + factor * (vertex_b - vertex_a)
        clipped_vertices.append(new_vertex)
        return len(clipped_vertices) - 1

    for triangle in triangles:
        ids = list(triangle)
        output: list[int] = []
        for index in range(3):
            current = ids[index]
            previous = ids[index - 1]
            current_inside = -clipped_vertices[current][2] >= near
            previous_inside = -clipped_vertices[previous][2] >= near
            if current_inside:
                if not previous_inside:
                    output.append(interpolate(clipped_vertices[previous], clipped_vertices[current]))
                output.append(current)
            elif previous_inside:
                output.append(interpolate(clipped_vertices[previous], clipped_vertices[current]))
        if len(output) >= 3:
            for index in range(1, len(output) - 1):
                clipped_triangles.append((output[0], output[index], output[index + 1]))
    return np.asarray(clipped_vertices, dtype=np.float64), np.asarray(
        clipped_triangles, dtype=np.int64
    )


def _transform_mesh(mesh: TileMesh, config: PipelineConfig):
    view = look_at(config.camera)
    projection = perspective(config.camera)
    homogeneous = np.concatenate(
        [mesh.vertices, np.ones((mesh.vertices.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera = (view @ homogeneous.T).T
    camera, triangles = _clip_near(camera, mesh.triangles, config.camera.near)
    if triangles.size == 0:
        return None
    clip = (projection @ camera.T).T
    w = clip[:, 3:4]
    ndc = clip[:, :3] / w
    width = config.image_width
    height = config.image_height
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * width
    screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    # Linear, positive view-space depth keeps the float32 buffer precise;
    # NDC z is compressed into a tiny range and would cause false occlusion.
    linear_depth = (-camera[:, 2]).astype(np.float32)
    screen = np.stack([screen_x, screen_y, linear_depth], axis=1).astype(
        np.float32
    )
    camera32 = camera[:, :3].astype(np.float32)
    return screen, camera32, triangles


SAMPLE_GRID = 2
SAMPLE_OFFSETS = tuple(
    ((index_x + 0.5) / SAMPLE_GRID, (index_y + 0.5) / SAMPLE_GRID)
    for index_y in range(SAMPLE_GRID)
    for index_x in range(SAMPLE_GRID)
)


def _is_back_facing(points: np.ndarray) -> bool:
    edge_a = points[1] - points[0]
    edge_b = points[2] - points[0]
    cross_z = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
    return cross_z >= 0.0


def render_scene(
    meshes: list[TileMesh],
    vertex_colors: list[np.ndarray],
    config: PipelineConfig,
) -> np.ndarray:
    """Render all tiles and return an ``(H, W)`` uint8 color framebuffer."""
    width = config.image_width
    height = config.image_height
    depth = np.full((height, width), np.inf, dtype=np.float32)
    color = np.zeros((height, width), dtype=np.uint8)

    ordered = sorted(meshes, key=lambda mesh: (mesh.key.level, mesh.key.iz, mesh.key.ix))
    color_lookup = {mesh.key: values for mesh, values in zip(meshes, vertex_colors)}

    for mesh in ordered:
        transformed = _transform_mesh(mesh, config)
        if transformed is None:
            continue
        screen, camera_points, triangles = transformed
        flat_colors = color_lookup[mesh.key]
        for triangle in triangles:
            points = screen[triangle]
            if _is_back_facing(points):
                continue
            # Half-pixel bbox dilation closes sub-pixel seams between
            # adjacent triangles; barycentric + depth tests stay exact.
            min_x = max(int(math.floor(float(points[:, 0].min()) - 0.5)), 0)
            max_x = min(
                int(math.ceil(float(points[:, 0].max()) + 0.5)), width - 1
            )
            min_y = max(int(math.floor(float(points[:, 1].min()) - 0.5)), 0)
            max_y = min(
                int(math.ceil(float(points[:, 1].max()) + 0.5)), height - 1
            )
            if min_x > max_x or min_y > max_y:
                continue
            denominator = (
                (points[1, 1] - points[2, 1]) * (points[0, 0] - points[2, 0])
                + (points[2, 0] - points[1, 0]) * (points[0, 1] - points[2, 1])
            )
            if denominator == 0.0:
                continue
            base_x = np.arange(min_x, max_x + 1, dtype=np.float32)
            base_y = np.arange(min_y, max_y + 1, dtype=np.float32)
            region_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
            region_color = color[min_y : max_y + 1, min_x : max_x + 1]
            for offset_x, offset_y in SAMPLE_OFFSETS:
                pixel_x, pixel_y = np.meshgrid(base_x + offset_x, base_y + offset_y)
                weight_0 = (
                    (points[1, 1] - points[2, 1]) * (pixel_x - points[2, 0])
                    + (points[2, 0] - points[1, 0]) * (pixel_y - points[2, 1])
                ) / denominator
                weight_1 = (
                    (points[2, 1] - points[0, 1]) * (pixel_x - points[2, 0])
                    + (points[0, 0] - points[2, 0]) * (pixel_y - points[2, 1])
                ) / denominator
                weight_2 = 1.0 - weight_0 - weight_1
                inside = (
                    (weight_0 >= -1.0e-6)
                    & (weight_1 >= -1.0e-6)
                    & (weight_2 >= -1.0e-6)
                )
                interpolated_depth = (
                    weight_0 * points[0, 2]
                    + weight_1 * points[1, 2]
                    + weight_2 * points[2, 2]
                )
                interpolated_color = (
                    weight_0 * flat_colors[triangle[0]]
                    + weight_1 * flat_colors[triangle[1]]
                    + weight_2 * flat_colors[triangle[2]]
                )
                update = inside & (interpolated_depth < region_depth)
                if not np.any(update):
                    continue
                region_depth[update] = interpolated_depth[update]
                region_color[update] = np.rint(
                    np.clip(interpolated_color[update], 0.0, 1.0) * 255.0
                ).astype(np.uint8)
    return color
