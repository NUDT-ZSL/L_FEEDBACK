from __future__ import annotations

from typing import Any

import numpy as np

from .math3d import F64, safe_normalize
from .scene import Scene


RASTER_ROW_CHUNK = 32


def _clip_plane(
    polygon: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    signed_distance: Any,
    plane_name: str,
    issues: dict[str, int],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not polygon:
        return polygon
    output: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    previous = polygon[-1]
    previous_distance = float(signed_distance(previous[0]))
    previous_inside = previous_distance >= 0.0
    for current in polygon:
        current_distance = float(signed_distance(current[0]))
        current_inside = current_distance >= 0.0
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) <= 1e-15:
                issues["degenerate_clip_count"] += 1
            else:
                t = previous_distance / denominator
                t = min(1.0, max(0.0, t))
                view = previous[0] + t * (current[0] - previous[0])
                world = previous[1] + t * (current[1] - previous[1])
                normal = previous[2] + t * (current[2] - previous[2])
                output.append((view, world, normal))
        if current_inside:
            output.append(current)
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    if not output:
        issues[f"clipped_{plane_name}_triangles"] += 1
    return output


def _compute_area(
    x0: float, y0: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def _bounds(
    x_values: np.ndarray, y_values: np.ndarray, width: int, height: int
) -> tuple[int, int, int, int] | None:
    minimum_x = float(np.min(x_values))
    maximum_x = float(np.max(x_values))
    minimum_y = float(np.min(y_values))
    maximum_y = float(np.max(y_values))
    if not all(
        np.isfinite(value)
        for value in (minimum_x, maximum_x, minimum_y, maximum_y)
    ):
        return None
    if maximum_x < 0.0 or maximum_y < 0.0:
        return None
    if minimum_x >= width or minimum_y >= height:
        return None
    x0 = max(0, int(np.floor(minimum_x)) - 1)
    y0 = max(0, int(np.floor(minimum_y)) - 1)
    x1 = min(width, int(np.ceil(maximum_x)) + 2)
    y1 = min(height, int(np.ceil(maximum_y)) + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def rasterize_gbuffer(
    scene: Scene,
    world_vertices: np.ndarray,
    normal_vertices: np.ndarray,
    view_vertices: np.ndarray,
    issues: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = scene.width
    height = scene.height
    depth_buffer = np.full((height, width), np.inf, dtype=F64)
    world_buffer = np.zeros((height, width, 3), dtype=F64)
    normal_buffer = np.zeros((height, width, 3), dtype=F64)
    material_buffer = np.full((height, width), -1, dtype=np.int32)

    triangles = scene.triangles
    triangle_materials = scene.triangle_materials
    total_triangles = triangles.shape[0]
    for triangle_index in range(total_triangles):
        indices = triangles[triangle_index]
        world_triangle = world_vertices[indices]
        normal_triangle = normal_vertices[indices]
        view_triangle = view_vertices[indices]
        if not (
            np.all(np.isfinite(world_triangle))
            and np.all(np.isfinite(normal_triangle))
            and np.all(np.isfinite(view_triangle))
        ):
            issues["invalid_triangle_count"] += 1
            continue
        face_cross = np.cross(
            world_triangle[1] - world_triangle[0],
            world_triangle[2] - world_triangle[0],
        )
        face_length = float(np.linalg.norm(face_cross))
        if not np.isfinite(face_length) or face_length <= 1e-15:
            issues["degenerate_triangle_count"] += 1
            continue
        face_normal = face_cross / face_length
        polygon = [
            (view_triangle[corner], world_triangle[corner], normal_triangle[corner])
            for corner in range(3)
        ]
        polygon = _clip_plane(
            polygon, lambda view: -view[2] - scene.camera.near, "near", issues
        )
        if len(polygon) < 3:
            continue
        polygon = _clip_plane(
            polygon, lambda view: scene.camera.far + view[2], "far", issues
        )
        if len(polygon) < 3:
            continue
        for fan_index in range(1, len(polygon) - 1):
            vertices = (polygon[0], polygon[fan_index], polygon[fan_index + 1])
            view_points = np.stack([item[0] for item in vertices], axis=0)
            world_points = np.stack([item[1] for item in vertices], axis=0)
            normal_points = np.stack([item[2] for item in vertices], axis=0)
            depths = -view_points[:, 2]
            if np.any(depths <= scene.camera.near - 1e-9):
                issues["invalid_projected_triangle_count"] += 1
                continue
            focal = 1.0 / np.tan(scene.camera.fov_y_radians * 0.5)
            aspect = width / height
            ndc_x = (focal / aspect) * view_points[:, 0] / depths
            ndc_y = focal * view_points[:, 1] / depths
            screen_x = (ndc_x + 1.0) * 0.5 * width
            screen_y = (1.0 - ndc_y) * 0.5 * height
            if not (
                np.all(np.isfinite(screen_x))
                and np.all(np.isfinite(screen_y))
                and np.all(np.isfinite(depths))
            ):
                issues["invalid_projected_triangle_count"] += 1
                continue
            bounds = _bounds(screen_x, screen_y, width, height)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            area = _compute_area(
                float(screen_x[0]),
                float(screen_y[0]),
                float(screen_x[1]),
                float(screen_y[1]),
                float(screen_x[2]),
                float(screen_y[2]),
            )
            if abs(area) <= 1e-12:
                issues["degenerate_triangle_count"] += 1
                continue
            inverse_area = 1.0 / area
            material_index = int(triangle_materials[triangle_index])

            for row_start in range(y0, y1, RASTER_ROW_CHUNK):
                row_end = min(y1, row_start + RASTER_ROW_CHUNK)
                pixel_x = np.arange(x0, x1, dtype=F64)[None, :] + 0.5
                pixel_y = (
                    np.arange(row_start, row_end, dtype=F64)[:, None] + 0.5
                )
                w0 = (
                    (screen_x[1] - pixel_x) * (screen_y[2] - pixel_y)
                    - (screen_y[1] - pixel_y) * (screen_x[2] - pixel_x)
                ) * inverse_area
                w1 = (
                    (screen_x[2] - pixel_x) * (screen_y[0] - pixel_y)
                    - (screen_y[2] - pixel_y) * (screen_x[0] - pixel_x)
                ) * inverse_area
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
                if not np.any(inside):
                    continue
                inverse_w = (
                    w0 / depths[0] + w1 / depths[1] + w2 / depths[2]
                )
                valid = inside & np.isfinite(inverse_w) & (inverse_w > 0.0)
                if not np.any(valid):
                    issues["invalid_raster_pixel_count"] += int(np.count_nonzero(inside))
                    continue
                depth = np.zeros_like(inverse_w)
                np.divide(1.0, inverse_w, out=depth, where=valid)
                perspective0 = np.divide(
                    w0 / depths[0],
                    inverse_w,
                    out=np.zeros_like(w0),
                    where=valid,
                )
                perspective1 = np.divide(
                    w1 / depths[1],
                    inverse_w,
                    out=np.zeros_like(w1),
                    where=valid,
                )
                perspective2 = np.divide(
                    w2 / depths[2],
                    inverse_w,
                    out=np.zeros_like(w2),
                    where=valid,
                )
                world_pixel = (
                    perspective0[..., None] * world_points[0][None, None, :]
                    + perspective1[..., None] * world_points[1][None, None, :]
                    + perspective2[..., None] * world_points[2][None, None, :]
                )
                normal_pixel = (
                    perspective0[..., None] * normal_points[0][None, None, :]
                    + perspective1[..., None] * normal_points[1][None, None, :]
                    + perspective2[..., None] * normal_points[2][None, None, :]
                )
                normal_length = np.linalg.norm(
                    normal_pixel, axis=-1, keepdims=True
                )
                normal_ok = (
                    np.isfinite(normal_length[..., 0])
                    & (normal_length[..., 0] > 1e-12)
                )
                if np.any(valid & ~normal_ok):
                    issues["invalid_normal_pixel_count"] += int(
                        np.count_nonzero(valid & ~normal_ok)
                    )
                normal_pixel = np.divide(
                    normal_pixel,
                    normal_length,
                    out=np.broadcast_to(
                        face_normal, normal_pixel.shape
                    ).copy(),
                    where=normal_ok[..., None] & valid[..., None],
                )

                depth_slice = depth_buffer[row_start:row_end, x0:x1]
                update = valid & (depth < depth_slice)
                update_count = int(np.count_nonzero(update))
                if update_count == 0:
                    continue
                update_3 = update[..., None]
                world_slice = world_buffer[row_start:row_end, x0:x1]
                normal_slice = normal_buffer[row_start:row_end, x0:x1]
                material_slice = material_buffer[row_start:row_end, x0:x1]
                depth_slice[...] = np.where(update, depth, depth_slice)
                world_slice[...] = np.where(update_3, world_pixel, world_slice)
                normal_slice[...] = np.where(update_3, normal_pixel, normal_slice)
                material_slice[...] = np.where(
                    update, material_index, material_slice
                )
                issues["rasterized_pixel_count"] += update_count
    return depth_buffer, world_buffer, normal_buffer, material_buffer


def rasterize_ortho_depth(
    screen_x: np.ndarray,
    screen_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    resolution: int,
    issues: dict[str, int],
) -> np.ndarray:
    depth_map = np.full((resolution, resolution), np.inf, dtype=F64)
    for indices in triangles:
        tx = screen_x[indices]
        ty = screen_y[indices]
        tz = depth[indices]
        if not (
            np.all(np.isfinite(tx))
            and np.all(np.isfinite(ty))
            and np.all(np.isfinite(tz))
        ):
            issues["shadow_invalid_triangle_count"] += 1
            continue
        bounds = _bounds(tx, ty, resolution, resolution)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        area = _compute_area(
            float(tx[0]),
            float(ty[0]),
            float(tx[1]),
            float(ty[1]),
            float(tx[2]),
            float(ty[2]),
        )
        if abs(area) <= 1e-12:
            issues["shadow_degenerate_triangle_count"] += 1
            continue
        inverse_area = 1.0 / area
        for row_start in range(y0, y1, RASTER_ROW_CHUNK):
            row_end = min(y1, row_start + RASTER_ROW_CHUNK)
            pixel_x = np.arange(x0, x1, dtype=F64)[None, :] + 0.5
            pixel_y = (
                np.arange(row_start, row_end, dtype=F64)[:, None] + 0.5
            )
            w0 = (
                (tx[1] - pixel_x) * (ty[2] - pixel_y)
                - (ty[1] - pixel_y) * (tx[2] - pixel_x)
            ) * inverse_area
            w1 = (
                (tx[2] - pixel_x) * (ty[0] - pixel_y)
                - (ty[2] - pixel_y) * (tx[0] - pixel_x)
            ) * inverse_area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not np.any(inside):
                continue
            pixel_depth = w0 * tz[0] + w1 * tz[1] + w2 * tz[2]
            valid = (
                inside
                & np.isfinite(pixel_depth)
                & (pixel_depth >= 0.0)
                & (pixel_depth <= 1.0)
            )
            if not np.any(valid):
                continue
            depth_slice = depth_map[row_start:row_end, x0:x1]
            update = valid & (pixel_depth < depth_slice)
            update_count = int(np.count_nonzero(update))
            if update_count:
                depth_slice[...] = np.where(update, pixel_depth, depth_slice)
                issues["shadow_rasterized_pixel_count"] += update_count
    return depth_map
