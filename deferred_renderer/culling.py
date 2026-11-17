from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .math3d import reconstruction_from_ndc
from .scene import Light, RenderConfig, Scene


@dataclass
class CullingResult:
    tile_records: list[dict[str, Any]]
    packed_lights: np.ndarray
    tiles_x: int
    tiles_y: int
    words_per_tile: int
    stats: dict[str, Any]

    @property
    def packed_bytes(self) -> bytes:
        return np.asarray(self.packed_lights, dtype="<u4").tobytes(order="C")


@dataclass(frozen=True)
class TileFrustum:
    vertices: np.ndarray
    planes: tuple[tuple[np.ndarray, float], ...]
    triangles: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    center: np.ndarray


def _plane_from_polygon(
    polygon: np.ndarray,
    inside_point: np.ndarray,
) -> tuple[np.ndarray, float]:
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("degenerate tile frustum face")
    normal /= length
    offset = float(np.dot(normal, polygon[0]))
    if float(np.dot(normal, inside_point) - offset) < 0.0:
        normal = -normal
        offset = -offset
    return normal, offset


def build_tile_frustum(
    ndc_x_min: float,
    ndc_x_max: float,
    ndc_y_min: float,
    ndc_y_max: float,
    depth_min: float,
    depth_max: float,
    inv_projection: np.ndarray,
    inv_view: np.ndarray,
) -> TileFrustum:
    """Build the depth-bounded visible frustum for one screen tile."""

    if depth_max - depth_min <= 1e-6:
        padding = max(1e-4, abs(depth_min) * 1e-5)
        depth_min -= padding
        depth_max += padding

    xs = (ndc_x_min, ndc_x_max)
    ys = (ndc_y_min, ndc_y_max)

    def point(x: float, y: float, depth: float) -> np.ndarray:
        return reconstruction_from_ndc(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(depth, dtype=np.float64),
            inv_projection,
            inv_view,
        )

    p00 = point(xs[0], ys[0], depth_min)
    p10 = point(xs[1], ys[0], depth_min)
    p11 = point(xs[1], ys[1], depth_min)
    p01 = point(xs[0], ys[1], depth_min)
    q00 = point(xs[0], ys[0], depth_max)
    q10 = point(xs[1], ys[0], depth_max)
    q11 = point(xs[1], ys[1], depth_max)
    q01 = point(xs[0], ys[1], depth_max)
    vertices = np.stack([p00, p10, p11, p01, q00, q10, q11, q01], axis=0)
    center = vertices.mean(axis=0)

    polygons = [
        np.stack([p00, p10, p11, p01]),
        np.stack([q00, q01, q11, q10]),
        np.stack([p00, p01, q01, q00]),
        np.stack([p10, q10, q11, p11]),
        np.stack([p00, q00, q10, p10]),
        np.stack([p01, p11, q11, q01]),
    ]
    planes = tuple(_plane_from_polygon(polygon, center) for polygon in polygons)
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for polygon in polygons:
        triangles.append((polygon[0], polygon[1], polygon[2]))
        triangles.append((polygon[0], polygon[2], polygon[3]))
    return TileFrustum(
        vertices=vertices,
        planes=planes,
        triangles=tuple(triangles),
        center=center,
    )


def _closest_point_on_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denominator = d1 - d3
        if abs(denominator) <= 1e-20:
            return a
        return a + (d1 / denominator) * ab

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denominator = d2 - d6
        if abs(denominator) <= 1e-20:
            return a
        return a + (d2 / denominator) * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denominator = (d4 - d3) + (d5 - d6)
        if abs(denominator) <= 1e-20:
            return b
        return b + ((d4 - d3) / denominator) * (c - b)

    denominator = va + vb + vc
    if abs(denominator) <= 1e-20:
        return a
    v = vb / denominator
    w = vc / denominator
    return a + ab * v + ac * w


def sphere_intersects_tile(
    sphere_center: np.ndarray,
    sphere_radius: float,
    frustum: TileFrustum,
) -> bool:
    """Exact sphere/convex-tile test (inside, face, edge or vertex contact)."""

    signed_distances = [
        float(np.dot(normal, sphere_center) - offset)
        for normal, offset in frustum.planes
    ]
    if all(distance >= -1e-7 for distance in signed_distances):
        return True

    radius_sq = sphere_radius * sphere_radius
    for triangle in frustum.triangles:
        closest = _closest_point_on_triangle(sphere_center, *triangle)
        delta = closest - sphere_center
        if float(np.dot(delta, delta)) <= radius_sq + 1e-10:
            return True
    return False


def coc_radius_array(
    linear_depth: np.ndarray,
    render: RenderConfig,
) -> np.ndarray:
    depth = np.asarray(linear_depth, dtype=np.float64)
    if render.max_blur_radius <= 0.0:
        return np.zeros_like(depth)
    focal_length_m = render.focal_length_mm * 0.001
    aperture_diameter = focal_length_m / render.aperture
    denominator = np.maximum(
        depth * abs(render.focus_distance - focal_length_m),
        1e-6,
    )
    coc_diameter_m = (
        aperture_diameter
        * focal_length_m
        * np.abs(depth - render.focus_distance)
        / denominator
    )
    pixel_size_m = render.sensor_width_mm * 0.001 / render.width
    radii = 0.5 * coc_diameter_m / pixel_size_m
    radii = np.minimum(np.maximum(radii, 0.0), render.max_blur_radius)
    radii = np.where(depth <= 0.0, render.max_blur_radius * 0.75, radii)
    return radii


def coc_statistics(linear_depth: np.ndarray, render: RenderConfig) -> dict[str, Any]:
    depth = np.asarray(linear_depth, dtype=np.float64)
    radii = coc_radius_array(depth, render)
    valid = depth > 0.0
    max_radius = float(render.max_blur_radius)
    bin_count = 16
    if max_radius > 0.0:
        edges = np.linspace(0.0, max_radius, bin_count + 1, dtype=np.float64)
        all_counts, _ = np.histogram(radii, bins=edges)
        geometry_counts, _ = np.histogram(radii[valid], bins=edges)
    else:
        edges = np.zeros(bin_count + 1, dtype=np.float64)
        all_counts = np.zeros(bin_count, dtype=np.int64)
        geometry_counts = np.zeros(bin_count, dtype=np.int64)
    foreground = valid & (depth < render.focus_distance)
    background = (~valid) | (depth >= render.focus_distance)
    return {
        "formula": (
            "r_px = 0.5 * (f/N) * f * |z-focus| / "
            "(max(z*|focus-f|, 1e-6)) / (sensor_width_px); "
            "background (z<=0) uses 0.75*max_blur_radius"
        ),
        "focus_distance": render.focus_distance,
        "aperture_f_number": render.aperture,
        "focal_length_mm": render.focal_length_mm,
        "sensor_width_mm": render.sensor_width_mm,
        "max_blur_radius_px": max_radius,
        "bin_edges_px": [float(value) for value in edges],
        "all_pixel_counts": [int(value) for value in all_counts],
        "geometry_pixel_counts": [int(value) for value in geometry_counts],
        "valid_geometry_pixels": int(valid.sum()),
        "foreground_pixels": int(foreground.sum()),
        "background_pixels": int(background.sum()),
        "valid_radius_min_px": float(radii[valid].min()) if valid.any() else 0.0,
        "valid_radius_mean_px": float(radii[valid].mean()) if valid.any() else 0.0,
        "valid_radius_max_px": float(radii[valid].max()) if valid.any() else 0.0,
    }


def build_tile_lights(scene: Scene, linear_depth: np.ndarray) -> CullingResult:
    depth = np.asarray(linear_depth, dtype=np.float64)
    height, width = depth.shape
    render = scene.render
    tile_size = render.tile_size
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    tile_count = tiles_x * tiles_y
    words_per_tile = math.ceil((len(scene.lights) + 1) / 4)
    packed = np.zeros((tile_count, words_per_tile * 4), dtype=np.uint32)
    records: list[dict[str, Any]] = []
    counts: list[int] = []
    inv_projection = scene.inv_projection
    inv_view = scene.camera.inv_view

    for tile_y in range(tiles_y):
        y0 = tile_y * tile_size
        y1 = min(y0 + tile_size, height)
        for tile_x in range(tiles_x):
            x0 = tile_x * tile_size
            x1 = min(x0 + tile_size, width)
            tile_depth = depth[y0:y1, x0:x1]
            valid_depth = tile_depth[np.isfinite(tile_depth) & (tile_depth > 0.0)]
            light_indices: list[int] = []
            depth_min = 0.0
            depth_max = 0.0
            if valid_depth.size:
                depth_min = float(valid_depth.min())
                depth_max = float(valid_depth.max())
                frustum = build_tile_frustum(
                    (x0 / width) * 2.0 - 1.0,
                    (x1 / width) * 2.0 - 1.0,
                    (y0 / height) * 2.0 - 1.0,
                    (y1 / height) * 2.0 - 1.0,
                    depth_min,
                    depth_max,
                    inv_projection,
                    inv_view,
                )
                for light_index, light in enumerate(scene.lights):
                    if sphere_intersects_tile(
                        np.asarray(light.position, dtype=np.float64),
                        light.radius,
                        frustum,
                    ):
                        light_indices.append(light_index)
            tile_index = tile_y * tiles_x + tile_x
            packed[tile_index, 0] = len(light_indices)
            if light_indices:
                packed[tile_index, 1 : 1 + len(light_indices)] = light_indices
            counts.append(len(light_indices))
            records.append(
                {
                    "tile": tile_index,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                    "lights": light_indices,
                }
            )

    histogram: dict[str, int] = {}
    for count in counts:
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1
    stats = {
        "tile_size": tile_size,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tile_count": tile_count,
        "words_per_tile": words_per_tile,
        "average_lights_per_tile": (
            float(sum(counts) / len(counts)) if counts else 0.0
        ),
        "max_lights_in_tile": int(max(counts) if counts else 0),
        "total_light_entries": int(sum(counts)),
        "light_count_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "coordinate_origin": "bottom_left",
        "cull_test": "exact sphere vs depth-bounded tile convex frustum",
    }
    return CullingResult(
        tile_records=records,
        packed_lights=packed,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        words_per_tile=words_per_tile,
        stats=stats,
    )

