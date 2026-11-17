"""Screen-space triangle coverage and deterministic line overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scene import Material


COVERAGE_EPS = 1e-10


@dataclass
class ProjectedTriangle:
    screen: np.ndarray
    depth: np.ndarray
    inv_w: np.ndarray
    world_pos: np.ndarray
    world_normal: np.ndarray
    uv: np.ndarray
    tangent: np.ndarray
    material: "Material"
    camera_depth: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DepthTriangle:
    screen: np.ndarray
    depth: np.ndarray
    bbox: tuple[int, int, int, int]


def screen_bbox(screen: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x_min = max(0, int(math.floor(float(np.min(screen[:, 0])))))
    y_min = max(0, int(math.floor(float(np.min(screen[:, 1])))))
    x_max = min(width - 1, int(math.ceil(float(np.max(screen[:, 0])))))
    y_max = min(height - 1, int(math.ceil(float(np.max(screen[:, 1])))))
    return x_min, y_min, x_max, y_max


def bbox_intersects(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int) -> bool:
    return x_min <= x_max and y_min <= y_max and x_max >= 0 and y_max >= 0 and x_min < width and y_min < height


def _edge(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def triangle_coverage(
    screen: np.ndarray,
    depth: np.ndarray,
    inv_w: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return covered pixel indices, interpolated depth and perspective weights."""
    x_min, y_min, x_max, y_max = bbox
    if x_min > x_max or y_min > y_max:
        empty_pixels = np.empty((0, 2), dtype=np.int64)
        return empty_pixels, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    xs = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    ys = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    v0, v1, v2 = screen[0], screen[1], screen[2]
    area = _edge(v0, v1, v2[0], v2[1])
    if abs(float(area)) <= COVERAGE_EPS:
        empty_pixels = np.empty((0, 2), dtype=np.int64)
        return empty_pixels, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    e0 = _edge(v1, v2, grid_x, grid_y)
    e1 = _edge(v2, v0, grid_x, grid_y)
    e2 = _edge(v0, v1, grid_x, grid_y)
    if area > 0.0:
        inside = (e0 >= -COVERAGE_EPS) & (e1 >= -COVERAGE_EPS) & (e2 >= -COVERAGE_EPS)
    else:
        inside = (e0 <= COVERAGE_EPS) & (e1 <= COVERAGE_EPS) & (e2 <= COVERAGE_EPS)

    b0 = e0 / area
    b1 = e1 / area
    b2 = e2 / area
    interpolated_depth = b0 * depth[0] + b1 * depth[1] + b2 * depth[2]
    inside &= (interpolated_depth >= 0.0) & (interpolated_depth <= 1.0)
    local_y, local_x = np.nonzero(inside)
    if local_x.size == 0:
        empty_pixels = np.empty((0, 2), dtype=np.int64)
        return empty_pixels, np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    pixel_x = local_x + x_min
    pixel_y = local_y + y_min
    selected_b0 = b0[local_y, local_x]
    selected_b1 = b1[local_y, local_x]
    selected_b2 = b2[local_y, local_x]
    denominator = (
        selected_b0 * inv_w[0] + selected_b1 * inv_w[1] + selected_b2 * inv_w[2]
    )
    safe_denominator = np.where(np.abs(denominator) > 1e-14, denominator, 1.0)
    weights = np.stack(
        [
            selected_b0 * inv_w[0] / safe_denominator,
            selected_b1 * inv_w[1] / safe_denominator,
            selected_b2 * inv_w[2] / safe_denominator,
        ],
        axis=1,
    )
    pixels = np.stack([pixel_y, pixel_x], axis=1)
    selected_depth = interpolated_depth[local_y, local_x]
    return pixels, selected_depth, weights


def depth_only_coverage(
    triangle: DepthTriangle,
) -> tuple[np.ndarray, np.ndarray]:
    pixels, depth, _ = triangle_coverage(
        triangle.screen,
        triangle.depth,
        np.ones(3, dtype=np.float64),
        triangle.bbox,
    )
    return pixels, depth


def rasterize_depth_triangles(
    triangles: list[DepthTriangle],
    depth_buffer: np.ndarray,
    less_or_equal: bool = False,
) -> int:
    if not triangles:
        return 0
    height, width = depth_buffer.shape
    rasterized = 0
    for triangle in triangles:
        pixels, depths = depth_only_coverage(triangle)
        if pixels.size == 0:
            continue
        y = pixels[:, 0]
        x = pixels[:, 1]
        if less_or_equal:
            mask = depths <= depth_buffer[y, x] + 1e-10
        else:
            mask = depths < depth_buffer[y, x] - 1e-12
        if not np.any(mask):
            continue
        rasterized += 1
        selected_y = y[mask]
        selected_x = x[mask]
        selected_depth = depths[mask]
        np.minimum.at(depth_buffer, (selected_y, selected_x), selected_depth)
    return rasterized


def _clip_line_to_viewport(
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, width: int, height: int
) -> tuple[float, float, float, float, float, float] | None:
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, width - 1.0 - x0),
        (-dy, y0),
        (dy, height - 1.0 - y0),
    ):
        if abs(p) <= 1e-15:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    ax = x0 + t0 * dx
    ay = y0 + t0 * dy
    az = z0 + t0 * dz
    bx = x0 + t1 * dx
    by = y0 + t1 * dy
    bz = z0 + t1 * dz
    return ax, ay, az, bx, by, bz


def draw_depth_tested_line(
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    color: np.ndarray,
    covered: np.ndarray,
) -> int:
    height, width = depth_buffer.shape
    clipped = _clip_line_to_viewport(*start, *end, width, height)
    if clipped is None:
        return 0
    x0, y0, z0, x1, y1, z1 = clipped
    steps = max(1, int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))) + 1)
    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    xs = np.floor(x0 + (x1 - x0) * t + 0.5).astype(np.int64)
    ys = np.floor(y0 + (y1 - y0) * t + 0.5).astype(np.int64)
    zs = z0 + (z1 - z0) * t
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    valid &= (zs >= 0.0) & (zs <= 1.0)
    if not np.any(valid):
        return 0
    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]
    visible = zs <= depth_buffer[ys, xs] + 2e-4
    if not np.any(visible):
        return 0
    ys = ys[visible]
    xs = xs[visible]
    color_buffer[ys, xs] = color
    before = int(np.count_nonzero(covered))
    covered[ys, xs] = True
    return int(np.count_nonzero(covered)) - before
