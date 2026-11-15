"""Deterministic geometry helpers used by rasterization and low-LOD cleanup."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


Point = tuple[float, float]
Rect = tuple[float, float, float, float]


def as_array(points: Iterable[Point]) -> np.ndarray:
    return np.asarray(tuple(points), dtype=np.float64)


def build_segments(points: np.ndarray, closed: bool) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 4), dtype=np.float64)
    starts = points[:-1]
    ends = points[1:]
    segments = np.concatenate((starts, ends), axis=1)
    if closed:
        last = np.concatenate((points[-1:], points[:1]), axis=1)
        segments = np.concatenate((segments, last), axis=0)
    return segments.astype(np.float64, copy=False)


def _cross(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    eps: float,
) -> bool:
    return (
        min(ax, bx) - eps <= px <= max(ax, bx) + eps
        and min(ay, by) - eps <= py <= max(ay, by) + eps
    )


def segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = 1e-9,
) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])
    o1 = _cross(ax, ay, bx, by, cx, cy)
    o2 = _cross(ax, ay, bx, by, dx, dy)
    o3 = _cross(cx, cy, dx, dy, ax, ay)
    o4 = _cross(cx, cy, dx, dy, bx, by)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    if abs(o1) <= eps and _on_segment(cx, cy, ax, ay, bx, by, eps):
        return True
    if abs(o2) <= eps and _on_segment(dx, dy, ax, ay, bx, by, eps):
        return True
    if abs(o3) <= eps and _on_segment(ax, ay, cx, cy, dx, dy, eps):
        return True
    if abs(o4) <= eps and _on_segment(bx, by, cx, cy, dx, dy, eps):
        return True
    return False


def has_self_intersection(points: np.ndarray, *, closed: bool) -> bool:
    """Return True for non-adjacent edge contact or crossing."""

    count = len(points)
    if count < 4:
        return False
    segment_count = count if closed else count - 1
    for first in range(segment_count):
        a = points[first]
        b = points[(first + 1) % count]
        for second in range(first + 1, segment_count):
            adjacent = second == first + 1
            if closed and first == 0 and second == segment_count - 1:
                adjacent = True
            if adjacent:
                continue
            c = points[second]
            d = points[(second + 1) % count]
            if segments_intersect(a, b, c, d):
                return True
    return False


def clip_polygon_to_rect(points: np.ndarray, rect: Rect) -> np.ndarray:
    """Sutherland-Hodgman clipping of one polygon ring to an axis-aligned rect."""

    if len(points) == 0:
        return points
    xmin, ymin, xmax, ymax = rect
    output = [np.asarray(point, dtype=np.float64) for point in points]

    boundaries = (
        ("left", xmin),
        ("right", xmax),
        ("bottom", ymin),
        ("top", ymax),
    )
    for edge_name, limit in boundaries:
        if not output:
            break
        source = output
        output = []

        def inside(point: np.ndarray) -> bool:
            value = float(point[0]) if edge_name in ("left", "right") else float(point[1])
            if edge_name in ("left", "bottom"):
                return value >= limit
            return value <= limit

        def intersection(
            start: np.ndarray, end: np.ndarray
        ) -> np.ndarray:
            sx, sy = float(start[0]), float(start[1])
            ex, ey = float(end[0]), float(end[1])
            if edge_name in ("left", "right"):
                denominator = ex - sx
                t = 0.0 if denominator == 0.0 else (limit - sx) / denominator
                return np.asarray((limit, sy + t * (ey - sy)), dtype=np.float64)
            denominator = ey - sy
            t = 0.0 if denominator == 0.0 else (limit - sy) / denominator
            return np.asarray((sx + t * (ex - sx), limit), dtype=np.float64)

        previous = source[-1]
        previous_inside = inside(previous)
        for current in source:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside

    cleaned: list[np.ndarray] = []
    for point in output:
        if not cleaned or not np.allclose(cleaned[-1], point, rtol=0.0, atol=1e-12):
            cleaned.append(point)
    if len(cleaned) > 1 and np.allclose(cleaned[0], cleaned[-1], rtol=0.0, atol=1e-12):
        cleaned.pop()
    return np.asarray(cleaned, dtype=np.float64)


def clip_segments_to_rect(segments: np.ndarray, rect: Rect) -> np.ndarray:
    """Liang-Barsky clipping for individual line segments."""

    xmin, ymin, xmax, ymax = rect
    clipped: list[tuple[float, float, float, float]] = []
    for segment in segments:
        x0, y0, x1, y1 = (float(value) for value in segment)
        dx = x1 - x0
        dy = y1 - y0
        t0 = 0.0
        t1 = 1.0
        accepted = True
        for p, q in (
            (-dx, x0 - xmin),
            (dx, xmax - x0),
            (-dy, y0 - ymin),
            (dy, ymax - y0),
        ):
            if p == 0.0:
                if q < 0.0:
                    accepted = False
                    break
                continue
            ratio = q / p
            if p < 0.0:
                if ratio > t1:
                    accepted = False
                    break
                if ratio > t0:
                    t0 = ratio
            else:
                if ratio < t0:
                    accepted = False
                    break
                if ratio < t1:
                    t1 = ratio
        if not accepted or t1 < t0:
            continue
        clipped.append(
            (
                x0 + t0 * dx,
                y0 + t0 * dy,
                x0 + t1 * dx,
                y0 + t1 * dy,
            )
        )
    if not clipped:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def points_inside_even_odd(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test for one ring."""

    if len(points) == 0 or len(vertices) < 3:
        return np.zeros(len(points), dtype=bool)
    px = points[:, 0]
    py = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)

    for index in range(len(vertices)):
        ax, ay = vertices[index]
        bx, by = vertices[(index + 1) % len(vertices)]
        crosses = (ay > py) != (by > py)
        if by != ay:
            with np.errstate(divide="ignore", invalid="ignore"):
                x_intersection = ax + (py - ay) * (bx - ax) / (by - ay)
            inside ^= crosses & (px < x_intersection)

    # Points exactly on an edge count as covered; this avoids platform-dependent
    # ray-casting decisions for aligned sample points.
    eps = 1e-10
    for index in range(len(vertices)):
        ax, ay = vertices[index]
        bx, by = vertices[(index + 1) % len(vertices)]
        dx = bx - ax
        dy = by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            continue
        t = ((px - ax) * dx + (py - ay) * dy) / length_squared
        t = np.clip(t, 0.0, 1.0)
        nearest_x = ax + t * dx
        nearest_y = ay + t * dy
        distance_squared = (px - nearest_x) ** 2 + (py - nearest_y) ** 2
        inside |= distance_squared <= eps * eps
    return inside


def distances_to_segment(
    points: np.ndarray, segment: np.ndarray
) -> np.ndarray:
    ax, ay, bx, by = (float(value) for value in segment)
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    px = points[:, 0]
    py = points[:, 1]
    if length_squared == 0.0:
        return np.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = np.clip(t, 0.0, 1.0)
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return np.hypot(px - nearest_x, py - nearest_y)


def merge_close_vertices(
    points: np.ndarray,
    *,
    closed: bool,
    distance: float,
    minimum_vertices: int = 3,
) -> np.ndarray:
    if len(points) < 2 or distance <= 0.0:
        return points.copy()
    kept = [points[0]]
    for point in points[1:]:
        delta = point - kept[-1]
        if math.hypot(float(delta[0]), float(delta[1])) > distance:
            kept.append(point)
    if closed and len(kept) > 1 and math.hypot(
        float(kept[-1][0] - kept[0][0]), float(kept[-1][1] - kept[0][1])
    ) <= distance:
        kept.pop()
    result = np.asarray(kept, dtype=np.float64)
    return points.copy() if len(result) < minimum_vertices else result


def remove_collinear_vertices(
    points: np.ndarray, *, closed: bool, tolerance: float
) -> np.ndarray:
    if len(points) <= 3 or tolerance < 0.0:
        return points.copy()
    result = points.copy()
    changed = True
    while changed:
        changed = False
        count = len(result)
        if not closed and count <= 2:
            break
        keep = np.ones(count, dtype=bool)
        for index in range(count):
            if not closed and (index == 0 or index == count - 1):
                continue
            prev_index = (index - 1) % count
            next_index = (index + 1) % count
            a = result[prev_index]
            p = result[index]
            b = result[next_index]
            chord = b - a
            chord_length = math.hypot(float(chord[0]), float(chord[1]))
            if chord_length == 0.0:
                keep[index] = False
                changed = True
                continue
            cross = chord[0] * (p[1] - a[1]) - chord[1] * (p[0] - a[0])
            perpendicular = abs(float(cross)) / chord_length
            dot = (p[0] - a[0]) * (b[0] - p[0]) + (p[1] - a[1]) * (b[1] - p[1])
            if perpendicular <= tolerance and dot >= 0.0:
                keep[index] = False
                changed = True
        minimum = 3 if closed else 2
        if np.count_nonzero(keep) < minimum:
            break
        result = result[keep]
    return result


def _rdp_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a = points[start]
        b = points[end]
        chord = b - a
        length = math.hypot(float(chord[0]), float(chord[1]))
        best_index = -1
        best_distance = -1.0
        for index in range(start + 1, end):
            p = points[index]
            if length == 0.0:
                distance = math.hypot(float(p[0] - a[0]), float(p[1] - a[1]))
            else:
                cross = chord[0] * (p[1] - a[1]) - chord[1] * (p[0] - a[0])
                distance = abs(float(cross)) / length
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_distance > tolerance:
            keep[best_index] = True
            stack.append((start, best_index))
            stack.append((best_index, end))
    return points[keep]


def simplify_for_low(
    points: np.ndarray,
    *,
    kind: str,
    closed: bool,
    merge_distance: float,
    tolerance: float,
) -> np.ndarray:
    """Merge adjacent low-LOD vertices and simplify polylines with RDP."""

    merged = merge_close_vertices(
        points,
        closed=kind == "polygon" or closed,
        distance=merge_distance,
        minimum_vertices=3 if kind == "polygon" else 2,
    )
    if kind == "polygon":
        return remove_collinear_vertices(
            merged, closed=True, tolerance=max(tolerance, 0.0)
        )

    if closed and len(merged) >= 3:
        ring = np.vstack((merged, merged[0]))
        simplified = _rdp_open(ring, max(tolerance, 0.0))
        if len(simplified) > 1 and np.allclose(
            simplified[-1], simplified[0], rtol=0.0, atol=1e-12
        ):
            simplified = simplified[:-1]
        if len(simplified) < 2:
            return merged
        return simplified

    simplified = _rdp_open(merged, max(tolerance, 0.0))
    return simplified if len(simplified) >= 2 else merged
