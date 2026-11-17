from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import Camera


def normalize(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length <= 1e-15:
        if fallback is None:
            return np.zeros_like(value)
        return np.asarray(fallback, dtype=np.float64).copy()
    return value / length


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    forward = normalize(np.asarray(target, dtype=np.float64) - eye)
    side = np.cross(forward, normalize(up))
    if float(np.linalg.norm(side)) <= 1e-12:
        raise ValueError("degenerate look-at basis")
    side = normalize(side)
    true_up = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = side
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(side, eye))
    matrix[1, 3] = -float(np.dot(true_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 2.0 / (right - left)
    matrix[1, 1] = 2.0 / (top - bottom)
    matrix[2, 2] = -2.0 / (far - near)
    matrix[0, 3] = -(right + left) / (right - left)
    matrix[1, 3] = -(top + bottom) / (top - bottom)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix


def camera_matrices(camera: Camera, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = camera.normalized(width, height)
    view = look_at(np.asarray(normalized.eye), np.asarray(normalized.target), np.asarray(normalized.up))
    projection = perspective(normalized.fov_y_deg, width / height, normalized.near, normalized.far)
    return view, projection, projection @ view


def camera_basis(camera: Camera, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = camera.normalized(width, height)
    eye = np.asarray(camera.eye, dtype=np.float64)
    forward = normalize(np.asarray(camera.target) - eye)
    right = normalize(np.cross(forward, np.asarray(camera.up, dtype=np.float64)))
    up = normalize(np.cross(right, forward))
    return eye, forward, right, up, np.asarray(camera.up, dtype=np.float64)


_CLIP_PLANES = (
    np.asarray((1.0, 0.0, 0.0, 1.0)),
    np.asarray((-1.0, 0.0, 0.0, 1.0)),
    np.asarray((0.0, 1.0, 0.0, 1.0)),
    np.asarray((0.0, -1.0, 0.0, 1.0)),
    np.asarray((0.0, 0.0, 1.0, 1.0)),
    np.asarray((0.0, 0.0, -1.0, 1.0)),
)


def clip_homogeneous(points: np.ndarray, attributes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Clip convex polygon vertices in homogeneous clip space.

    Points have shape (n, 4). Attributes, when supplied, are interpolated
    linearly in homogeneous space and have shape (n, k).
    """
    polygon = np.asarray(points, dtype=np.float64)
    attrs = None if attributes is None else np.asarray(attributes, dtype=np.float64)
    if polygon.shape[0] == 0:
        return polygon.reshape(0, 4), None if attrs is None else attrs.reshape(0, -1)
    original_attrs = attrs
    if attrs is not None:
        attrs = attrs.copy()
    for plane in _CLIP_PLANES:
        if polygon.shape[0] == 0:
            break
        distances = polygon @ plane
        output: list[np.ndarray] = []
        output_attrs: list[np.ndarray] = []
        count = polygon.shape[0]
        for i in range(count):
            j = (i + 1) % count
            da = float(distances[i])
            db = float(distances[j])
            inside_a = da >= 0.0
            inside_b = db >= 0.0
            if inside_a:
                output.append(polygon[i])
                if attrs is not None:
                    output_attrs.append(attrs[i])
            if inside_a != inside_b:
                denominator = da - db
                if abs(denominator) <= 1e-15:
                    continue
                t = da / denominator
                t = min(max(t, 0.0), 1.0)
                point = polygon[i] + t * (polygon[j] - polygon[i])
                output.append(point)
                if attrs is not None:
                    output_attrs.append(attrs[i] + t * (attrs[j] - attrs[i]))
        if not output:
            polygon = np.zeros((0, 4), dtype=np.float64)
            attrs = None if original_attrs is None else np.zeros((0, original_attrs.shape[1]), dtype=np.float64)
        else:
            polygon = np.asarray(output, dtype=np.float64)
            if attrs is not None:
                attrs = np.asarray(output_attrs, dtype=np.float64)
    return polygon, attrs


def clip_to_frustum(points: np.ndarray, attributes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    return clip_homogeneous(points, attributes)


def clip_line_segment_near(a: np.ndarray, b: np.ndarray, near_epsilon: float = 1e-7) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a homogeneous line segment to w > near_epsilon."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    wa = float(a[3])
    wb = float(b[3])
    inside_a = wa > near_epsilon
    inside_b = wb > near_epsilon
    if not inside_a and not inside_b:
        return None
    if inside_a and inside_b:
        return a, b
    denominator = wa - wb
    if abs(denominator) <= 1e-15:
        return None
    t = (wa - near_epsilon) / denominator
    t = min(max(t, 0.0), 1.0)
    crossing = a + t * (b - a)
    return (a, crossing) if inside_a else (crossing, b)


def project_homogeneous(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    w = points[:, 3]
    if np.any(w <= 1e-12) or not np.all(np.isfinite(points)):
        raise ValueError("projection reached non-positive w")
    ndc = points[:, :3] / w[:, None]
    screen = np.empty((points.shape[0], 3), dtype=np.float64)
    screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
    screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * height
    screen[:, 2] = ndc[:, 2]
    return screen, w


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def triangle_screen_area(triangle: np.ndarray) -> np.ndarray:
    a = triangle[:, 0, :2]
    b = triangle[:, 1, :2]
    c = triangle[:, 2, :2]
    return cross2(b - a, c - a)


def frustum_corners(camera: Camera, width: int, height: int, near: float, far: float) -> np.ndarray:
    eye, forward, right, up, _ = camera_basis(camera, width, height)
    tan_half = math.tan(math.radians(camera.fov_y_deg) * 0.5)
    aspect = width / height
    corners = []
    for depth in (near, far):
        center = eye + forward * depth
        half_h = tan_half * depth
        half_w = half_h * aspect
        for sy, sx in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)):
            corners.append(center + right * (sx * half_w) + up * (sy * half_h))
    return np.asarray(corners, dtype=np.float64)


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1)


def select_up(direction: np.ndarray) -> np.ndarray:
    direction = normalize(direction)
    if abs(float(direction[1])) < 0.92:
        return np.asarray((0.0, 1.0, 0.0))
    return np.asarray((0.0, 0.0, 1.0))


def finite_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def iter_unique_rows(rows: Iterable[np.ndarray]) -> list[np.ndarray]:
    seen: set[bytes] = set()
    result: list[np.ndarray] = []
    for row in rows:
        key = np.asarray(row, dtype=np.float64).tobytes()
        if key not in seen:
            seen.add(key)
            result.append(np.asarray(row, dtype=np.float64))
    return result
