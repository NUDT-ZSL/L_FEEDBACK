"""Small, deterministic vector and camera helpers.

The renderer keeps all geometry in world space. Camera-space conversion and
projection are separate so the same rasterizer can be used by the primary
camera and by perspective/orthographic shadow cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-12


def normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / np.maximum(length, EPS)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def look_at_basis(
    eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(target - eye)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    if abs(float(np.dot(forward, normalize(up_hint)))) > 0.999999:
        raise ValueError("camera up vector is parallel to its view direction")
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    return right, up, forward


def basis_from_forward(
    forward: np.ndarray, up_hint: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(forward)
    up_hint = normalize(up_hint)
    if abs(float(np.dot(forward, up_hint))) > 0.999999:
        raise ValueError("view up hint is parallel to view direction")
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    return right, up, forward


@dataclass
class PerspectiveCamera:
    eye: np.ndarray
    target: np.ndarray
    up_hint: np.ndarray
    fov_y_radians: float
    near: float
    far: float
    width: int
    height: int

    def __post_init__(self) -> None:
        self.right, self.up, self.forward = look_at_basis(
            self.eye, self.target, self.up_hint
        )
        self.aspect = float(self.width) / float(self.height)
        self.focal = 1.0 / np.tan(self.fov_y_radians * 0.5)

    def to_view(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        delta = points - self.eye
        return np.stack(
            (
                delta @ self.right,
                delta @ self.up,
                -(delta @ self.forward),
            ),
            axis=-1,
        )

    def project(self, view_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return screen x/y and reciprocal homogeneous depth."""
        view_points = np.asarray(view_points, dtype=np.float64)
        depth = np.maximum(-view_points[:, 2], EPS)
        inv_depth = 1.0 / depth
        ndc_x = self.focal / self.aspect * view_points[:, 0] * inv_depth
        ndc_y = self.focal * view_points[:, 1] * inv_depth
        screen_x = (ndc_x * 0.5 + 0.5) * self.width
        screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height
        return screen_x, screen_y, inv_depth

    def depth_metric(self, world_points: np.ndarray) -> np.ndarray:
        return (np.asarray(world_points) - self.eye) @ self.forward

    def mesh_in_frustum(self, center: np.ndarray, radius: float) -> bool:
        view = self.to_view(np.asarray(center, dtype=np.float64))
        tan_y = np.tan(self.fov_y_radians * 0.5)
        tan_x = tan_y * self.aspect
        side_x_norm = np.sqrt(1.0 + tan_x * tan_x)
        side_y_norm = np.sqrt(1.0 + tan_y * tan_y)
        planes = (
            (np.asarray([0.0, 0.0, -1.0]), -self.near),
            (np.asarray([0.0, 0.0, 1.0]), self.far),
            (
                np.asarray([1.0, 0.0, -tan_x]) / side_x_norm,
                0.0,
            ),
            (
                np.asarray([-1.0, 0.0, -tan_x]) / side_x_norm,
                0.0,
            ),
            (
                np.asarray([0.0, 1.0, -tan_y]) / side_y_norm,
                0.0,
            ),
            (
                np.asarray([0.0, -1.0, -tan_y]) / side_y_norm,
                0.0,
            ),
        )
        for normal, offset in planes:
            if float(np.dot(normal, view) + offset) < -radius:
                return False
        return True


@dataclass
class OrthographicCamera:
    eye: np.ndarray
    forward: np.ndarray
    up_hint: np.ndarray
    half_extent: float
    near: float
    far: float
    width: int
    height: int

    def __post_init__(self) -> None:
        self.right, self.up, self.forward = basis_from_forward(
            self.forward, self.up_hint
        )

    def to_view(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        delta = points - self.eye
        return np.stack(
            (
                delta @ self.right,
                delta @ self.up,
                -(delta @ self.forward),
            ),
            axis=-1,
        )

    def project(self, view_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        view_points = np.asarray(view_points, dtype=np.float64)
        ndc_x = view_points[:, 0] / self.half_extent
        ndc_y = view_points[:, 1] / self.half_extent
        screen_x = (ndc_x * 0.5 + 0.5) * self.width
        screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height
        inv_depth = np.ones(view_points.shape[0], dtype=np.float64)
        return screen_x, screen_y, inv_depth

    def depth_metric(self, world_points: np.ndarray) -> np.ndarray:
        return (np.asarray(world_points) - self.eye) @ self.forward


def clip_polygon_near(
    view_points: np.ndarray,
    world_points: np.ndarray,
    normals: np.ndarray,
    near: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip a convex polygon against z_view <= -near."""
    return _clip_polygon_plane(view_points, world_points, normals, near, keep_near=True)


def clip_polygon_far(
    view_points: np.ndarray,
    world_points: np.ndarray,
    normals: np.ndarray,
    far: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip a convex polygon against z_view >= -far."""
    return _clip_polygon_plane(view_points, world_points, normals, far, keep_near=False)


def _clip_polygon_plane(
    view_points: np.ndarray,
    world_points: np.ndarray,
    normals: np.ndarray,
    distance: float,
    *,
    keep_near: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view_points = np.asarray(view_points, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    count = view_points.shape[0]
    if count == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    def inside(point: np.ndarray) -> bool:
        if keep_near:
            return bool(point[2] <= -distance)
        return bool(point[2] >= -distance)

    out_view: list[np.ndarray] = []
    out_world: list[np.ndarray] = []
    out_normals: list[np.ndarray] = []
    for index in range(count):
        current = index
        previous = (index - 1) % count
        current_inside = inside(view_points[current])
        previous_inside = inside(view_points[previous])
        if current_inside != previous_inside:
            a = view_points[previous]
            b = view_points[current]
            denominator = float(b[2] - a[2])
            if abs(denominator) < EPS:
                continue
            if keep_near:
                t = (-distance - float(a[2])) / denominator
            else:
                t = (-distance - float(a[2])) / denominator
            t = min(max(t, 0.0), 1.0)
            out_view.append(a + t * (b - a))
            out_world.append(
                world_points[previous]
                + t * (world_points[current] - world_points[previous])
            )
            out_normals.append(
                normals[previous] + t * (normals[current] - normals[previous])
            )
        if current_inside:
            out_view.append(view_points[current])
            out_world.append(world_points[current])
            out_normals.append(normals[current])

    if not out_view:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(out_view, dtype=np.float64),
        np.asarray(out_world, dtype=np.float64),
        np.asarray(out_normals, dtype=np.float64),
    )
