from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

Mat4 = np.ndarray
Vec3 = np.ndarray


def identity() -> Mat4:
    return np.eye(4, dtype=np.float64)


def translation(x: float, y: float, z: float) -> Mat4:
    m = identity()
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scaling(x: float, y: float, z: float) -> Mat4:
    return np.diag([x, y, z, 1.0]).astype(np.float64)


def rotation_x(degrees: float) -> Mat4:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def rotation_y(degrees: float) -> Mat4:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def rotation_z(degrees: float) -> Mat4:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def compose(*matrices: Mat4) -> Mat4:
    result = identity()
    for matrix in matrices:
        result = result @ matrix
    return result


def from_column_major(values: Iterable[float]) -> Mat4:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size != 16:
        raise ValueError(f"matrix needs 16 values, got {data.size}")
    return data.reshape((4, 4), order="F").copy()


def to_column_major(matrix: Mat4) -> list[float]:
    return [float(v) for v in np.asarray(matrix, dtype=np.float64).reshape(16, order="F")]


def transform_point(matrix: Mat4, point: np.ndarray) -> Vec3:
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    result = matrix @ p
    if abs(result[3]) < 1e-12:
        return result[:3]
    return result[:3] / result[3]


def transform_direction(matrix: Mat4, direction: np.ndarray) -> Vec3:
    return (matrix @ np.array([*direction[:3], 0.0], dtype=np.float64))[:3]


def normal_matrix(matrix: Mat4) -> np.ndarray:
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    return np.linalg.inv(linear).T


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> Vec3:
    v = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        if fallback is None:
            raise ValueError("cannot normalize a zero vector")
        return np.asarray(fallback, dtype=np.float64)
    return v / length


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Mat4:
    eye = np.asarray(eye, dtype=np.float64)
    forward = normalize(np.asarray(target, dtype=np.float64) - eye)
    right = normalize(np.cross(forward, np.asarray(up, dtype=np.float64)))
    true_up = np.cross(right, forward)
    m = identity()
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -float(np.dot(right, eye))
    m[1, 3] = -float(np.dot(true_up, eye))
    m[2, 3] = float(np.dot(forward, eye))
    return m


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> Mat4:
    if aspect <= 0 or near <= 0 or far <= near:
        raise ValueError("invalid perspective parameters")
    f = 1.0 / math.tan(math.radians(fov_y_degrees) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> Mat4:
    if abs(right - left) < 1e-12 or abs(top - bottom) < 1e-12 or abs(far - near) < 1e-12:
        raise ValueError("invalid orthographic parameters")
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def gl_matrix(matrix: Mat4) -> np.ndarray:
    """Return column-major float32 data suitable for glUniformMatrix*fv."""
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32).T).reshape(-1)


def gl_normal_matrix(matrix: Mat4) -> np.ndarray:
    return np.ascontiguousarray(normal_matrix(matrix).astype(np.float32).T).reshape(-1)


def cube_face_view_matrices(position: np.ndarray) -> list[Mat4]:
    """Six standard OpenGL cube-map face views, in +X,-X,+Y,-Y,+Z,-Z order."""
    p = np.asarray(position, dtype=np.float64)
    targets = [
        p + np.array([1.0, 0.0, 0.0]),
        p + np.array([-1.0, 0.0, 0.0]),
        p + np.array([0.0, 1.0, 0.0]),
        p + np.array([0.0, -1.0, 0.0]),
        p + np.array([0.0, 0.0, 1.0]),
        p + np.array([0.0, 0.0, -1.0]),
    ]
    ups = [
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    return [look_at(p, target, up) for target, up in zip(targets, ups, strict=True)]


def world_bounds(nodes: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    for transform, positions in nodes:
        homogeneous = np.concatenate(
            [np.asarray(positions, dtype=np.float64), np.ones((len(positions), 1))], axis=1
        )
        points.append((homogeneous @ np.asarray(transform).T)[:, :3])
    if not points:
        raise ValueError("cannot compute bounds without geometry")
    all_points = np.concatenate(points, axis=0)
    return all_points.min(axis=0), all_points.max(axis=0)

