from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray, fallback: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return vector / length


def normalize4(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        return np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64)
    return vector / length


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) * 0.5)
    result = np.zeros((4, 4), dtype=np.float64)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = (2.0 * far * near) / (near - far)
    result[3, 2] = -1.0
    return result


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    forward = normalize(np.asarray(target, dtype=np.float64) - eye)
    right = normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)
    result = np.eye(4, dtype=np.float64)
    result[0, :3] = right
    result[1, :3] = true_up
    result[2, :3] = -forward
    result[0, 3] = -float(np.dot(right, eye))
    result[1, 3] = -float(np.dot(true_up, eye))
    result[2, 3] = float(np.dot(forward, eye))
    return result


def light_view_projections(
    position: np.ndarray,
    light_range: float,
    cube_size: int,
) -> list[np.ndarray]:
    del cube_size
    eye = np.asarray(position, dtype=np.float64)
    near = max(0.025, light_range * 0.002)
    projection = perspective(90.0, 1.0, near, max(near + 0.1, light_range))
    directions = (
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    )
    matrices = []
    for direction, up in directions:
        view = look_at(eye, eye + np.asarray(direction), np.asarray(up))
        matrices.append(projection @ view)
    return matrices


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    clip = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return clip


def sphere_frustum_factor(view_proj: np.ndarray, center: np.ndarray, radius: float) -> float:
    clip = transform_points(view_proj, np.asarray(center, dtype=np.float64)[None, :])[0]
    if clip[3] <= 1.0e-5:
        return 0.15
    ndc = clip[:3] / clip[3]
    margin = max(0.05, radius * 0.18)
    inside = (
        ndc[0] >= -1.0 - margin
        and ndc[0] <= 1.0 + margin
        and ndc[1] >= -1.0 - margin
        and ndc[1] <= 1.0 + margin
        and ndc[2] >= -1.0
        and ndc[2] <= 1.0
    )
    return 1.0 if inside else 0.15


def ray_segment_intersects_triangles(
    origin: np.ndarray,
    target: np.ndarray,
    triangles: np.ndarray,
    *,
    epsilon: float = 0.003,
) -> bool:
    """Return True if the open segment origin->target is occluded by a triangle."""
    if len(triangles) == 0:
        return False
    origin = np.asarray(origin, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    direction = target - origin
    distance = float(np.linalg.norm(direction))
    if distance <= epsilon * 2.0:
        return False
    direction /= distance

    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    h = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1.0e-9
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    s = origin[None, :] - v0
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.dot(q, direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = (
        valid
        & (u >= -1.0e-7)
        & (v >= -1.0e-7)
        & (u + v <= 1.0 + 1.0e-7)
        & (t > epsilon)
        & (t < distance - epsilon)
    )
    return bool(np.any(hit))

