from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .scene import PlaneObject, Scene, SphereObject

RAY_EPSILON = 1e-7


@dataclass(frozen=True)
class GBuffer:
    width: int
    height: int
    world: np.ndarray
    geometric_normal: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    object_id: np.ndarray
    valid: np.ndarray
    edge: np.ndarray


def camera_rays(scene: Scene) -> np.ndarray:
    width = scene.width
    height = scene.height
    aspect = float(width) / float(height)
    tan_half = float(np.tan(np.deg2rad(scene.camera.fov_degrees) * 0.5))
    x = (2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0)
    y = (1.0 - 2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height)
    x_ndc = x[None, :] * aspect * tan_half
    y_ndc = y[:, None] * tan_half
    directions = (
        scene.camera.forward[None, None, :]
        + x_ndc[..., None] * scene.camera.right[None, None, :]
        + y_ndc[..., None] * scene.camera.true_up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.ascontiguousarray(directions, dtype=np.float64)


def _sphere_intersection(
    origin: np.ndarray,
    directions: np.ndarray,
    sphere: SphereObject,
) -> Tuple[np.ndarray, np.ndarray]:
    offset = origin - sphere.center
    b = 2.0 * np.einsum("...k,k->...", directions, offset)
    c = float(np.dot(offset, offset)) - sphere.radius * sphere.radius
    discriminant = b * b - 4.0 * c
    hit = discriminant >= 0.0
    near = np.full(discriminant.shape, np.inf, dtype=np.float64)
    far = np.full(discriminant.shape, np.inf, dtype=np.float64)
    if np.any(hit):
        root = np.sqrt(discriminant[hit])
        near[hit] = (-b[hit] - root) * 0.5
        far[hit] = (-b[hit] + root) * 0.5
    distance = np.where(near > RAY_EPSILON, near, far)
    valid = hit & (distance > RAY_EPSILON)
    return distance, valid


def _plane_intersection(
    origin: np.ndarray,
    directions: np.ndarray,
    plane: PlaneObject,
) -> Tuple[np.ndarray, np.ndarray]:
    denominator = np.einsum("...k,k->...", directions, plane.normal)
    numerator = float(np.dot(plane.point - origin, plane.normal))
    distance = np.full(denominator.shape, np.inf, dtype=np.float64)
    parallel = np.abs(denominator) < 1e-12
    distance[~parallel] = numerator / denominator[~parallel]
    valid = (~parallel) & (distance > RAY_EPSILON)
    safe_distance = np.where(valid, distance, 0.0)
    hit_points = origin[None, None, :] + directions * safe_distance[..., None]
    local_u = np.einsum("...k,k->...", hit_points - plane.point, plane.tangent)
    local_v = np.einsum("...k,k->...", hit_points - plane.point, plane.bitangent)
    valid &= (np.abs(local_u) <= plane.size[0] * 0.5) & (
        np.abs(local_v) <= plane.size[1] * 0.5
    )
    return distance, valid


def _edge_mask(valid: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(valid, dtype=bool)
    if valid.shape[0] > 2 and valid.shape[1] > 2:
        interior[1:-1, 1:-1] = (
            valid[1:-1, 1:-1]
            & valid[:-2, 1:-1]
            & valid[2:, 1:-1]
            & valid[1:-1, :-2]
            & valid[1:-1, 2:]
        )
    return valid & (~interior)


def render_gbuffer(scene: Scene) -> GBuffer:
    width = scene.width
    height = scene.height
    origin = scene.camera.position
    directions = camera_rays(scene)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    object_id = np.full((height, width), -1, dtype=np.int32)

    for index, obj in enumerate(scene.objects):
        if isinstance(obj, SphereObject):
            candidate_depth, candidate_valid = _sphere_intersection(origin, directions, obj)
        else:
            candidate_depth, candidate_valid = _plane_intersection(origin, directions, obj)
        update = candidate_valid & (candidate_depth < depth)
        depth[update] = candidate_depth[update]
        object_id[update] = index

    valid = object_id >= 0
    safe_depth = np.where(valid, depth, 0.0)
    world = origin[None, None, :] + directions * safe_depth[..., None]
    geometric_normal = np.zeros((height, width, 3), dtype=np.float64)
    uv = np.zeros((height, width, 2), dtype=np.float64)

    for index, obj in enumerate(scene.objects):
        selected = object_id == index
        if not np.any(selected):
            continue
        points = world[selected]
        if isinstance(obj, SphereObject):
            local = points - obj.center
            local /= np.linalg.norm(local, axis=-1, keepdims=True)
            geometric_normal[selected] = local
            u = 0.5 + np.arctan2(local[:, 0], local[:, 2]) / (2.0 * np.pi)
            v = np.arccos(np.clip(local[:, 1], -1.0, 1.0)) / np.pi
            uv[selected, 0] = u
            uv[selected, 1] = v
        else:
            geometric_normal[selected] = obj.normal
            local_u = np.einsum("pk,k->p", points - obj.point, obj.tangent)
            local_v = np.einsum("pk,k->p", points - obj.point, obj.bitangent)
            uv[selected, 0] = local_u / obj.size[0] + 0.5
            uv[selected, 1] = local_v / obj.size[1] + 0.5

    edge = _edge_mask(valid)
    return GBuffer(
        width=width,
        height=height,
        world=np.ascontiguousarray(world),
        geometric_normal=np.ascontiguousarray(geometric_normal),
        uv=np.ascontiguousarray(uv),
        depth=np.ascontiguousarray(depth),
        object_id=object_id,
        valid=valid,
        edge=edge,
    )
