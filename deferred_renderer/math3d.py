from __future__ import annotations

import math

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return vector / length


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    forward = normalize(np.asarray(target, dtype=np.float64) - eye)
    side = normalize(np.cross(forward, np.asarray(up, dtype=np.float64)))
    upward = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = side
    matrix[1, :3] = upward
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(side, eye))
    matrix[1, 3] = -float(np.dot(upward, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def perspective(
    fov_y_deg: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    focal = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def translation(offset: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(offset, dtype=np.float64)
    return matrix


def rotation_euler_deg(angles: np.ndarray) -> np.ndarray:
    x, y, z = np.radians(np.asarray(angles, dtype=np.float64))
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array(
        [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def model_matrix(center: np.ndarray, rotation_deg: np.ndarray | None) -> np.ndarray:
    rotation = (
        rotation_euler_deg(rotation_deg)
        if rotation_deg is not None
        else np.eye(4, dtype=np.float64)
    )
    return translation(center) @ rotation


def normal_matrix(model: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(model, dtype=np.float64)[:3, :3]).T


def matrix_bytes(matrix: np.ndarray) -> bytes:
    """Return column-major float32 bytes suitable for OpenGL uniforms."""

    return np.asarray(matrix, dtype=np.float32).T.tobytes(order="C")


def reconstruct_view_rays(
    inv_projection: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Return unit-depth view rays for pixel centers.

    The returned z coordinate is -1. Multiplying x/y/z by a positive linear
    depth yields a view-space point at that depth.
    """

    xs = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0
    gx, gy = np.meshgrid(xs, ys)
    clip = np.stack(
        [gx, gy, -np.ones_like(gx), np.ones_like(gx)],
        axis=-1,
    )
    near_h = clip @ np.asarray(inv_projection, dtype=np.float64).T
    near_view = near_h[..., :3] / near_h[..., 3:4]
    return near_view / (-near_view[..., 2:3])


def view_point_from_depth(
    ndc_x: np.ndarray | float,
    ndc_y: np.ndarray | float,
    depth: np.ndarray | float,
    inv_projection: np.ndarray,
) -> np.ndarray:
    ndc_x = np.asarray(ndc_x, dtype=np.float64)
    ndc_y = np.asarray(ndc_y, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    clip = np.stack([ndc_x, ndc_y, -np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=-1)
    near_h = clip @ np.asarray(inv_projection, dtype=np.float64).T
    ray = near_h[..., :3] / near_h[..., 3:4]
    ray = ray / (-ray[..., 2:3])
    return ray * depth[..., None]


def reconstruct_world(
    depth: np.ndarray,
    inv_projection: np.ndarray,
    inv_view: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    rays = reconstruct_view_rays(inv_projection, width, height)
    view_pos = rays * depth[..., None]
    homogeneous = np.concatenate(
        [view_pos, np.ones((height, width, 1), dtype=np.float64)],
        axis=-1,
    )
    world_h = homogeneous @ np.asarray(inv_view, dtype=np.float64).T
    return world_h[..., :3] / world_h[..., 3:4]


def reconstruction_from_ndc(
    ndc_x: np.ndarray,
    ndc_y: np.ndarray,
    depth: np.ndarray,
    inv_projection: np.ndarray,
    inv_view: np.ndarray,
) -> np.ndarray:
    view_pos = view_point_from_depth(ndc_x, ndc_y, depth, inv_projection)
    homogeneous = np.concatenate(
        [view_pos, np.ones(view_pos.shape[:-1] + (1,), dtype=np.float64)],
        axis=-1,
    )
    world_h = homogeneous @ np.asarray(inv_view, dtype=np.float64).T
    return world_h[..., :3] / world_h[..., 3:4]

