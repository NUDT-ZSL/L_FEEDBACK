"""Float64 camera math: view/projection matrices and six frustum planes.

Matrices use the standard OpenGL column-vector storage (``p_clip = M @ p``).
Frustum planes are extracted with the Gribb-Hartmann column-combination rule
and normalised so the AABB test below is scale independent.
"""

from __future__ import annotations

import math

import numpy as np

from .config import CameraConfig


def _normalize(vector):
    array = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(array))
    return array / length, length


def look_at(camera: CameraConfig) -> np.ndarray:
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.look_at, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward, _ = _normalize(target - eye)
    right, _ = _normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(right @ eye)
    matrix[1, 3] = -float(true_up @ eye)
    matrix[2, 3] = float(forward @ eye)
    return matrix


def perspective(camera: CameraConfig) -> np.ndarray:
    fov = math.radians(camera.fov_y_deg)
    focal = 1.0 / math.tan(fov * 0.5)
    near, far = camera.near, camera.far
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[3, 2] = -1.0
    matrix[2, 3] = 2.0 * far * near / (near - far)
    return matrix


class Frustum:
    """Six outward-facing planes ``(a, b, c, d)`` with ``a*x+b*y+c*z+d=0``."""

    def __init__(self, camera: CameraConfig):
        view_projection = perspective(camera) @ look_at(camera)
        rows = [view_projection[:, i].copy() for i in range(4)]
        combinations = [
            rows[3] + rows[0],  # left
            rows[3] - rows[0],  # right
            rows[3] + rows[1],  # bottom
            rows[3] - rows[1],  # top
            rows[3] + rows[2],  # near
            rows[3] - rows[2],  # far
        ]
        self.planes = np.zeros((6, 4), dtype=np.float64)
        for index, plane in enumerate(combinations):
            normal_length = math.sqrt(
                plane[0] * plane[0]
                + plane[1] * plane[1]
                + plane[2] * plane[2]
            )
            self.planes[index, :] = plane / normal_length

    def intersects_aabb(self, aabb) -> bool:
        """Conservative plane-vs-AABB test.

        For every plane, the positive p-vertex maximises the signed distance.
        If even one plane leaves the entire AABB strictly outside, the box is
        rejected; everything on or across a plane is kept (no false negatives).
        """
        x0, y0, z0, x1, y1, z1 = aabb
        for plane in self.planes:
            a, b, c, d = plane
            px = x1 if a >= 0.0 else x0
            py = y1 if b >= 0.0 else y0
            pz = z1 if c >= 0.0 else z0
            if a * px + b * py + c * pz + d < 0.0:
                return False
        return True
