"""Perspective camera, frustum model and camera version ids."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .errors import DegenerateFovError

FOV_MIN_DEG = 1.0
FOV_MAX_DEG = 179.0


@dataclass(frozen=True)
class Camera:
    position: Tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    vfov_deg: float
    aspect: float
    near: float = 0.1
    far: float = 500.0

    def __post_init__(self):
        if not (FOV_MIN_DEG <= self.vfov_deg <= FOV_MAX_DEG):
            raise DegenerateFovError(
                "vertical fov out of supported range",
                vfov_deg=self.vfov_deg,
                min_deg=FOV_MIN_DEG,
                max_deg=FOV_MAX_DEG,
            )
        if not (self.aspect > 0):
            raise DegenerateFovError("aspect ratio must be positive", aspect=self.aspect)
        if not (0 < self.near < self.far):
            raise DegenerateFovError(
                "near/far planes invalid", near=self.near, far=self.far
            )

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def look_at(position, target, vfov_deg, aspect, near=0.1, far=500.0) -> "Camera":
        f = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        n = np.linalg.norm(f)
        if n == 0:
            raise DegenerateFovError("look_at target equals camera position")
        f = f / n
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[1]))))
        yaw = math.degrees(math.atan2(-f[0], -f[2]))
        return Camera(
            position=tuple(float(v) for v in position),
            yaw_deg=yaw,
            pitch_deg=pitch,
            vfov_deg=vfov_deg,
            aspect=aspect,
            near=near,
            far=far,
        )

    # -- versioning ------------------------------------------------------------
    @property
    def version(self) -> str:
        payload = {
            "position": [round(v, 9) for v in self.position],
            "yaw_deg": round(self.yaw_deg, 9),
            "pitch_deg": round(self.pitch_deg, 9),
            "vfov_deg": round(self.vfov_deg, 9),
            "aspect": round(self.aspect, 9),
            "near": round(self.near, 9),
            "far": round(self.far, 9),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    # -- basis / transforms ----------------------------------------------------
    def basis(self):
        """Return (forward, right, up) unit vectors in world space."""
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        cp = math.cos(pitch)
        forward = np.array(
            [-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp],
            dtype=np.float64,
        )
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
        up = np.cross(right, forward)
        return forward, right, up

    def to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """World (N,3) -> camera space (N,3), +z forward."""
        forward, right, up = self.basis()
        rel = np.asarray(points_world, dtype=np.float64) - np.asarray(self.position)
        return np.stack(
            [rel @ right, rel @ up, rel @ forward], axis=1
        )

    def project(self, points_cam: np.ndarray, width: int, height: int):
        """Camera-space points -> (sx, sy) pixel coords. z must be > 0."""
        pts = np.asarray(points_cam, dtype=np.float64)
        tan = math.tan(math.radians(self.vfov_deg) * 0.5)
        x_ndc = pts[:, 0] / (pts[:, 2] * tan * self.aspect)
        y_ndc = pts[:, 1] / (pts[:, 2] * tan)
        sx = (x_ndc + 1.0) * 0.5 * width
        sy = (1.0 - y_ndc) * 0.5 * height
        return sx, sy

    # -- visibility --------------------------------------------------------------
    def aabb_visible(self, mins, maxs) -> bool:
        """Conservative AABB-vs-frustum test.

        Returns False only when all 8 corners violate the same frustum
        plane; partially intersecting boxes are always kept.
        """
        mn = np.asarray(mins, dtype=np.float64)
        mx = np.asarray(maxs, dtype=np.float64)
        corners = np.array(
            [
                [x, y, z]
                for x in (mn[0], mx[0])
                for y in (mn[1], mx[1])
                for z in (mn[2], mx[2])
            ]
        )
        cam = self.to_camera(corners)
        tan = math.tan(math.radians(self.vfov_deg) * 0.5)
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        planes = [
            z - self.near,                 # z >= near
            self.far - z,                  # z <= far
            z * tan * self.aspect - x,     # x <=  z*tan*aspect
            z * tan * self.aspect + x,     # x >= -z*tan*aspect
            z * tan - y,                   # y <=  z*tan
            z * tan + y,                   # y >= -z*tan
        ]
        for side in planes:
            if np.all(side < 0.0):
                return False
        return True

    def screen_bbox(self, mins, maxs, width: int, height: int):
        """Conservative screen-space bbox (x0, y0, x1, y1) of an AABB.

        If any corner is at/behind the near plane the projection is
        unbounded; the full image is returned (conservative, never
        under-covers).
        """
        mn = np.asarray(mins, dtype=np.float64)
        mx = np.asarray(maxs, dtype=np.float64)
        corners = np.array(
            [
                [x, y, z]
                for x in (mn[0], mx[0])
                for y in (mn[1], mx[1])
                for z in (mn[2], mx[2])
            ]
        )
        cam = self.to_camera(corners)
        if np.any(cam[:, 2] <= self.near):
            return (0, 0, width, height)
        sx, sy = self.project(cam, width, height)
        x0 = max(0, int(math.floor(sx.min())) - 1)
        y0 = max(0, int(math.floor(sy.min())) - 1)
        x1 = min(width, int(math.ceil(sx.max())) + 1)
        y1 = min(height, int(math.ceil(sy.max())) + 1)
        return (x0, y0, x1, y1)
