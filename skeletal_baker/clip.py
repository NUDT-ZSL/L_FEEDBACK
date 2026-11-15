"""Orthographic frustum clipping in clip (NDC) space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrthoFrustum:
    """Fixed orthographic projection parameters."""

    center_x: float = 0.0
    center_y: float = 0.0
    half_width: float = 1.2
    half_height: float = 1.2
    near: float = -2.0
    far: float = 2.0

    def to_ndc(self, positions: np.ndarray) -> np.ndarray:
        """World/view space -> normalized device coordinates in [-1, 1]."""
        out = np.empty_like(positions)
        out[:, 0] = (positions[:, 0] - self.center_x) / self.half_width
        out[:, 1] = (positions[:, 1] - self.center_y) / self.half_height
        out[:, 2] = (positions[:, 2] - self.near) / (self.far - self.near) \
            * 2.0 - 1.0
        return out


@dataclass
class ClipStats:
    triangles_in: int = 0
    triangles_passed: int = 0   # fully inside, unchanged
    triangles_clipped: int = 0  # partially clipped
    triangles_culled: int = 0   # fully outside
    triangles_out: int = 0      # emitted after re-triangulation

    def as_dict(self) -> dict:
        return {
            "triangles_in": self.triangles_in,
            "triangles_passed": self.triangles_passed,
            "triangles_clipped": self.triangles_clipped,
            "triangles_culled": self.triangles_culled,
            "triangles_out": self.triangles_out,
        }


# Six clip planes as (axis, bound, keep_greater).
_PLANES = [
    (0, -1.0, True), (0, 1.0, False),
    (1, -1.0, True), (1, 1.0, False),
    (2, -1.0, True), (2, 1.0, False),
]


def _signed_distance(coord: float, bound: float, keep_greater: bool) -> float:
    return coord - bound if keep_greater else bound - coord


def _clip_polygon_against_plane(
    poly: list[tuple[np.ndarray, np.ndarray]],
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sutherland-Hodgman half-space clip of one (position, normal) polygon."""
    if not poly:
        return []
    out: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(poly)
    for i in range(n):
        cur_p, cur_n = poly[i]
        prev_p, prev_n = poly[i - 1]
        cur_d = _signed_distance(cur_p[axis], bound, keep_greater)
        prev_d = _signed_distance(prev_p[axis], bound, keep_greater)
        cur_in = cur_d >= 0.0
        prev_in = prev_d >= 0.0
        if cur_in != prev_in:
            denom = prev_d - cur_d
            t = prev_d / denom if denom != 0.0 else 0.0
            hit_p = prev_p + t * (cur_p - prev_p)
            hit_n = prev_n + t * (cur_n - prev_n)
            out.append((hit_p, hit_n))
        if cur_in:
            out.append((cur_p, cur_n))
    return out


def clip_frame(
    positions_ndc: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ClipStats]:
    """Clip all triangles of a frame against the ortho NDC frustum.

    Returns (positions, normals, face_indices) where the emitted triangle
    soup is a fan re-triangulation of each clipped polygon, plus per-frame
    clipping statistics.
    """
    stats = ClipStats(triangles_in=len(triangles))
    out_pos: list[np.ndarray] = []
    out_nrm: list[np.ndarray] = []
    out_face: list[int] = []

    for tri_idx, tri in enumerate(triangles):
        poly = [(positions_ndc[v].copy(), normals[v].copy()) for v in tri]
        for axis, bound, keep_greater in _PLANES:
            poly = _clip_polygon_against_plane(poly, axis, bound, keep_greater)
            if not poly:
                break
        if not poly:
            stats.triangles_culled += 1
            continue
        if len(poly) == 3:
            stats.triangles_passed += 1
        else:
            stats.triangles_clipped += 1
        # Fan re-triangulation of the convex clipped polygon.
        for i in range(1, len(poly) - 1):
            for p, n in (poly[0], poly[i], poly[i + 1]):
                out_pos.append(p)
                out_nrm.append(n)
                out_face.append(tri_idx)
            stats.triangles_out += 1

    if out_pos:
        positions = np.asarray(out_pos, dtype=np.float64)
        norms = np.asarray(out_nrm, dtype=np.float64)
        lengths = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(lengths < 1e-12, 1.0, lengths)
        face_idx = np.asarray(out_face, dtype=np.int32)
    else:
        positions = np.zeros((0, 3), dtype=np.float64)
        norms = np.zeros((0, 3), dtype=np.float64)
        face_idx = np.zeros((0,), dtype=np.int32)
    return positions, norms, face_idx, stats
