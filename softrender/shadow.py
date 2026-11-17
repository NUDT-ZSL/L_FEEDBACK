"""Shadow maps: 2 directional maps + 1 point-light cube map, 4x4 PCF."""

from __future__ import annotations

import numpy as np

from . import math3d

# Cube face layout: (major axis, right, down). Rendering uses a camera
# looking along ``major`` with up = -down, so render and sample agree.
CUBE_FACES = []
for _major, _up_hint in (
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
):
    _m = np.asarray(_major)
    _u = np.asarray(_up_hint)
    _right = math3d.normalize(np.cross(_m, _u))
    _down = -np.cross(_right, _m)
    CUBE_FACES.append((_m, _right, _down))

_FACE_MAJOR = np.array([f[0] for f in CUBE_FACES])
_FACE_RIGHT = np.array([f[1] for f in CUBE_FACES])
_FACE_DOWN = np.array([f[2] for f in CUBE_FACES])

# 4x4 PCF kernel offsets (texel units).
_PCF_OFFSETS = np.array(
    [(i, j) for i in (-1.5, -0.5, 0.5, 1.5) for j in (-1.5, -0.5, 0.5, 1.5)],
    dtype=np.float64,
)


def _bilinear_compare(depth_map, u, v, z):
    """Bilinear blend of 4 depth comparisons at continuous texel (u, v)."""
    res = depth_map.shape[0]
    x = u - 0.5
    y = v - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x0c = np.clip(x0, 0, res - 1)
    x1c = np.clip(x0 + 1, 0, res - 1)
    y0c = np.clip(y0, 0, res - 1)
    y1c = np.clip(y0 + 1, 0, res - 1)
    c00 = (z <= depth_map[y0c, x0c]).astype(np.float64)
    c01 = (z <= depth_map[y0c, x1c]).astype(np.float64)
    c10 = (z <= depth_map[y1c, x0c]).astype(np.float64)
    c11 = (z <= depth_map[y1c, x1c]).astype(np.float64)
    return (c00 * (1 - fx) + c01 * fx) * (1 - fy) + (c10 * (1 - fx) + c11 * fx) * fy


def _slope_bias(cos_nl, const_bias, slope_bias):
    cos_nl = np.clip(cos_nl, 1e-3, 1.0)
    tan = np.sqrt(1.0 - cos_nl * cos_nl) / cos_nl
    tan = np.minimum(tan, 8.0)
    return const_bias + slope_bias * tan


class DirShadowMap:
    """Orthographic depth map for one directional light."""

    def __init__(self, depth: np.ndarray, rig, light_dir_world: np.ndarray):
        self.depth = depth.astype(np.float32)
        self.res = depth.shape[0]
        self.rig = rig
        self.light_dir = math3d.normalize(light_dir_world)
        self.view_proj = rig.proj @ rig.view
        # World size of one shadow texel (for the normal offset).
        self.texel_world = self._estimate_texel_world()
        self.depth_range = self._estimate_depth_range()

    def _estimate_texel_world(self) -> float:
        # NDC x spans 2 units over `res` texels; world per NDC unit = 1/P[0,0].
        return float(2.0 / (self.res * abs(self.rig.proj[0, 0])))

    def _estimate_depth_range(self) -> float:
        return float(1.0 / abs(self.rig.proj[2, 2]))

    def visibility(self, pos: np.ndarray, nrm: np.ndarray) -> np.ndarray:
        """pos, nrm: (N, 3) world space. Returns (N,) shadow factor in [0, 1]."""
        # Normal offset to fight acne without losing contact shadows.
        p = pos + nrm * (1.5 * self.texel_world)
        clip = math3d.apply_mat4(self.view_proj, p)
        w = clip[:, 3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip[:, :3] / w[:, None]
        u = (ndc[:, 0] * 0.5 + 0.5) * self.res
        v = (0.5 - ndc[:, 1] * 0.5) * self.res
        z = ndc[:, 2]
        cos_nl = np.clip(np.sum(nrm * -self.light_dir, axis=1), 0.0, 1.0)
        bias = _slope_bias(cos_nl, 0.35 * self.texel_world, 1.2 * self.texel_world)
        zb = z - bias / self.depth_range
        vis = np.zeros_like(zb)
        for ox, oy in _PCF_OFFSETS:
            vis += _bilinear_compare(self.depth, u + ox, v + oy, zb)
        vis /= len(_PCF_OFFSETS)
        outside = ((u < -0.5) | (u > self.res + 0.5)
                   | (v < -0.5) | (v > self.res + 0.5) | (z > 1.0) | (z < 0.0))
        return np.where(outside, 1.0, vis)


def _dir_to_face_uv(direction: np.ndarray):
    """direction: (N, 3). Returns (face (N,), s (N,), t (N,)) in [-1, 1]."""
    ad = np.abs(direction)
    axis = np.argmax(ad, axis=1)
    sign = np.take_along_axis(direction, axis[:, None], axis=1)[:, 0] >= 0
    face = axis * 2 + (~sign).astype(np.int64)
    major = _FACE_MAJOR[face]
    right = _FACE_RIGHT[face]
    down = _FACE_DOWN[face]
    denom = np.sum(direction * major, axis=1)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    s = np.sum(direction * right, axis=1) / denom
    t = np.sum(direction * down, axis=1) / denom
    return face, s, t


class CubeShadowMap:
    """Six-face cube shadow map storing radial distance (world units)."""

    def __init__(self, faces: np.ndarray, position: np.ndarray):
        self.faces = faces.astype(np.float32)  # (6, res, res)
        self.res = faces.shape[1]
        self.position = np.asarray(position, dtype=np.float64)

    def _fetch(self, face, i, j):
        """Texel fetch with seamless cross-face addressing."""
        res = self.res
        out = np.empty(face.shape[0], dtype=np.float64)
        inside = (i >= 0) & (i < res) & (j >= 0) & (j < res)
        if inside.any():
            out[inside] = self.faces[face[inside], j[inside], i[inside]]
        if (~inside).any():
            idx = np.nonzero(~inside)[0]
            f = face[idx]
            # Direction of the requested texel center on the original face.
            s = ((i[idx] + 0.5) / res) * 2.0 - 1.0
            t = ((j[idx] + 0.5) / res) * 2.0 - 1.0
            direction = (_FACE_MAJOR[f] + s[:, None] * _FACE_RIGHT[f]
                         + t[:, None] * _FACE_DOWN[f])
            f2, s2, t2 = _dir_to_face_uv(direction)
            i2 = np.clip(np.floor((s2 * 0.5 + 0.5) * res).astype(np.int64), 0, res - 1)
            j2 = np.clip(np.floor((t2 * 0.5 + 0.5) * res).astype(np.int64), 0, res - 1)
            out[idx] = self.faces[f2, j2, i2]
        return out

    def _bilinear_compare(self, direction, z):
        res = self.res
        face, s, t = _dir_to_face_uv(direction)
        x = (s * 0.5 + 0.5) * res - 0.5
        y = (t * 0.5 + 0.5) * res - 0.5
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = x - x0
        fy = y - y0
        c00 = (z <= self._fetch(face, x0, y0)).astype(np.float64)
        c01 = (z <= self._fetch(face, x0 + 1, y0)).astype(np.float64)
        c10 = (z <= self._fetch(face, x0, y0 + 1)).astype(np.float64)
        c11 = (z <= self._fetch(face, x0 + 1, y0 + 1)).astype(np.float64)
        return (c00 * (1 - fx) + c01 * fx) * (1 - fy) + (c10 * (1 - fx) + c11 * fx) * fy

    def visibility(self, pos: np.ndarray, nrm: np.ndarray) -> np.ndarray:
        """pos, nrm: (N, 3) world space. Returns (N,) shadow factor in [0, 1]."""
        dvec = pos - self.position
        dist = np.linalg.norm(dvec, axis=1)
        dist = np.maximum(dist, 1e-6)
        # Normal offset scaled by the world size of a texel at this distance.
        texel_world = 2.0 * dist / self.res
        p = pos + nrm * (1.5 * texel_world)[:, None]
        dvec = p - self.position
        dist = np.linalg.norm(dvec, axis=1)
        dist = np.maximum(dist, 1e-6)
        light_dir = dvec / dist[:, None]
        cos_nl = np.clip(np.sum(nrm * -light_dir, axis=1), 0.0, 1.0)
        bias = _slope_bias(cos_nl, 0.35 * texel_world, 1.2 * texel_world)
        zb = dist - bias
        # Tangent frame around the sample direction for PCF offsets.
        ref = np.where(np.abs(light_dir[:, 1:2]) < 0.9,
                       np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        right = np.cross(light_dir, ref)
        right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        up = np.cross(right, light_dir)
        step = (2.0 * dist / self.res)[:, None]
        vis = np.zeros_like(zb)
        for ox, oy in _PCF_OFFSETS:
            tap = dvec + (right * ox + up * oy) * step
            vis += self._bilinear_compare(tap, zb)
        return vis / len(_PCF_OFFSETS)
