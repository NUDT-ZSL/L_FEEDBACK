"""4x4 matrix math (float64) used for every geometric transform."""

from __future__ import annotations

import math

import numpy as np


def normalize(v, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("cannot normalize near-zero vector")
    return v / n


def identity() -> np.ndarray:
    return np.identity(4, dtype=np.float64)


def translation(offset) -> np.ndarray:
    m = identity()
    m[:3, 3] = np.asarray(offset, dtype=np.float64)
    return m


def uniform_scale(s: float) -> np.ndarray:
    m = identity()
    m[0, 0] = m[1, 1] = m[2, 2] = float(s)
    return m


def rotation_y(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def look_at(eye, target, up) -> np.ndarray:
    """Right-handed view matrix; camera looks down -Z, +Y is up."""
    eye = np.asarray(eye, dtype=np.float64)
    f = normalize(np.asarray(target, dtype=np.float64) - eye)
    s = normalize(np.cross(f, np.asarray(up, dtype=np.float64)))
    u = np.cross(s, f)
    m = identity()
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(s @ eye)
    m[1, 3] = -float(u @ eye)
    m[2, 3] = float(f @ eye)
    return m


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Perspective projection, NDC z in [0, 1]."""
    f = 1.0 / math.tan(fovy_rad * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = near * far / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(l: float, r: float, b: float, t: float, n: float, f: float) -> np.ndarray:
    """Orthographic projection, NDC z in [0, 1]."""
    m = identity()
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = 1.0 / (n - f)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = n / (n - f)
    return m


def apply_mat4(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 4) homogeneous transform."""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((*pts.shape[:-1], 1), dtype=np.float64)
    return np.concatenate([pts, ones], axis=-1) @ m.T


def perspective_frustum_planes(fovy_rad: float, aspect: float, near: float, far: float):
    """Camera-space clip planes as (a, b, c, d) with inside = ax+by+cz+d >= 0."""
    ty = math.tan(fovy_rad * 0.5)
    tx = ty * aspect
    return [
        (0.0, 0.0, -1.0, -near),   # near:  -z >= near
        (0.0, 0.0, 1.0, far),      # far:   -z <= far
        (1.0, 0.0, -tx, 0.0),      # left:   x >= z*tx
        (-1.0, 0.0, -tx, 0.0),     # right:  x <= -z*tx
        (0.0, 1.0, -ty, 0.0),      # bottom: y >= z*ty
        (0.0, -1.0, -ty, 0.0),     # top:    y <= -z*ty
    ]


def ortho_frustum_planes(l: float, r: float, b: float, t: float, n: float, f: float):
    return [
        (0.0, 0.0, -1.0, -n),
        (0.0, 0.0, 1.0, f),
        (1.0, 0.0, 0.0, -l),
        (-1.0, 0.0, 0.0, r),
        (0.0, 1.0, 0.0, -b),
        (0.0, -1.0, 0.0, t),
    ]
