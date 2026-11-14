"""Small, explicit linear algebra helpers used by the baker and viewer."""

from __future__ import annotations

import math

import numpy as np

from .errors import numeric


def vec3(value, label: str = "vector") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise numeric(f"{label} must have 3 components")
    if not np.all(np.isfinite(arr)):
        raise numeric(f"{label} contains a non-finite value")
    return arr


def vec4(value, label: str = "vector") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise numeric(f"{label} must have 4 components")
    if not np.all(np.isfinite(arr)):
        raise numeric(f"{label} contains a non-finite value")
    return arr


def normalize(v: np.ndarray, label: str = "vector") -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-15:
        raise numeric(f"{label} is degenerate")
    return v / n


def safe_normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < eps:
        return None
    return v / n


def mat4(value, label: str = "matrix") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise numeric(f"{label} must be a 4x4 matrix or 16 row-major values")
    if not np.all(np.isfinite(arr)):
        raise numeric(f"{label} contains a non-finite value")
    return arr.copy()


def homogenous_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix.T


def quat_normalize(q: np.ndarray, label: str = "quaternion") -> np.ndarray:
    q = vec4(q, label)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        raise numeric(f"{label} is degenerate")
    return q / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def slerp_short(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(q0)
    b = quat_normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return quat_normalize(a + t * (b - a))
    theta0 = math.acos(dot)
    sin0 = math.sin(theta0)
    w0 = math.sin((1.0 - t) * theta0) / sin0
    w1 = math.sin(t * theta0) / sin0
    return quat_normalize(w0 * a + w1 * b)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = quat_normalize(a)
    bx, by, bz, bw = quat_normalize(b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def compose_trs(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    m = quat_to_matrix(rotation)
    m[:3, :3] = m[:3, :3] @ np.diag(scale)
    m[:3, 3] = translation
    return m


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_vectors(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return vectors @ matrix[:3, :3].T


def orthonormal_tangent(normal: np.ndarray, tangent_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    n = safe_normalize(normal)
    if n is None:
        return None
    t = np.asarray(tangent_hint, dtype=np.float64)
    projected = t - np.dot(t, n) * n
    projected = safe_normalize(projected, 1e-12)
    if projected is None:
        return None
    b = np.cross(n, projected)
    b = safe_normalize(b, 1e-12)
    if b is None:
        return None
    basis = np.stack([n, projected, b], axis=1)
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-10, rtol=0.0):
        return None
    return projected, b


def fibonacci_sphere(count: int = 64) -> np.ndarray:
    if count < 8:
        count = 8
    i = np.arange(count, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * i
    return np.stack([np.cos(theta) * radius, y, np.sin(theta) * radius], axis=1)


def sphere_fallback_tangent(normal: np.ndarray) -> np.ndarray:
    n = safe_normalize(normal)
    if n is None:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    samples = fibonacci_sphere(64)
    best = None
    best_score = -1.0
    for sample in samples:
        projected = sample - np.dot(sample, n) * n
        length = float(np.linalg.norm(projected))
        if length > best_score:
            best_score = length
            best = projected / length
    if best is None or best_score < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, n))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        best = normalize(axis - np.dot(axis, n) * n)
    return best


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

