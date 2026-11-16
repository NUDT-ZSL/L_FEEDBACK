from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _bernstein(t: np.ndarray):
    one = 1.0 - t
    basis = np.stack(
        (one * one * one, 3.0 * t * one * one, 3.0 * t * t * one, t * t * t),
        axis=-1,
    )
    deriv = np.stack(
        (-3.0 * one * one, 3.0 * one * one - 6.0 * t * one, 6.0 * t * one - 3.0 * t * t, 3.0 * t * t),
        axis=-1,
    )
    return basis, deriv


def _control_grid() -> np.ndarray:
    xs = (-1.72, -0.58, 0.58, 1.72)
    zs = (-1.08, -0.36, 0.36, 1.08)
    grid = np.zeros((4, 4, 3), dtype=np.float64)
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            xn = x / 1.72
            zn = z / 1.08
            crown = 0.39 * (1.0 - xn * xn) * (1.0 - 0.25 * zn * zn)
            spine = 0.085 * np.exp(-(x / 0.43) ** 2) * (1.0 - 0.18 * zn * zn)
            nose_lift = 0.018 * np.exp(-((x / 0.85) ** 2)) * max(0.0, zn)
            grid[i, j] = (x, crown + spine + nose_lift, z)
    return grid


_CONTROL = _control_grid()


def eval_hood(u: np.ndarray, v: np.ndarray):
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    bu, dbu = _bernstein(uu)
    bv, dbv = _bernstein(vv)

    point = np.einsum("...i,...j,ijk->...k", bu, bv, _CONTROL)
    du = np.einsum("...i,...j,ijk->...k", dbu, bv, _CONTROL)
    dv = np.einsum("...i,...j,ijk->...k", bu, dbv, _CONTROL)
    normal = np.cross(dv, du)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    tangent = du
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
    return point, normal, tangent


@dataclass(frozen=True)
class HoodMesh:
    vertices: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    subdivisions_u: int
    subdivisions_v: int


def hood_mesh(subdivisions_u: int = 128, subdivisions_v: int = 128) -> HoodMesh:
    su = max(96, int(subdivisions_u))
    sv = max(96, int(subdivisions_v))
    u = np.linspace(0.0, 1.0, su + 1, dtype=np.float64)
    v = np.linspace(0.0, 1.0, sv + 1, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points, normals, tangents = eval_hood(uu, vv)

    vertices = np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float64)
    normal_array = np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float64)
    tangent_array = np.ascontiguousarray(tangents.reshape(-1, 3), dtype=np.float64)
    uvs = np.ascontiguousarray(
        np.stack((uu.reshape(-1), vv.reshape(-1)), axis=-1), dtype=np.float64
    )

    ids = np.arange((su + 1) * (sv + 1), dtype=np.int32).reshape(su + 1, sv + 1)
    a = ids[:-1, :-1].reshape(-1)
    b = ids[1:, :-1].reshape(-1)
    c = ids[1:, 1:].reshape(-1)
    d = ids[:-1, 1:].reshape(-1)
    triangles = np.concatenate(
        (
            np.stack((a, b, c), axis=-1),
            np.stack((a, c, d), axis=-1),
        ),
        axis=0,
    ).astype(np.int32)
    return HoodMesh(
        vertices=vertices,
        normals=normal_array,
        tangents=tangent_array,
        uvs=uvs,
        triangles=triangles,
        subdivisions_u=su,
        subdivisions_v=sv,
    )

