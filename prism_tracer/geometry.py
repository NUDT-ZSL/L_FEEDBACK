"""Vectorised ray intersections used by camera paths and photons."""

from __future__ import annotations

import numpy as np

from .scene import Scene

KIND_NONE = -1
KIND_TRIANGLE = 0
KIND_WATER = 1
KIND_GROUND = 2
KIND_LIGHT = 3

_EPS = 1.0e-7


def _plane_intersection(
    origins: np.ndarray,
    directions: np.ndarray,
    plane_y: float,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    tmin: float,
) -> np.ndarray:
    dy = directions[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_y - origins[:, 1]) / dy
    valid = (
        (np.abs(dy) > _EPS)
        & (t > tmin)
        & np.isfinite(t)
    )
    px = origins[:, 0] + t * directions[:, 0]
    pz = origins[:, 2] + t * directions[:, 2]
    valid &= (px >= x0) & (px <= x1) & (pz >= z0) & (pz <= z1)
    return np.where(valid, t, np.inf)


def intersect_scene(
    scene: Scene,
    origins: np.ndarray,
    directions: np.ndarray,
    tmin: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest hit distance, kind, and primitive index.

    Kind is one of KIND_TRIANGLE/WATER/GROUND/LIGHT, primitive is the triangle
    index for glass and -1 otherwise.  Rays are one-dimensional batches.
    """

    count = origins.shape[0]
    best_t = np.full(count, np.inf, dtype=np.float64)
    best_kind = np.full(count, KIND_NONE, dtype=np.int8)
    best_prim = np.full(count, -1, dtype=np.int32)

    v0 = scene.triangle_v0
    e1 = scene.triangle_e1
    e2 = scene.triangle_e2
    assert v0 is not None and e1 is not None and e2 is not None

    for triangle_index in range(v0.shape[0]):
        pvec = np.cross(directions, e2[triangle_index])
        det = pvec @ e1[triangle_index]
        valid = np.abs(det) > _EPS
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = origins - v0[triangle_index]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1[triangle_index])
        v = np.einsum("ij,ij->i", directions, qvec) * inv_det
        t = (qvec @ e2[triangle_index]) * inv_det
        valid &= (u >= -_EPS) & (v >= -_EPS) & (u + v <= 1.0 + _EPS) & (t > tmin)
        closer = valid & (t < best_t)
        best_t[closer] = t[closer]
        best_kind[closer] = KIND_TRIANGLE
        best_prim[closer] = triangle_index

    water_t = _plane_intersection(
        origins,
        directions,
        scene.water_y,
        scene.water_x0,
        scene.water_x1,
        scene.water_z0,
        scene.water_z1,
        tmin,
    )
    closer = water_t < best_t
    best_t[closer] = water_t[closer]
    best_kind[closer] = KIND_WATER
    best_prim[closer] = -1

    ground_t = _plane_intersection(
        origins,
        directions,
        scene.ground_y,
        scene.ground_x0,
        scene.ground_x1,
        scene.ground_z0,
        scene.ground_z1,
        tmin,
    )
    closer = ground_t < best_t
    best_t[closer] = ground_t[closer]
    best_kind[closer] = KIND_GROUND
    best_prim[closer] = -1

    # Light rectangle: front face is the side opposite its emission normal.
    assert (
        scene.light_center is not None
        and scene.light_normal is not None
        and scene.light_u_hat is not None
        and scene.light_v_hat is not None
    )
    denom = directions @ scene.light_normal
    rel = scene.light_center - origins
    with np.errstate(divide="ignore", invalid="ignore"):
        light_t = np.einsum("ij,j->i", rel, scene.light_normal) / denom
    local = origins + light_t[:, None] * directions - scene.light_center
    su = np.abs(local @ scene.light_u_hat)
    sv = np.abs(local @ scene.light_v_hat)
    light_valid = (
        (np.abs(denom) > _EPS)
        & (light_t > tmin)
        & np.isfinite(light_t)
        & (denom < 0.0)
        & (su <= scene.light_u_half + _EPS)
        & (sv <= scene.light_v_half + _EPS)
    )
    light_t = np.where(light_valid, light_t, np.inf)
    closer = light_t < best_t
    best_t[closer] = light_t[closer]
    best_kind[closer] = KIND_LIGHT
    best_prim[closer] = -1

    return best_t, best_kind, best_prim


def occluded_by_prism(
    scene: Scene,
    origins: np.ndarray,
    directions: np.ndarray,
    max_distance: np.ndarray,
    tmin: float = 1.0e-5,
) -> np.ndarray:
    """Any-hit test against the glass mesh only."""

    blocked = np.zeros(origins.shape[0], dtype=bool)
    v0 = scene.triangle_v0
    e1 = scene.triangle_e1
    e2 = scene.triangle_e2
    assert v0 is not None and e1 is not None and e2 is not None
    for triangle_index in range(v0.shape[0]):
        active = ~blocked
        if not np.any(active):
            break
        dirs = directions[active]
        orgs = origins[active]
        pvec = np.cross(dirs, e2[triangle_index])
        det = pvec @ e1[triangle_index]
        valid = np.abs(det) > _EPS
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = orgs - v0[triangle_index]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1[triangle_index])
        v = np.einsum("ij,ij->i", dirs, qvec) * inv_det
        t = (qvec @ e2[triangle_index]) * inv_det
        valid &= (
            (u >= -_EPS)
            & (v >= -_EPS)
            & (u + v <= 1.0 + _EPS)
            & (t > tmin)
            & (t < max_distance[active] - 1.0e-5)
        )
        selected = np.zeros_like(blocked)
        selected[active] = valid
        blocked |= selected
    return blocked


def point_on_prism_surface(point: np.ndarray, scene: Scene) -> bool:
    """Used by validation/tests; checks proximity to any triangle."""

    v0 = scene.triangle_v0
    e1 = scene.triangle_e1
    e2 = scene.triangle_e2
    assert v0 is not None and e1 is not None and e2 is not None
    for index in range(v0.shape[0]):
        rel = point - v0[index]
        normal = np.cross(e1[index], e2[index])
        normal /= max(np.linalg.norm(normal), 1.0e-30)
        if abs(float(np.dot(rel, normal))) > 1.0e-6:
            continue
        dot00 = float(np.dot(e1[index], e1[index]))
        dot01 = float(np.dot(e1[index], e2[index]))
        dot02 = float(np.dot(e1[index], rel))
        dot11 = float(np.dot(e2[index], e2[index]))
        dot12 = float(np.dot(e2[index], rel))
        denom = dot00 * dot11 - dot01 * dot01
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        if u >= -1.0e-6 and v >= -1.0e-6 and u + v <= 1.0 + 1.0e-6:
            return True
    return False

