from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh


@dataclass
class RayCastResult:
    hit_t: np.ndarray
    hit_triangle: np.ndarray
    hit_barycentric: np.ndarray
    leaf_count: int
    candidate_leaf_tests: int
    ray_triangle_tests: int


class _LeafBVH:
    def __init__(self, mesh: Mesh, max_leaf_size: int = 8) -> None:
        self.positions = np.asarray(mesh.positions, dtype=np.float64)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int32)
        self.max_leaf_size = int(max_leaf_size)
        self.leaf_triangle_ids: list[np.ndarray] = []
        self.leaf_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        self._build(np.arange(self.triangles.shape[0], dtype=np.int32), 0)

        self.leaf_vertices: list[np.ndarray] = []
        self.leaf_v0: list[np.ndarray] = []
        self.leaf_e1: list[np.ndarray] = []
        self.leaf_e2: list[np.ndarray] = []
        for triangle_ids in self.leaf_triangle_ids:
            vertices = self.positions[self.triangles[triangle_ids]]
            self.leaf_vertices.append(vertices)
            self.leaf_v0.append(vertices[:, 0])
            self.leaf_e1.append(vertices[:, 1] - vertices[:, 0])
            self.leaf_e2.append(vertices[:, 2] - vertices[:, 0])

    def _build(self, triangle_ids: np.ndarray, depth: int) -> None:
        triangle_vertices = self.positions[self.triangles[triangle_ids]]
        bounds_min = triangle_vertices.reshape(-1, 3).min(axis=0)
        bounds_max = triangle_vertices.reshape(-1, 3).max(axis=0)
        if triangle_ids.size <= self.max_leaf_size or depth >= 24:
            self.leaf_triangle_ids.append(triangle_ids.copy())
            self.leaf_bounds.append((bounds_min, bounds_max))
            return

        centroids = triangle_vertices.mean(axis=1)
        extents = bounds_max - bounds_min
        axis = int(np.argmax(extents))
        order = np.argsort(centroids[:, axis], kind="stable")
        ordered = triangle_ids[order]
        middle = ordered.size // 2
        self._build(ordered[:middle], depth + 1)
        self._build(ordered[middle:], depth + 1)


def _slab_interval(
    origins: np.ndarray,
    inverse_directions: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = (bounds_min[None, :] - origins) * inverse_directions
    second = (bounds_max[None, :] - origins) * inverse_directions
    near = np.max(np.minimum(first, second), axis=1)
    far = np.min(np.maximum(first, second), axis=1)
    return near, far


def build_leaf_bvh(mesh: Mesh, max_leaf_size: int = 8) -> _LeafBVH:
    return _LeafBVH(mesh, max_leaf_size=max_leaf_size)


def cast_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    high_mesh: Mesh,
    bvh: _LeafBVH | None = None,
) -> RayCastResult:
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    ray_count = origins.shape[0]
    if bvh is None:
        bvh = _LeafBVH(high_mesh)

    inverse_directions = np.empty_like(directions)
    nonzero = np.abs(directions) > 1.0e-15
    np.divide(1.0, directions, out=inverse_directions, where=nonzero)
    inverse_directions[~nonzero] = 1.0e30

    best_t = np.full(ray_count, np.inf, dtype=np.float64)
    best_triangle = np.full(ray_count, -1, dtype=np.int32)
    best_uv = np.zeros((ray_count, 2), dtype=np.float64)
    candidate_leaf_tests = 0
    ray_triangle_tests = 0
    epsilon = 1.0e-6

    for leaf_index in range(len(bvh.leaf_triangle_ids)):
        bounds_min, bounds_max = bvh.leaf_bounds[leaf_index]
        near, far = _slab_interval(origins, inverse_directions, bounds_min, bounds_max)
        candidates = (
            (far >= np.maximum(near, 0.0))
            & (near <= best_t)
            & (far > epsilon)
        )
        ray_ids = np.nonzero(candidates)[0]
        if ray_ids.size == 0:
            continue
        candidate_leaf_tests += int(ray_ids.size)

        v0 = bvh.leaf_v0[leaf_index]
        edge1 = bvh.leaf_e1[leaf_index]
        edge2 = bvh.leaf_e2[leaf_index]
        ray_origins = origins[ray_ids]
        ray_directions = directions[ray_ids]

        pvec = np.cross(ray_directions[:, None, :], edge2[None, :, :])
        determinant = np.einsum("ckj,ckj->ck", edge1[None, :, :], pvec)
        ray_triangle_tests += int(determinant.size)
        valid = np.abs(determinant) > 1.0e-12
        inverse_determinant = np.zeros_like(determinant)
        np.divide(1.0, determinant, out=inverse_determinant, where=valid)

        tvec = ray_origins[:, None, :] - v0[None, :, :]
        u = np.einsum("ckj,ckj->ck", tvec, pvec) * inverse_determinant
        qvec = np.cross(tvec, edge1[None, :, :])
        v = np.einsum("ckj,ckj->ck", ray_directions[:, None, :], qvec) * inverse_determinant
        t = np.einsum("ckj,ckj->ck", edge2[None, :, :], qvec) * inverse_determinant

        valid &= (
            (u >= -1.0e-9)
            & (v >= -1.0e-9)
            & ((u + v) <= 1.0 + 1.0e-9)
            & (t > epsilon)
            & np.isfinite(t)
        )
        local_t = np.where(valid, t, np.inf)
        local_index = np.argmin(local_t, axis=1)
        local_best = local_t[np.arange(ray_ids.size), local_index]
        update = local_best < best_t[ray_ids] - 1.0e-12
        if np.any(update):
            updated_rays = ray_ids[update]
            selected = local_index[update]
            best_t[updated_rays] = local_best[update]
            best_triangle[updated_rays] = bvh.leaf_triangle_ids[leaf_index][selected]
            best_uv[updated_rays, 0] = u[update, selected]
            best_uv[updated_rays, 1] = v[update, selected]

    return RayCastResult(
        hit_t=best_t,
        hit_triangle=best_triangle,
        hit_barycentric=best_uv,
        leaf_count=len(bvh.leaf_triangle_ids),
        candidate_leaf_tests=candidate_leaf_tests,
        ray_triangle_tests=ray_triangle_tests,
    )


def interpolate_hit_normals(
    high_mesh: Mesh,
    hit_triangle: np.ndarray,
    hit_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hit_mask = hit_triangle >= 0
    normals = np.zeros((hit_triangle.size, 3), dtype=np.float64)
    if not np.any(hit_mask):
        return normals, hit_mask

    triangles = np.asarray(high_mesh.triangles, dtype=np.int64)[hit_triangle[hit_mask]]
    u = hit_uv[hit_mask, 0]
    v = hit_uv[hit_mask, 1]
    w = 1.0 - u - v
    weights = np.stack((w, u, v), axis=1)
    interpolated = np.einsum(
        "ni,nij->nj",
        weights,
        np.asarray(high_mesh.normals, dtype=np.float64)[triangles],
    )
    lengths = np.linalg.norm(interpolated, axis=1, keepdims=True)
    fallback = np.cross(
        high_mesh.positions[triangles[:, 1]] - high_mesh.positions[triangles[:, 0]],
        high_mesh.positions[triangles[:, 2]] - high_mesh.positions[triangles[:, 0]],
    )
    fallback_lengths = np.linalg.norm(fallback, axis=1, keepdims=True)
    np.divide(fallback, fallback_lengths, out=fallback, where=fallback_lengths > 1.0e-14)
    np.divide(interpolated, lengths, out=interpolated, where=lengths > 1.0e-14)
    bad = lengths[:, 0] <= 1.0e-14
    interpolated[bad] = fallback[bad]
    normals[hit_mask] = interpolated
    return normals, hit_mask

