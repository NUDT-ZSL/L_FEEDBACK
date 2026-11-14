"""Triangle voxelization, spatial hash indexing, and DDA ray traversal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .scene import Scene

Cell = Tuple[int, int, int]


@dataclass(frozen=True)
class RayHit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    albedo: np.ndarray
    triangle_index: int
    cell: Cell


class SpatialHash:
    """Sparse triangle buckets keyed by integer voxel coordinates."""

    def __init__(self) -> None:
        self._buckets: Dict[Cell, List[int]] = {}

    def insert(self, cell: Cell, triangle_index: int) -> None:
        self._buckets.setdefault(cell, []).append(int(triangle_index))

    def get(self, cell: Cell) -> Tuple[int, ...]:
        return tuple(self._buckets.get(cell, ()))

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def entry_count(self) -> int:
        return int(sum(len(values) for values in self._buckets.values()))


class VoxelGrid:
    """Uniform triangular surface voxelization with a sparse spatial index."""

    def __init__(
        self,
        origin: np.ndarray,
        voxel_size: float,
        dims: Tuple[int, int, int],
        occupied: np.ndarray,
        spatial_hash: SpatialHash,
        triangles: np.ndarray,
        triangle_normals: np.ndarray,
        triangle_materials: np.ndarray,
        material_albedos: np.ndarray,
        mesh_bounds_min: np.ndarray,
        mesh_bounds_max: np.ndarray,
    ) -> None:
        self.origin = np.ascontiguousarray(origin, dtype=np.float64)
        self.voxel_size = float(voxel_size)
        self.dims = dims
        self.occupied = occupied
        self.spatial_hash = spatial_hash
        self.triangles = triangles
        self.triangle_normals = triangle_normals
        self.triangle_materials = triangle_materials
        self.material_albedos = material_albedos
        self.mesh_bounds_min = mesh_bounds_min
        self.mesh_bounds_max = mesh_bounds_max
        self.grid_min = self.origin.copy()
        self.grid_max = self.origin + np.asarray(dims, dtype=np.float64) * self.voxel_size
        self._e1 = triangles[:, 1, :] - triangles[:, 0, :]
        self._e2 = triangles[:, 2, :] - triangles[:, 0, :]

    @classmethod
    def build(cls, scene: Scene) -> "VoxelGrid":
        occluder_meshes = [mesh for mesh in scene.meshes if mesh.occluder]
        if not occluder_meshes:
            raise ValueError("scene has no occluder triangles")
        material_ids = sorted(scene.materials.keys())
        material_index = {material_id: index for index, material_id in enumerate(material_ids)}
        material_albedos = np.asarray(
            [scene.materials[material_id].albedo for material_id in material_ids],
            dtype=np.float64,
        )

        triangle_chunks: List[np.ndarray] = []
        material_chunks: List[np.ndarray] = []
        for mesh in occluder_meshes:
            triangle_chunks.append(np.asarray(mesh.triangles, dtype=np.float64))
            material_chunks.append(
                np.full(
                    mesh.triangles.shape[0],
                    material_index[mesh.material_id],
                    dtype=np.int32,
                )
            )
        triangles = np.ascontiguousarray(np.concatenate(triangle_chunks, axis=0))
        triangle_materials = np.ascontiguousarray(np.concatenate(material_chunks))
        vertices = triangles.reshape(-1, 3)
        bounds_min = np.min(vertices, axis=0)
        bounds_max = np.max(vertices, axis=0)
        extent = bounds_max - bounds_min
        max_extent = float(np.max(extent))
        if not math.isfinite(max_extent) or max_extent <= 1.0e-12:
            raise ValueError("mesh bounds have zero extent")

        resolution = int(scene.bake.voxel_resolution)
        padding = int(scene.bake.padding_voxels)
        interior_cells = resolution - 2 * padding
        voxel_size = max_extent / float(interior_cells)
        origin = bounds_min - float(padding) * voxel_size
        dims_array = np.ceil(extent / voxel_size).astype(np.int64) + 2 * padding
        dims = tuple(int(max(1, value)) for value in dims_array)
        occupied = np.zeros(dims, dtype=np.bool_)
        spatial_hash = SpatialHash()

        half = 0.5 * voxel_size
        for triangle_index, triangle in enumerate(triangles):
            tri_min = np.min(triangle, axis=0)
            tri_max = np.max(triangle, axis=0)
            lo = np.floor((tri_min - origin) / voxel_size).astype(np.int64)
            hi = np.floor((tri_max - origin) / voxel_size).astype(np.int64)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, np.asarray(dims, dtype=np.int64) - 1)
            for ix in range(int(lo[0]), int(hi[0]) + 1):
                for iy in range(int(lo[1]), int(hi[1]) + 1):
                    for iz in range(int(lo[2]), int(hi[2]) + 1):
                        cell: Cell = (ix, iy, iz)
                        center = origin + (
                            np.asarray(cell, dtype=np.float64) + 0.5
                        ) * voxel_size
                        if cls._triangle_box_overlap(triangle, center, half):
                            occupied[ix, iy, iz] = True
                            spatial_hash.insert(cell, triangle_index)

        cross_products = np.cross(
            triangles[:, 1, :] - triangles[:, 0, :],
            triangles[:, 2, :] - triangles[:, 0, :],
        )
        lengths = np.linalg.norm(cross_products, axis=1)
        triangle_normals = cross_products / lengths[:, None]
        return cls(
            origin=origin,
            voxel_size=voxel_size,
            dims=dims,
            occupied=occupied,
            spatial_hash=spatial_hash,
            triangles=triangles,
            triangle_normals=np.ascontiguousarray(triangle_normals),
            triangle_materials=triangle_materials,
            material_albedos=np.ascontiguousarray(material_albedos),
            mesh_bounds_min=bounds_min,
            mesh_bounds_max=bounds_max,
        )

    @staticmethod
    def _triangle_box_overlap(
        triangle: np.ndarray,
        center: np.ndarray,
        half_extent: float,
    ) -> bool:
        """Separating-axis triangle/AABB overlap test."""

        v0 = triangle[0] - center
        v1 = triangle[1] - center
        v2 = triangle[2] - center
        vertices = (v0, v1, v2)
        half = np.array([half_extent, half_extent, half_extent], dtype=np.float64)

        for axis in range(3):
            values = (v0[axis], v1[axis], v2[axis])
            if min(values) > half[axis] or max(values) < -half[axis]:
                return False

        edges = (v1 - v0, v2 - v1, v0 - v2)
        box_axes = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        for edge in edges:
            for box_axis in box_axes:
                axis = np.cross(edge, box_axis)
                if float(np.dot(axis, axis)) <= 1.0e-24:
                    continue
                p0 = float(np.dot(v0, axis))
                p1 = float(np.dot(v1, axis))
                p2 = float(np.dot(v2, axis))
                radius = half_extent * (
                    abs(float(axis[0])) + abs(float(axis[1])) + abs(float(axis[2]))
                )
                if min(p0, p1, p2) > radius or max(p0, p1, p2) < -radius:
                    return False

        normal = np.cross(v1 - v0, v2 - v0)
        distance = float(np.dot(normal, v0))
        radius = half_extent * (
            abs(float(normal[0])) + abs(float(normal[1])) + abs(float(normal[2]))
        )
        return abs(distance) <= radius

    @property
    def occupied_voxel_count(self) -> int:
        return int(np.count_nonzero(self.occupied))

    def info(self) -> Dict[str, object]:
        return {
            "origin": [float(value) for value in self.origin],
            "voxel_size": float(self.voxel_size),
            "dims": [int(value) for value in self.dims],
            "occupied_voxels": self.occupied_voxel_count,
            "spatial_hash_buckets": self.spatial_hash.bucket_count,
            "spatial_hash_entries": self.spatial_hash.entry_count,
            "triangle_count": int(self.triangles.shape[0]),
        }

    def cell_at(self, point: np.ndarray) -> Optional[Cell]:
        local = (np.asarray(point, dtype=np.float64) - self.origin) / self.voxel_size
        cell = np.floor(local).astype(np.int64)
        if np.any(cell < 0) or np.any(cell >= np.asarray(self.dims)):
            return None
        return (int(cell[0]), int(cell[1]), int(cell[2]))

    def point_is_occupied(self, point: np.ndarray) -> bool:
        cell = self.cell_at(point)
        if cell is None:
            return False
        return bool(self.occupied[cell[0], cell[1], cell[2]])

    def _ray_grid_interval(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        t_limit: float,
    ) -> Optional[Tuple[float, float]]:
        t_enter = 0.0
        t_exit = float(t_limit)
        for axis in range(3):
            component = float(direction[axis])
            low = float(self.grid_min[axis])
            high = float(self.grid_max[axis])
            start = float(origin[axis])
            if abs(component) <= 1.0e-15:
                if start < low or start > high:
                    return None
                continue
            t0 = (low - start) / component
            t1 = (high - start) / component
            if t0 > t1:
                t0, t1 = t1, t0
            t_enter = max(t_enter, t0)
            t_exit = min(t_exit, t1)
            if t_enter > t_exit:
                return None
        if t_exit < 0.0 or t_enter > t_limit:
            return None
        return max(0.0, t_enter), min(t_exit, t_limit)

    def trace_primary(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = 1.0e30,
    ) -> Optional[RayHit]:
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        interval = self._ray_grid_interval(o, d, max_distance)
        if interval is None:
            return None
        t_enter, t_limit = interval
        start = o + d * t_enter
        local = (start - self.origin) / self.voxel_size
        cell_array = np.floor(local).astype(np.int64)
        cell_array = np.minimum(
            np.maximum(cell_array, 0), np.asarray(self.dims, dtype=np.int64) - 1
        )
        cell = (int(cell_array[0]), int(cell_array[1]), int(cell_array[2]))

        step = np.where(d > 0.0, 1, -1).astype(np.int64)
        t_max = np.full(3, np.inf, dtype=np.float64)
        t_delta = np.full(3, np.inf, dtype=np.float64)
        for axis in range(3):
            component = float(d[axis])
            if abs(component) <= 1.0e-15:
                continue
            if component > 0.0:
                boundary = self.origin[axis] + (cell[axis] + 1) * self.voxel_size
            else:
                boundary = self.origin[axis] + cell[axis] * self.voxel_size
            t_max[axis] = (boundary - o[axis]) / component
            t_delta[axis] = self.voxel_size / abs(component)

        current_t = t_enter
        iteration_limit = int(sum(self.dims) + 8)
        for _ in range(iteration_limit):
            cell_exit = min(t_limit, float(np.min(t_max)))
            if bool(self.occupied[cell]):
                candidates = self.spatial_hash.get(cell)
                if candidates:
                    hit = self._intersect_triangles(
                        o,
                        d,
                        candidates,
                        max(current_t, 0.0) + 1.0e-9,
                        cell_exit + 1.0e-9,
                        cell,
                    )
                    if hit is not None:
                        return hit
            if cell_exit >= t_limit:
                break
            axis = int(np.argmin(t_max))
            if not math.isfinite(float(t_max[axis])):
                break
            current_t = float(t_max[axis])
            t_max[axis] += t_delta[axis]
            cell_array[axis] += step[axis]
            if (
                cell_array[axis] < 0
                or cell_array[axis] >= self.dims[axis]
            ):
                break
            cell = (
                int(cell_array[0]),
                int(cell_array[1]),
                int(cell_array[2]),
            )
        return None

    def _intersect_triangles(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        candidates: Sequence[int],
        t_min: float,
        t_max: float,
        cell: Cell,
    ) -> Optional[RayHit]:
        ids = np.asarray(candidates, dtype=np.int64)
        vertices = self.triangles[ids]
        v0 = vertices[:, 0, :]
        e1 = self._e1[ids]
        e2 = self._e2[ids]
        pvec = np.cross(direction[None, :], e2)
        determinant = np.einsum("ij,ij->i", e1, pvec)
        valid = np.abs(determinant) > 1.0e-12
        if not np.any(valid):
            return None
        inv_det = np.zeros_like(determinant)
        inv_det[valid] = 1.0 / determinant[valid]
        tvec = origin[None, :] - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", direction[None, :], qvec) * inv_det
        t = np.einsum("ij,ij->i", e2, qvec) * inv_det
        valid &= (u >= -1.0e-10) & (v >= -1.0e-10) & (u + v <= 1.0 + 1.0e-10)
        valid &= (t >= t_min) & (t <= t_max)
        if not np.any(valid):
            return None
        valid_indices = np.flatnonzero(valid)
        best_local = int(valid_indices[int(np.argmin(t[valid]))])
        triangle_index = int(ids[best_local])
        best_t = float(t[best_local])
        point = origin + direction * best_t
        normal = self.triangle_normals[triangle_index].copy()
        if float(np.dot(normal, direction)) > 0.0:
            normal = -normal
        material_index = int(self.triangle_materials[triangle_index])
        return RayHit(
            t=best_t,
            point=point,
            normal=normal,
            albedo=self.material_albedos[material_index].copy(),
            triangle_index=triangle_index,
            cell=cell,
        )

    def is_occluded(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        skip_start_cell: bool = True,
    ) -> bool:
        o = np.asarray(origin, dtype=np.float64)
        delta = np.asarray(target, dtype=np.float64) - o
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            return False
        d = delta / distance
        interval = self._ray_grid_interval(o, d, distance)
        if interval is None:
            return False
        t_enter, t_limit = interval
        origin_inside_grid = t_enter <= 1.0e-9
        start = o + d * t_enter
        local = (start - self.origin) / self.voxel_size
        cell_array = np.floor(local).astype(np.int64)
        cell_array = np.minimum(
            np.maximum(cell_array, 0), np.asarray(self.dims, dtype=np.int64) - 1
        )
        cell = (int(cell_array[0]), int(cell_array[1]), int(cell_array[2]))
        start_cell = cell

        step = np.where(d > 0.0, 1, -1).astype(np.int64)
        t_max = np.full(3, np.inf, dtype=np.float64)
        t_delta = np.full(3, np.inf, dtype=np.float64)
        for axis in range(3):
            component = float(d[axis])
            if abs(component) <= 1.0e-15:
                continue
            if component > 0.0:
                boundary = self.origin[axis] + (cell[axis] + 1) * self.voxel_size
            else:
                boundary = self.origin[axis] + cell[axis] * self.voxel_size
            t_max[axis] = (boundary - o[axis]) / component
            t_delta[axis] = self.voxel_size / abs(component)

        iteration_limit = int(sum(self.dims) + 8)
        for iteration in range(iteration_limit):
            skip_initial = skip_start_cell and origin_inside_grid and iteration == 0
            if self.occupied[cell] and not skip_initial:
                return True
            if min(t_limit, float(np.min(t_max))) >= t_limit:
                break
            axis = int(np.argmin(t_max))
            if not math.isfinite(float(t_max[axis])):
                break
            t_max[axis] += t_delta[axis]
            cell_array[axis] += step[axis]
            if (
                cell_array[axis] < 0
                or cell_array[axis] >= self.dims[axis]
            ):
                break
            cell = (
                int(cell_array[0]),
                int(cell_array[1]),
                int(cell_array[2]),
            )
        return False

    def first_occupied_cells(self, count: int = 8) -> List[Cell]:
        indices = np.argwhere(self.occupied)
        return [
            (int(index[0]), int(index[1]), int(index[2]))
            for index in indices[: int(count)]
        ]
