from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshGroup:
    material_index: int
    material_name: str
    index_offset: int
    index_count: int
    vertex_offset: int
    vertex_count: int

    @property
    def index_byte_offset(self) -> int:
        return self.index_offset * np.dtype(np.uint32).itemsize


@dataclass
class MeshData:
    vertices: np.ndarray
    indices: np.ndarray
    groups: tuple[MeshGroup, ...]

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.indices = np.ascontiguousarray(self.indices, dtype=np.uint32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 7:
            raise ValueError("vertices must have position(3), normal(3), material(1)")
        if self.indices.ndim != 1 or len(self.indices) % 3:
            raise ValueError("indices must be a triangle index list")
        if len(self.vertices) and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= len(self.vertices)
        ):
            raise ValueError("index out of range")
        expected = 0
        for group in self.groups:
            if group.index_offset != expected:
                raise ValueError("material groups must be contiguous and ordered")
            expected += group.index_count
        if expected != len(self.indices):
            raise ValueError("group index counts do not cover the index buffer")

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self.vertices[:, :3]
        return positions.min(axis=0), positions.max(axis=0)

    def triangles(self) -> np.ndarray:
        return self.vertices[self.indices, :3].reshape(-1, 3, 3)

    def checksum(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.vertices.tobytes())
        digest.update(self.indices.tobytes())
        for group in self.groups:
            digest.update(
                f"{group.material_index}:{group.index_offset}:{group.index_count}:"
                f"{group.vertex_offset}:{group.vertex_count}".encode("ascii")
            )
        return digest.hexdigest()


class MeshBuilder:
    """Builds a single VBO/IBO while retaining material draw ranges."""

    def __init__(self) -> None:
        self._vertices: dict[int, list[tuple[float, float, float, float, float, float, float]]] = {}
        self._indices: dict[int, list[int]] = {}
        self._material_names: dict[int, str] = {}

    def add_triangle(
        self,
        material_index: int,
        material_name: str,
        points: np.ndarray,
        normals: np.ndarray,
    ) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(3, 3)
        normals = np.asarray(normals, dtype=np.float64).reshape(3, 3)
        geometric = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(geometric, normals.mean(axis=0))) < 0.0:
            points = points[[0, 2, 1]]
            normals = normals[[0, 2, 1]]
        local = self._vertices.setdefault(material_index, [])
        index_list = self._indices.setdefault(material_index, [])
        self._material_names[material_index] = material_name
        base = len(local)
        for point, normal in zip(points, normals):
            local.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(material_index),
                )
            )
        index_list.extend((base, base + 1, base + 2))

    def add_quad(
        self,
        material_index: int,
        material_name: str,
        points: np.ndarray,
        expected_normal: np.ndarray,
    ) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(4, 3)
        normal = np.asarray(expected_normal, dtype=np.float64)
        normal /= max(1.0e-12, float(np.linalg.norm(normal)))
        normals = np.repeat(normal[None, :], 4, axis=0)
        self.add_triangle(material_index, material_name, points[[0, 1, 2]], normals[[0, 1, 2]])
        self.add_triangle(material_index, material_name, points[[0, 2, 3]], normals[[0, 2, 3]])

    def finalize(self) -> MeshData:
        vertices: list[tuple[float, ...]] = []
        indices: list[int] = []
        groups: list[MeshGroup] = []
        vertex_offset = 0
        for material_index in sorted(self._vertices):
            local_vertices = self._vertices[material_index]
            local_indices = self._indices[material_index]
            if not local_indices:
                continue
            group_vertex_offset = vertex_offset
            vertices.extend(local_vertices)
            indices.extend(index + group_vertex_offset for index in local_indices)
            groups.append(
                MeshGroup(
                    material_index=material_index,
                    material_name=self._material_names[material_index],
                    index_offset=len(indices) - len(local_indices),
                    index_count=len(local_indices),
                    vertex_offset=group_vertex_offset,
                    vertex_count=len(local_vertices),
                )
            )
            vertex_offset += len(local_vertices)
        if not vertices:
            raise ValueError("cannot finalize an empty mesh")
        return MeshData(
            vertices=np.asarray(vertices, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.uint32),
            groups=tuple(groups),
        )

