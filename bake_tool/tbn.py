from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, normalized_rows


@dataclass
class TBNResult:
    tangents: np.ndarray
    bitangents: np.ndarray
    normals: np.ndarray
    degenerate_triangles: int
    mirrored_triangles: int
    fallback_vertices: int
    invalid_uv_triangles: int
    adjacent_tangent_flips: int


def _fallback_tangent(normal: np.ndarray) -> np.ndarray:
    axis = np.asarray([1.0, 0.0, 0.0])
    if abs(float(normal[0])) > 0.8:
        axis = np.asarray([0.0, 1.0, 0.0])
    tangent = np.cross(normal, axis)
    length = np.linalg.norm(tangent)
    if length < 1.0e-12:
        axis = np.asarray([0.0, 0.0, 1.0])
        tangent = np.cross(normal, axis)
        length = np.linalg.norm(tangent)
    return tangent / max(length, 1.0e-12)


def build_vertex_tbn(mesh: Mesh) -> TBNResult:
    positions = np.asarray(mesh.positions, dtype=np.float64)
    normals = normalized_rows(mesh.normals)
    uvs = np.asarray(mesh.uvs, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)

    tri_pos = positions[triangles]
    tri_uv = uvs[triangles]
    edge1 = tri_pos[:, 1] - tri_pos[:, 0]
    edge2 = tri_pos[:, 2] - tri_pos[:, 0]
    duv1 = tri_uv[:, 1] - tri_uv[:, 0]
    duv2 = tri_uv[:, 2] - tri_uv[:, 0]

    determinant = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    valid_uv = np.abs(determinant) > 1.0e-12
    safe_det = np.where(valid_uv, determinant, 1.0)
    raw_tangent = (
        edge1 * duv2[:, 1, None] - edge2 * duv1[:, 1, None]
    ) / safe_det[:, None]
    raw_tangent[~valid_uv] = 0.0

    triangle_normal_geo = np.cross(edge1, edge2)
    triangle_area = 0.5 * np.linalg.norm(triangle_normal_geo, axis=1)
    avg_normal = normalized_rows(normals[triangles].mean(axis=1))
    tangent = raw_tangent - avg_normal * np.einsum(
        "ij,ij->i", raw_tangent, avg_normal
    )[:, None]
    tangent_length = np.linalg.norm(tangent, axis=1)
    usable = valid_uv & (tangent_length > 1.0e-10) & (triangle_area > 1.0e-14)

    accumulator = np.zeros_like(positions)
    weights = np.where(usable, triangle_area, 0.0)
    for corner in range(3):
        np.add.at(
            accumulator,
            triangles[:, corner],
            tangent * weights[:, None],
        )

    projected = accumulator - normals * np.einsum(
        "ij,ij->i", accumulator, normals
    )[:, None]
    projected_length = np.linalg.norm(projected, axis=1)
    fallback_mask = projected_length <= 1.0e-10
    tangents = np.zeros_like(projected)
    np.divide(
        projected,
        projected_length[:, None],
        out=tangents,
        where=projected_length[:, None] > 1.0e-10,
    )
    for vertex_index in np.nonzero(fallback_mask)[0]:
        tangents[vertex_index] = _fallback_tangent(normals[vertex_index])

    bitangents = normalized_rows(np.cross(normals, tangents))

    edge_keys: set[tuple[int, int]] = set()
    flips = 0
    for triangle in triangles:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            key = (min(first, second), max(first, second))
            if key in edge_keys:
                continue
            edge_keys.add(key)
            if float(np.dot(tangents[first], tangents[second])) < -0.25:
                flips += 1

    return TBNResult(
        tangents=tangents,
        bitangents=bitangents,
        normals=normals,
        degenerate_triangles=int(np.count_nonzero(~valid_uv)),
        mirrored_triangles=int(np.count_nonzero(determinant < -1.0e-12)),
        fallback_vertices=int(np.count_nonzero(fallback_mask)),
        invalid_uv_triangles=int(np.count_nonzero(~valid_uv)),
        adjacent_tangent_flips=flips,
    )


def per_pixel_tbn(
    barycentric: np.ndarray,
    triangle_indices: np.ndarray,
    mesh: Mesh,
    tbn: TBNResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(mesh.triangles)[triangle_indices]
    weights = np.asarray(barycentric, dtype=np.float64)
    normal = normalized_rows(
        np.einsum("ni,nij->nj", weights, tbn.normals[triangles])
    )
    tangent = np.einsum("ni,nij->nj", weights, tbn.tangents[triangles])
    tangent = tangent - normal * np.einsum("ij,ij->i", tangent, normal)[:, None]
    tangent = normalized_rows(tangent)
    bitangent = normalized_rows(np.cross(normal, tangent))
    return tangent, bitangent, normal

