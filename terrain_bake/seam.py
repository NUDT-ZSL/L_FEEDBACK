"""Construction and verification of conforming LOD transition strips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import E_GEOMETRY_INVALID, BakeError
from .mesh import TerrainMesh, topology_summary


@dataclass(frozen=True)
class SeamResult:
    mesh: TerrainMesh
    high_level: int
    low_level: int
    high_boundary_vertices: int
    low_boundary_vertices: int
    stitch_triangles: int
    t_junction_count: int
    high_vertices_reused: bool
    combined_topology: dict[str, object]


def _boundary_at_z(
    mesh: TerrainMesh,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = 1e-9
    indices = np.flatnonzero(np.abs(mesh.vertices[:, 2] - z_value) <= tolerance)
    if len(indices) < 2:
        raise BakeError(
            E_GEOMETRY_INVALID,
            "LOD boundary does not contain enough vertices for stitching",
            {"z": float(z_value), "vertices": int(len(indices))},
        )
    order = np.lexsort((mesh.vertices[indices, 2], mesh.vertices[indices, 0]))
    indices = indices[order]
    return indices, mesh.vertices[indices]


def _count_t_junctions_on_polylines(
    upper: np.ndarray,
    lower: np.ndarray,
) -> int:
    """Count vertices lying strictly inside an opposite-boundary segment."""

    count = 0
    epsilon = 1e-10
    for points, segments in ((upper, lower), (lower, upper)):
        for point in points:
            for start, end in zip(segments[:-1], segments[1:]):
                direction = end - start
                length_squared = float(np.dot(direction, direction))
                if length_squared <= epsilon:
                    continue
                t = float(np.dot(point - start, direction) / length_squared)
                if t <= 1e-8 or t >= 1.0 - 1e-8:
                    continue
                closest = start + t * direction
                if float(np.linalg.norm(point - closest)) <= epsilon:
                    count += 1
    return count


def build_transition_mesh(
    high_mesh: TerrainMesh,
    low_mesh: TerrainMesh,
    high_level: int,
    low_level: int,
    gap: float = 0.012,
) -> SeamResult:
    """Place adjacent LOD blocks end-to-end and triangulate the gap.

    The high-detail block keeps its exact coordinates.  The low block is
    translated along -Z by one domain extent plus ``gap`` so both blocks are
    adjacent.  Every stitch triangle references existing high/low vertices;
    no midpoint or newly evaluated height is introduced.
    """

    high_min_z = float(np.min(high_mesh.vertices[:, 2]))
    low_max_z = float(np.max(low_mesh.vertices[:, 2]))
    high_indices, high_boundary = _boundary_at_z(high_mesh, high_min_z)
    low_indices, low_boundary_original = _boundary_at_z(low_mesh, low_max_z)

    high_extent = sorted(
        {
            round(float(value), 12)
            for value in high_boundary[:, 0]
        }
    )
    if high_extent[0] > 1e-8 or high_extent[-1] < 1.0 - 1e-8:
        raise BakeError(
            E_GEOMETRY_INVALID,
            "high-detail LOD boundary does not span the full X extent",
        )

    shift_z = -(low_max_z - high_min_z + gap)
    low_shifted_vertices = low_mesh.vertices.copy()
    low_shifted_vertices[:, 2] += shift_z
    low_boundary = low_boundary_original + np.array([0.0, 0.0, shift_z])

    if abs(high_boundary[0, 0] - low_boundary[0, 0]) > 1e-9:
        raise BakeError(E_GEOMETRY_INVALID, "LOD seam start points do not align")
    if abs(high_boundary[-1, 0] - low_boundary[-1, 0]) > 1e-9:
        raise BakeError(E_GEOMETRY_INVALID, "LOD seam end points do not align")

    t_junctions = _count_t_junctions_on_polylines(high_boundary, low_boundary)
    if t_junctions:
        raise BakeError(
            E_GEOMETRY_INVALID,
            "LOD transition contains T-junctions",
            {"t_junction_count": t_junctions},
        )

    high_vertex_count = high_mesh.vertex_count
    combined_vertices = np.vstack((high_mesh.vertices, low_shifted_vertices))
    combined_weights = np.vstack((high_mesh.weights, low_mesh.weights))
    combined_faces: list[tuple[int, int, int]] = [
        tuple(int(index) for index in face) for face in high_mesh.faces.tolist()
    ]
    combined_faces.extend(
        tuple(int(index) + high_vertex_count for index in face)
        for face in low_mesh.faces.tolist()
    )

    low_ids = low_indices + high_vertex_count
    high_ids = high_indices
    high_count = len(high_ids)
    low_count = len(low_ids)
    i = 0
    j = 0
    stitch_faces: list[tuple[int, int, int]] = []
    while i < high_count - 1 or j < low_count - 1:
        advance_high = j >= low_count - 1 or (
            i < high_count - 1
            and high_boundary[i + 1, 0] <= low_boundary[j + 1, 0] + 1e-12
        )
        if advance_high:
            triangle = (int(high_ids[i]), int(high_ids[i + 1]), int(low_ids[j]))
            i += 1
        else:
            triangle = (int(high_ids[i]), int(low_ids[j + 1]), int(low_ids[j]))
            j += 1
        stitch_faces.append(triangle)

    for triangle in stitch_faces:
        points = combined_vertices[np.asarray(triangle)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if normal[1] <= 0.0:
            raise BakeError(
                E_GEOMETRY_INVALID,
                "stitch triangle winding is not terrain-upward",
            )
    combined_faces.extend(stitch_faces)

    combined = TerrainMesh(
        vertices=np.ascontiguousarray(combined_vertices),
        faces=np.ascontiguousarray(np.asarray(combined_faces, dtype=np.int32)),
        weights=np.ascontiguousarray(combined_weights),
    )
    combined_topology = topology_summary(combined, level=None)

    high_reused = bool(
        np.array_equal(combined.vertices[high_ids], high_mesh.vertices[high_indices])
    )
    if not high_reused:
        raise BakeError(
            E_GEOMETRY_INVALID,
            "stitch vertices do not reuse exact high-detail coordinates",
        )

    return SeamResult(
        mesh=combined,
        high_level=int(high_level),
        low_level=int(low_level),
        high_boundary_vertices=int(high_count),
        low_boundary_vertices=int(low_count),
        stitch_triangles=int(len(stitch_faces)),
        t_junction_count=int(t_junctions),
        high_vertices_reused=high_reused,
        combined_topology=combined_topology,
    )
