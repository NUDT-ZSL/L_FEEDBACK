"""Terrain mesh construction, edge-collapse LODs and topology validation."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .coordinates import HEIGHT_SCALE, heightfield_grid
from .errors import E_GEOMETRY_INVALID, BakeError


@dataclass
class TerrainMesh:
    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray

    def copy(self) -> "TerrainMesh":
        return TerrainMesh(
            self.vertices.copy(),
            self.faces.copy(),
            self.weights.copy(),
        )

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class SimplificationResult:
    mesh: TerrainMesh
    target_vertices: int
    reached_target: bool
    collapse_count: int
    relative_error: float
    error_threshold: float


def build_base_mesh(
    heightfield: np.ndarray,
    weights: np.ndarray,
) -> TerrainMesh:
    rows, cols = heightfield.shape
    x_grid, z_grid = heightfield_grid(rows, cols)
    vertices = np.column_stack(
        (
            x_grid.reshape(-1),
            (heightfield * HEIGHT_SCALE).reshape(-1),
            z_grid.reshape(-1),
        )
    ).astype(np.float64, copy=False)

    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = a + cols
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return TerrainMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32),
        weights=np.ascontiguousarray(weights.reshape(-1, weights.shape[2])),
    )


def compute_vertex_normals(mesh: TerrainMesh) -> np.ndarray:
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-15
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = 0.0

    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 1.0, 0.0])
    return normals


def tangent_basis(mesh: TerrainMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T, B, N with T cross B == N for terrain preview shading."""

    normals = compute_vertex_normals(mesh)
    tangent = np.broadcast_to(
        np.array([1.0, 0.0, 0.0]), mesh.vertices.shape
    ).copy()
    tangent -= normals * np.sum(normals * tangent, axis=1, keepdims=True)
    lengths = np.linalg.norm(tangent, axis=1)
    valid = lengths > 1e-12
    tangent[valid] /= lengths[valid, None]
    tangent[~valid] = np.array([0.0, 0.0, 1.0])
    bitangent = np.cross(normals, tangent)
    return tangent, bitangent, normals


def _mesh_scale(vertices: np.ndarray) -> float:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    return max(float(np.linalg.norm(maximum - minimum)), 1e-12)


def topology_summary(mesh: TerrainMesh, level: int | None = None) -> dict[str, object]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    label = f"LOD {level}" if level is not None else "mesh"
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} has no triangular faces",
        )
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise BakeError(E_GEOMETRY_INVALID, f"{label} has out-of-range indices")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        raise BakeError(E_GEOMETRY_INVALID, f"{label} has degenerate index triplets")
    if not np.all(np.isfinite(vertices)):
        raise BakeError(E_GEOMETRY_INVALID, f"{label} has non-finite vertex coordinates")

    tri = vertices[faces]
    cross_products = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    area_epsilon = max(_mesh_scale(vertices) ** 2 * 1e-12, 1e-16)
    if np.any(areas <= area_epsilon):
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} contains zero-area or degenerate triangles",
            {"minimum_area": float(np.min(areas))},
        )

    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(faces.tolist()):
        a, b, c = face
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append((u, v, face_index))

    non_manifold_edges = 0
    orientation_errors = 0
    boundary_edges: list[tuple[int, int]] = []
    for key, uses in edge_faces.items():
        if len(uses) > 2:
            non_manifold_edges += 1
        if len(uses) == 1:
            boundary_edges.append(key)
        if len(uses) == 2 and uses[0][:2] == uses[1][:2]:
            orientation_errors += 1
    if non_manifold_edges:
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} has non-manifold edges",
            {"non_manifold_edges": int(non_manifold_edges)},
        )
    if orientation_errors:
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} has inconsistent triangle winding across shared edges",
            {"orientation_errors": int(orientation_errors)},
        )

    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for u, v in edge_faces:
        adjacency[u].append(v)
        adjacency[v].append(u)
    active = np.unique(faces.reshape(-1))
    if len(active) != len(vertices):
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} contains vertices that are not referenced by any face",
        )
    seen: set[int] = set()
    stack = [int(active[0])]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current])
    if len(seen) != len(active):
        raise BakeError(E_GEOMETRY_INVALID, f"{label} is not a connected mesh")

    boundary_adjacency: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        boundary_adjacency.setdefault(u, []).append(v)
        boundary_adjacency.setdefault(v, []).append(u)
    if any(len(neighbors) != 2 for neighbors in boundary_adjacency.values()):
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} has a non-simple boundary",
        )
    boundary_loops = 0
    unvisited = set(boundary_adjacency)
    while unvisited:
        boundary_loops += 1
        start = min(unvisited)
        previous = -1
        current = start
        while True:
            unvisited.discard(current)
            choices = [n for n in boundary_adjacency[current] if n != previous]
            if not choices:
                break
            nxt = min(choices)
            previous, current = current, nxt
            if current == start:
                break
    if boundary_loops != 1:
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} must have exactly one boundary loop",
            {"boundary_loops": int(boundary_loops)},
        )

    euler = int(len(active) - len(edge_faces) + len(faces))
    if euler != 1:
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} is not a disk topology",
            {"euler_characteristic": euler},
        )

    return {
        "vertex_count": int(len(active)),
        "face_count": int(len(faces)),
        "edge_count": int(len(edge_faces)),
        "boundary_edges": int(len(boundary_edges)),
        "boundary_loops": int(boundary_loops),
        "euler_characteristic": euler,
        "non_manifold_edges": 0,
        "orientation_errors": 0,
        "minimum_area": float(np.min(areas)),
        "manifold": True,
    }


def _build_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    count = len(vertices)
    quadrics = np.zeros((count, 4, 4), dtype=np.float64)
    triangle = vertices[faces]
    normals = np.cross(
        triangle[:, 1] - triangle[:, 0],
        triangle[:, 2] - triangle[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    offsets = -np.sum(normals * triangle[:, 0], axis=1)
    planes = np.column_stack((normals, offsets))
    outer = planes[:, :, None] * planes[:, None, :]
    flat = outer.reshape(-1, 16)
    for corner in range(3):
        np.add.at(quadrics.reshape(count, 16), faces[:, corner], flat)
    return quadrics


def _eval_quadric(quadric: np.ndarray, position: np.ndarray) -> float:
    homogeneous = np.array([position[0], position[1], position[2], 1.0])
    return max(float(homogeneous @ quadric @ homogeneous), 0.0)


def _vertices_in_faces(
    face_ids: Iterable[int],
    faces: list[tuple[int, int, int] | None],
) -> set[int]:
    result: set[int] = set()
    for face_id in face_ids:
        face = faces[face_id]
        if face is not None:
            result.update(face)
    return result


def _make_candidate(
    u: int,
    v: int,
    vertices: np.ndarray,
    alive: np.ndarray,
    face_sets: list[set[int]],
    faces: list[tuple[int, int, int] | None],
    quadrics: np.ndarray,
    versions: np.ndarray,
    pinned: set[int],
    mesh_scale: float,
) -> tuple[float, int, int, int, int] | None:
    if u == v or not alive[u] or not alive[v]:
        return None
    if u in pinned or v in pinned:
        return None
    incident = face_sets[u] & face_sets[v]
    if len(incident) not in (1, 2):
        return None

    eu = vertices[u]
    ev = vertices[v]
    eps = 1e-12
    boundary_u = bool(
        abs(eu[0]) <= eps
        or abs(eu[0] - 1.0) <= eps
        or abs(eu[2]) <= eps
        or abs(eu[2] - 1.0) <= eps
    )
    boundary_v = bool(
        abs(ev[0]) <= eps
        or abs(ev[0] - 1.0) <= eps
        or abs(ev[2]) <= eps
        or abs(ev[2] - 1.0) <= eps
    )
    if boundary_u != boundary_v:
        return None
    if boundary_u and len(incident) != 1:
        return None
    if not boundary_u and len(incident) != 2:
        return None

    opposite: set[int] = set()
    for face_id in incident:
        face = faces[face_id]
        assert face is not None
        remaining = set(face) - {u, v}
        if len(remaining) != 1:
            return None
        opposite.update(remaining)

    neighbors_u = _vertices_in_faces(face_sets[u], faces) - {u, v}
    neighbors_v = _vertices_in_faces(face_sets[v], faces) - {u, v}
    if neighbors_u & neighbors_v != opposite:
        return None

    combined = quadrics[u] + quadrics[v]
    cost_u = _eval_quadric(combined, vertices[u])
    cost_v = _eval_quadric(combined, vertices[v])
    if cost_u < cost_v - 1e-15:
        keep, remove, cost = u, v, cost_u
    elif cost_v < cost_u - 1e-15:
        keep, remove, cost = v, u, cost_v
    elif u < v:
        keep, remove, cost = u, v, cost_u
    else:
        keep, remove, cost = v, u, cost_v
    relative_error = math.sqrt(max(cost, 0.0)) / mesh_scale
    return relative_error, keep, remove, int(versions[keep]), int(versions[remove])


def _collapse_edge(
    keep: int,
    remove: int,
    alive: np.ndarray,
    face_sets: list[set[int]],
    faces: list[tuple[int, int, int] | None],
    quadrics: np.ndarray,
    versions: np.ndarray,
) -> bool:
    incident = face_sets[keep] & face_sets[remove]
    if not incident:
        return False

    existing_keys = {
        tuple(sorted(faces[face_id]))
        for face_id in face_sets[keep]
        if faces[face_id] is not None
    }
    rewired: list[tuple[int, tuple[int, int, int], tuple[int, int, int]]] = []
    for face_id in sorted(face_sets[remove]):
        face = faces[face_id]
        if face is None or face_id in incident:
            continue
        replaced = tuple(keep if index == remove else index for index in face)
        key = tuple(sorted(replaced))
        if key in existing_keys:
            return False
        existing_keys.add(key)
        rewired.append((face_id, face, replaced))

    for face_id in incident:
        face = faces[face_id]
        if face is None:
            continue
        for index in set(face):
            face_sets[index].discard(face_id)
        faces[face_id] = None

    for face_id, old_face, new_face in rewired:
        for index in set(old_face):
            face_sets[index].discard(face_id)
        faces[face_id] = new_face
        for index in set(new_face):
            face_sets[index].add(face_id)

    face_sets[keep] |= face_sets[remove]
    face_sets[remove].clear()
    alive[remove] = False
    quadrics[keep] += quadrics[remove]
    quadrics[remove].fill(0.0)
    versions[keep] += 1
    versions[remove] += 1
    return True


def simplify_mesh(
    mesh: TerrainMesh,
    target_vertices: int,
    max_relative_error: float,
    label: str,
) -> SimplificationResult:
    """Apply deterministic endpoint edge collapses under a relative QEM cap."""

    vertices = mesh.vertices.copy()
    weights = mesh.weights.copy()
    original_faces = mesh.faces.copy()
    vertex_count = len(vertices)
    target_vertices = max(int(target_vertices), 4)
    if target_vertices >= vertex_count:
        topology_summary(mesh)
        return SimplificationResult(
            mesh=mesh.copy(),
            target_vertices=target_vertices,
            reached_target=True,
            collapse_count=0,
            relative_error=0.0,
            error_threshold=float(max_relative_error),
        )

    faces: list[tuple[int, int, int] | None] = [
        tuple(int(index) for index in face) for face in original_faces.tolist()
    ]
    face_sets: list[set[int]] = [set() for _ in range(vertex_count)]
    for face_id, face in enumerate(faces):
        for index in set(face):
            face_sets[index].add(face_id)

    alive = np.ones(vertex_count, dtype=bool)
    versions = np.zeros(vertex_count, dtype=np.int64)
    quadrics = _build_quadrics(vertices, original_faces)
    scale = _mesh_scale(vertices)

    pinned = {
        int(np.argmin(np.sum((vertices[:, [0, 2]] - corner) ** 2, axis=1)))
        for corner in (
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        )
    }

    heap: list[tuple[float, int, int, int, int]] = []

    def push_edge(u: int, v: int) -> None:
        if u == v:
            return
        candidate = _make_candidate(
            u,
            v,
            vertices,
            alive,
            face_sets,
            faces,
            quadrics,
            versions,
            pinned,
            scale,
        )
        if candidate is not None:
            heapq.heappush(heap, candidate)

    initial_edges: set[tuple[int, int]] = set()
    for face in original_faces.tolist():
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            initial_edges.add((min(u, v), max(u, v)))
    for u, v in sorted(initial_edges):
        push_edge(u, v)

    alive_count = vertex_count
    collapse_count = 0
    maximum_error = 0.0
    while alive_count > target_vertices and heap:
        relative_error, keep, remove, keep_version, remove_version = heapq.heappop(heap)
        if not alive[keep] or not alive[remove]:
            continue
        if versions[keep] != keep_version or versions[remove] != remove_version:
            continue
        candidate = _make_candidate(
            keep,
            remove,
            vertices,
            alive,
            face_sets,
            faces,
            quadrics,
            versions,
            pinned,
            scale,
        )
        if candidate is None:
            continue
        relative_error, keep, remove, _, _ = candidate
        if relative_error > max_relative_error + 1e-15:
            break
        if not _collapse_edge(
            keep,
            remove,
            alive,
            face_sets,
            faces,
            quadrics,
            versions,
        ):
            continue
        alive_count -= 1
        collapse_count += 1
        maximum_error = max(maximum_error, relative_error)
        neighbors = _vertices_in_faces(face_sets[keep], faces) - {keep}
        for neighbor in sorted(neighbors):
            push_edge(keep, neighbor)

    if collapse_count == 0:
        raise BakeError(
            E_GEOMETRY_INVALID,
            f"{label} could not perform any edge collapse within the error cap",
            {"max_relative_error": float(max_relative_error)},
        )

    old_indices = np.flatnonzero(alive)
    remap = np.full(vertex_count, -1, dtype=np.int64)
    remap[old_indices] = np.arange(len(old_indices), dtype=np.int64)
    compact_faces = [
        tuple(int(remap[index]) for index in face)
        for face in faces
        if face is not None
    ]
    result = TerrainMesh(
        vertices=np.ascontiguousarray(vertices[old_indices]),
        faces=np.ascontiguousarray(np.asarray(compact_faces, dtype=np.int32)),
        weights=np.ascontiguousarray(weights[old_indices]),
    )
    topology_summary(result, level=None)
    return SimplificationResult(
        mesh=result,
        target_vertices=target_vertices,
        reached_target=alive_count <= target_vertices,
        collapse_count=collapse_count,
        relative_error=float(maximum_error),
        error_threshold=float(max_relative_error),
    )


def lod_target_ratios(level_count: int) -> list[float]:
    if level_count <= 1:
        return []
    if level_count == 2:
        return [0.20]
    return [
        0.55 - 0.40 * (index / (level_count - 2))
        for index in range(level_count - 1)
    ]


def lod_error_thresholds(level_count: int, maximum_error: float) -> list[float]:
    if level_count <= 1:
        return []
    return [
        float(maximum_error)
        * (0.35 + 0.65 * (index / (level_count - 2)))
        for index in range(level_count - 1)
    ]


def write_obj(mesh: TerrainMesh, path: Path, comment: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# {comment}\n")
        handle.write(f"# vertices={mesh.vertex_count} faces={mesh.face_count}\n")
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in mesh.faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")
