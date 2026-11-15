"""Deterministic built-in cube and UV-sphere geometry generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshData:
    name: str
    vertices: np.ndarray
    indices: np.ndarray


def generate_cube() -> MeshData:
    """Return a unit cube with 24 vertices so each face has a flat normal."""
    face_specs = (
        ((-1, -1, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((1, -1, -1), (0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ((1, -1, 1), (1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((-1, -1, -1), (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((-1, 1, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((-1, -1, -1), (0, -1, 0), (1, 0, 0), (0, 0, 1)),
    )

    vertices: list[float] = []
    indices: list[int] = []
    for origin, normal, tangent, bitangent in face_specs:
        base = len(vertices) // 8
        for y, x in ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)):
            position = np.asarray(origin, dtype=np.float64)
            position += float(x) * np.asarray(tangent, dtype=np.float64)
            position += float(y) * np.asarray(bitangent, dtype=np.float64)
            vertices.extend(float(value) for value in position)
            vertices.extend(float(value) for value in normal)
            vertices.extend((float(x), float(y)))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    return MeshData(
        name="cube",
        vertices=np.asarray(vertices, dtype=np.float32).reshape(-1, 8),
        indices=np.asarray(indices, dtype=np.uint32),
    )


def generate_uv_sphere(segments: int = 32, rings: int = 20) -> MeshData:
    """Return a UV sphere with outward normals and stable triangle winding."""
    if segments < 3 or rings < 2:
        raise ValueError("sphere requires segments >= 3 and rings >= 2")

    vertices: list[float] = []
    for ring in range(rings + 1):
        theta = np.pi * ring / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for segment in range(segments + 1):
            phi = 2.0 * np.pi * segment / segments
            x = sin_theta * np.cos(phi)
            y = cos_theta
            z = sin_theta * np.sin(phi)
            vertices.extend(
                (
                    float(x),
                    float(y),
                    float(z),
                    float(x),
                    float(y),
                    float(z),
                    float(segment / segments),
                    float(1.0 - ring / rings),
                )
            )

    indices: list[int] = []
    row = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            a = ring * row + segment
            b = a + row
            if ring != 0:
                indices.extend((a, b, a + 1))
            if ring != rings - 1:
                indices.extend((a + 1, b, b + 1))

    return MeshData(
        name="sphere",
        vertices=np.asarray(vertices, dtype=np.float32).reshape(-1, 8),
        indices=np.asarray(indices, dtype=np.uint32),
    )


def generate_builtin_meshes() -> dict[str, MeshData]:
    return {
        "cube": generate_cube(),
        "sphere": generate_uv_sphere(),
    }

