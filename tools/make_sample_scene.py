#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(lengths, 1.0e-12)


def make_sphere(stacks: int = 18, slices: int = 36, radius: float = 1.0) -> Dict[str, np.ndarray]:
    positions: List[np.ndarray] = [np.array([0.0, radius, 0.0])]
    normals: List[np.ndarray] = [np.array([0.0, 1.0, 0.0])]
    tangents: List[np.ndarray] = [np.array([1.0, 0.0, 0.0])]
    uvs: List[Tuple[float, float]] = [(0.5, 1.0)]
    ring_indices: List[List[int]] = []
    for stack in range(1, stacks):
        theta = np.pi * stack / stacks
        y = np.cos(theta)
        ring_radius = np.sin(theta)
        ring: List[int] = []
        for slice_index in range(slices + 1):
            phi = 2.0 * np.pi * slice_index / slices
            position = np.array(
                [ring_radius * np.cos(phi), y, ring_radius * np.sin(phi)]
            ) * radius
            normal = position / radius
            tangent = np.array([-np.sin(phi), 0.0, np.cos(phi)])
            ring.append(len(positions))
            positions.append(position)
            normals.append(normal)
            tangents.append(tangent)
            uvs.append((slice_index / slices, 1.0 - stack / stacks))
        ring_indices.append(ring)
    bottom_index = len(positions)
    positions.append(np.array([0.0, -radius, 0.0]))
    normals.append(np.array([0.0, -1.0, 0.0]))
    tangents.append(np.array([1.0, 0.0, 0.0]))
    uvs.append((0.5, 0.0))

    triangles: List[Tuple[int, int, int]] = []
    first_ring = ring_indices[0]
    for slice_index in range(slices):
        triangles.append((0, first_ring[slice_index], first_ring[slice_index + 1]))
    for row in range(len(ring_indices) - 1):
        upper = ring_indices[row]
        lower = ring_indices[row + 1]
        for slice_index in range(slices):
            v00 = upper[slice_index]
            v01 = upper[slice_index + 1]
            v10 = lower[slice_index]
            v11 = lower[slice_index + 1]
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))
    last_ring = ring_indices[-1]
    for slice_index in range(slices):
        triangles.append(
            (bottom_index, last_ring[slice_index + 1], last_ring[slice_index])
        )

    position_array = np.asarray(positions, dtype=np.float64)
    normal_array = _normalize_rows(np.asarray(normals, dtype=np.float64))
    tangent_array = _normalize_rows(np.asarray(tangents, dtype=np.float64))
    index_array = np.asarray(triangles, dtype=np.int64)
    for triangle_index, triangle in enumerate(index_array):
        p0, p1, p2 = position_array[triangle]
        geometric_normal = np.cross(p1 - p0, p2 - p0)
        average_normal = normal_array[list(triangle)].mean(axis=0)
        if np.dot(geometric_normal, average_normal) < 0.0:
            index_array[triangle_index] = triangle[[0, 2, 1]]

    face_centers = position_array[index_array].mean(axis=1)
    material_indices = np.where(face_centers[:, 1] > 0.15, 0, 2).astype(np.int64)
    bitangents = _normalize_rows(np.cross(normal_array, tangent_array))
    return {
        "positions": position_array,
        "normals": normal_array,
        "tangents": tangent_array,
        "bitangents": bitangents,
        "uvs": np.asarray(uvs, dtype=np.float64),
        "indices": index_array,
        "material_indices": material_indices,
    }


def make_box(half_extent: float = 0.72) -> Dict[str, np.ndarray]:
    faces = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
    ]
    positions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    bitangents: List[np.ndarray] = []
    uvs: List[Tuple[float, float]] = []
    indices: List[Tuple[int, int, int]] = []
    for normal, tangent in faces:
        bitangent = np.cross(normal, tangent)
        base = len(positions)
        for u, v in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            positions.append(
                (tangent * u + bitangent * v) * half_extent
            )
            normals.append(normal)
            tangents.append(tangent)
            bitangents.append(bitangent)
            uvs.append(((u + 1.0) * 0.5, (v + 1.0) * 0.5))
        indices.append((base, base + 1, base + 2))
        indices.append((base, base + 2, base + 3))
    material_indices = np.asarray(
        [0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0], dtype=np.int64
    )
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "tangents": np.asarray(tangents, dtype=np.float64),
        "bitangents": np.asarray(bitangents, dtype=np.float64),
        "uvs": np.asarray(uvs, dtype=np.float64),
        "indices": np.asarray(indices, dtype=np.int64),
        "material_indices": material_indices,
    }


def make_ground(
    nx: int = 14, nz: int = 12, x_extent: float = 10.0, z_extent: float = 8.0
) -> Dict[str, np.ndarray]:
    positions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    bitangents: List[np.ndarray] = []
    uvs: List[Tuple[float, float]] = []
    for iz in range(nz + 1):
        for ix in range(nx + 1):
            x = -x_extent + 2.0 * x_extent * ix / nx
            z = -z_extent + 2.0 * z_extent * iz / nz
            positions.append(np.array([x, 0.0, z]))
            normals.append(np.array([0.0, 1.0, 0.0]))
            tangents.append(np.array([1.0, 0.0, 0.0]))
            bitangents.append(np.array([0.0, 0.0, -1.0]))
            uvs.append((ix / nx, iz / nz))
    indices: List[Tuple[int, int, int]] = []
    for iz in range(nz):
        for ix in range(nx):
            v00 = iz * (nx + 1) + ix
            v10 = v00 + 1
            v01 = v00 + nx + 1
            v11 = v01 + 1
            indices.append((v00, v01, v11))
            indices.append((v00, v11, v10))
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "tangents": np.asarray(tangents, dtype=np.float64),
        "bitangents": np.asarray(bitangents, dtype=np.float64),
        "uvs": np.asarray(uvs, dtype=np.float64),
        "indices": np.asarray(indices, dtype=np.int64),
        "material_indices": np.ones(len(indices), dtype=np.int64),
    }


def make_wall(
    nx: int = 8, ny: int = 4, width: float = 14.0, height: float = 5.0
) -> Dict[str, np.ndarray]:
    positions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    bitangents: List[np.ndarray] = []
    uvs: List[Tuple[float, float]] = []
    z = -4.8
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = -width * 0.5 + width * ix / nx
            y = height * iy / ny
            positions.append(np.array([x, y, z]))
            normals.append(np.array([0.0, 0.0, 1.0]))
            tangents.append(np.array([1.0, 0.0, 0.0]))
            bitangents.append(np.array([0.0, 1.0, 0.0]))
            uvs.append((ix / nx, iy / ny))
    indices: List[Tuple[int, int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            v00 = iy * (nx + 1) + ix
            v10 = v00 + 1
            v01 = v00 + nx + 1
            v11 = v01 + 1
            indices.append((v00, v10, v11))
            indices.append((v00, v11, v01))
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "tangents": np.asarray(tangents, dtype=np.float64),
        "bitangents": np.asarray(bitangents, dtype=np.float64),
        "uvs": np.asarray(uvs, dtype=np.float64),
        "indices": np.asarray(indices, dtype=np.int64),
        "material_indices": np.ones(len(indices), dtype=np.int64),
    }


def _save_mesh(path: Path, mesh: Dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **mesh)
    return path.name


def build_sample(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mesh_directory = output / "meshes"
    _save_mesh(mesh_directory / "ground.npz", make_ground())
    _save_mesh(mesh_directory / "wall.npz", make_wall())
    _save_mesh(mesh_directory / "sphere.npz", make_sphere())
    _save_mesh(mesh_directory / "box.npz", make_box())

    materials = {
        "materials": [
            {
                "id": "brushed_steel",
                "base_color": [0.27, 0.32, 0.37],
                "specular_color": [0.95, 0.97, 1.0],
                "metallic": 0.88,
                "roughness": 0.24,
                "tangent_specular_exponent": 34.0,
                "bitangent_specular_exponent": 240.0,
                "anisotropy": 0.78,
                "anisotropy_axis": "tangent",
                "anisotropy_rotation_deg": 32.0,
            },
            {
                "id": "dark_concrete",
                "base_color": [0.10, 0.11, 0.12],
                "specular_color": [0.35, 0.35, 0.35],
                "metallic": 0.0,
                "roughness": 0.82,
                "base_specular_exponent": 16.0,
                "anisotropy": 0.0,
            },
            {
                "id": "brushed_copper",
                "base_color": [0.56, 0.24, 0.10],
                "specular_color": [1.0, 0.72, 0.52],
                "metallic": 0.92,
                "roughness": 0.28,
                "base_specular_exponent": 150.0,
                "anisotropy": 0.72,
                "anisotropy_axis": "bitangent",
                "anisotropy_rotation_deg": -18.0,
            },
        ]
    }
    with (output / "materials.json").open("w", encoding="utf-8") as handle:
        json.dump(materials, handle, indent=2)
        handle.write("\n")

    scene = {
        "schema_version": 1,
        "width": 480,
        "height": 320,
        "materials": "materials.json",
        "camera": {
            "position": [6.2, 4.4, 9.2],
            "target": [0.0, 1.15, -0.2],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 48.0,
            "near": 0.1,
            "far": 50.0,
        },
        "shadows": {
            "resolution": 256,
            "cascade_count": 3,
            "split_lambda": 0.62,
            "shadow_distance": 35.0,
            "pcf_radius": 1,
        },
        "depth": {
            "bias": 0.0015,
            "slope_scale": 3.0,
            "compare_tolerance": 0.006,
        },
        "lights": [
            {
                "id": "sun",
                "type": "directional",
                "direction": [0.30, -0.85, -0.42],
                "color": [1.0, 0.93, 0.80],
                "intensity": 2.8,
            },
            {
                "id": "point_blue",
                "type": "point",
                "position": [-3.0, 2.7, 1.8],
                "range": 8.5,
                "color": [0.18, 0.55, 1.0],
                "intensity": 22.0,
            },
            {
                "id": "spot_amber",
                "type": "spot",
                "position": [3.4, 4.2, 2.6],
                "direction": [-0.63, -0.66, -0.41],
                "range": 14.0,
                "inner_angle_deg": 15.0,
                "outer_angle_deg": 24.0,
                "color": [1.0, 0.28, 0.12],
                "intensity": 72.0,
            },
        ],
        "meshes": [
            {
                "name": "ground",
                "path": "meshes/ground.npz",
                "transform": {"translate": [0.0, 0.0, 0.0]},
                "cast_shadow": True,
            },
            {
                "name": "wall",
                "path": "meshes/wall.npz",
                "transform": {"translate": [0.0, 0.0, 0.0]},
                "cast_shadow": True,
            },
            {
                "name": "steel_sphere",
                "path": "meshes/sphere.npz",
                "transform": {
                    "translate": [0.0, 1.18, 0.15],
                    "scale": [1.13, 1.13, 1.13],
                },
                "cast_shadow": True,
            },
            {
                "name": "copper_box",
                "path": "meshes/box.npz",
                "transform": {
                    "translate": [-2.3, 0.76, 0.9],
                    "rotate_xyz_deg": [0.0, 27.0, 0.0],
                },
                "cast_shadow": True,
            },
            {
                "name": "small_sphere",
                "path": "meshes/sphere.npz",
                "transform": {
                    "translate": [2.05, 0.64, 0.55],
                    "scale": [0.64, 0.64, 0.64],
                },
                "cast_shadow": True,
            },
        ],
    }
    with (output / "sample_scene.json").open("w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic sample assets.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "sample"),
        help="sample directory",
    )
    args = parser.parse_args()
    build_sample(Path(args.output).resolve())
    print("generated deterministic sample in %s" % Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
