"""Deterministically generate the bundled JSON sample scene.

Run from the repository root:
    python examples/generate_sample.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def make_cube(center, half_size):
    center = np.asarray(center, dtype=float)
    half_size = np.asarray(half_size, dtype=float)
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[list[int]] = []
    for normal in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        normal_array = np.asarray(normal, dtype=float)
        up = (
            np.asarray([0.0, 1.0, 0.0])
            if abs(normal_array[1]) < 0.9
            else np.asarray([0.0, 0.0, 1.0])
        )
        u = np.cross(up, normal_array)
        u /= np.linalg.norm(u)
        v = np.cross(normal_array, u)
        face_center = center + normal_array * half_size
        corners = (
            -u * half_size[0] - v * half_size[1],
            u * half_size[0] - v * half_size[1],
            u * half_size[0] + v * half_size[1],
            -u * half_size[0] + v * half_size[1],
        )
        current = len(vertices)
        for corner in corners:
            vertices.append([float(value) for value in face_center + corner])
            normals.append([float(value) for value in normal_array])
        indices.extend([[current, current + 1, current + 2], [current, current + 2, current + 3]])
    return vertices, normals, indices


def make_uv_sphere(center, radius, latitude_segments=16, longitude_segments=24):
    center = np.asarray(center, dtype=float)
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    for latitude_index in range(latitude_segments + 1):
        latitude = -0.5 * math.pi + math.pi * latitude_index / latitude_segments
        for longitude_index in range(longitude_segments):
            longitude = 2.0 * math.pi * longitude_index / longitude_segments
            normal = np.asarray(
                [
                    math.cos(latitude) * math.cos(longitude),
                    math.sin(latitude),
                    math.cos(latitude) * math.sin(longitude),
                ]
            )
            vertices.append(
                [float(value) for value in center + radius * normal]
            )
            normals.append([float(value) for value in normal])

    indices: list[list[int]] = []
    for latitude_index in range(latitude_segments):
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            a = latitude_index * longitude_segments + longitude_index
            b = (latitude_index + 1) * longitude_segments + longitude_index
            c = (latitude_index + 1) * longitude_segments + next_longitude
            d = latitude_index * longitude_segments + next_longitude
            indices.append([a, b, c])
            indices.append([a, c, d])
    return vertices, normals, indices


def build_scene():
    materials = [
        {
            "name": "polished_floor",
            "layers": [
                {
                    "base_color": [0.18, 0.21, 0.26],
                    "metallic": 0.0,
                    "roughness": 0.62,
                    "clearcoat": 0.0,
                    "normal_strength": 0.7,
                    "normal": [0.22, 0.08, 1.0],
                    "opacity": 1.0,
                },
                {
                    "base_color": [0.08, 0.10, 0.13],
                    "metallic": 0.0,
                    "roughness": 0.10,
                    "clearcoat": 1.0,
                    "normal_strength": 0.25,
                    "normal": [0.02, -0.03, 1.0],
                    "opacity": 0.24,
                },
            ],
        },
        {
            "name": "copper_glaze",
            "layers": [
                {
                    "base_color": [0.58, 0.27, 0.10],
                    "metallic": 0.88,
                    "roughness": 0.34,
                    "clearcoat": 0.20,
                    "normal_strength": 0.35,
                    "normal": [0.12, 0.05, 1.0],
                    "opacity": 1.0,
                },
                {
                    "base_color": [0.65, 0.42, 0.28],
                    "metallic": 0.0,
                    "roughness": 0.07,
                    "clearcoat": 1.0,
                    "normal_strength": 0.15,
                    "normal": [0.0, 0.0, 1.0],
                    "opacity": 0.34,
                },
            ],
        },
        {
            "name": "studio_wall",
            "layers": [
                {
                    "base_color": [0.22, 0.25, 0.31],
                    "metallic": 0.0,
                    "roughness": 0.78,
                    "clearcoat": 0.0,
                    "normal_strength": 0.45,
                    "normal": [0.08, 0.04, 1.0],
                    "opacity": 1.0,
                },
                {
                    "base_color": [0.10, 0.13, 0.20],
                    "metallic": 0.0,
                    "roughness": 0.16,
                    "clearcoat": 0.75,
                    "normal_strength": 0.1,
                    "normal": [0.0, 0.0, 1.0],
                    "opacity": 0.12,
                },
            ],
        },
        {
            "name": "shadow_caster",
            "layers": [
                {
                    "base_color": [0.34, 0.07, 0.045],
                    "metallic": 0.0,
                    "roughness": 0.55,
                    "clearcoat": 0.1,
                    "normal_strength": 0.2,
                    "normal": [0.0, 0.0, 1.0],
                    "opacity": 1.0,
                }
            ],
        },
        {
            "name": "pedestal",
            "layers": [
                {
                    "base_color": [0.12, 0.14, 0.17],
                    "metallic": 0.55,
                    "roughness": 0.48,
                    "clearcoat": 0.0,
                    "normal_strength": 0.2,
                    "normal": [0.05, 0.03, 1.0],
                    "opacity": 1.0,
                },
                {
                    "base_color": [0.12, 0.14, 0.17],
                    "metallic": 0.0,
                    "roughness": 0.12,
                    "clearcoat": 1.0,
                    "normal_strength": 0.0,
                    "normal": [0.0, 0.0, 1.0],
                    "opacity": 0.2,
                },
            ],
        },
    ]

    meshes = []

    floor_vertices = [
        [-7.0, 0.0, -6.0],
        [-7.0, 0.0, 5.0],
        [7.0, 0.0, 5.0],
        [7.0, 0.0, -6.0],
    ]
    meshes.append(
        {
            "name": "floor",
            "vertices": floor_vertices,
            "normals": [[0.0, 1.0, 0.0]] * 4,
            "triangles": [[0, 1, 2], [0, 2, 3]],
            "material": 0,
        }
    )

    meshes.append(
        {
            "name": "back_wall",
            "vertices": [
                [-7.0, 0.0, -6.0],
                [7.0, 0.0, -6.0],
                [7.0, 5.0, -6.0],
                [-7.0, 5.0, -6.0],
            ],
            "normals": [[0.0, 0.0, 1.0]] * 4,
            "triangles": [[0, 1, 2], [0, 2, 3]],
            "material": 2,
        }
    )

    meshes.append(
        {
            "name": "left_wall",
            "vertices": [
                [-7.0, 0.0, 5.0],
                [-7.0, 0.0, -6.0],
                [-7.0, 5.0, -6.0],
                [-7.0, 5.0, 5.0],
            ],
            "normals": [[1.0, 0.0, 0.0]] * 4,
            "triangles": [[0, 1, 2], [0, 2, 3]],
            "material": 2,
        }
    )

    pedestal_vertices, pedestal_normals, pedestal_indices = make_cube(
        [0.0, 0.35, 0.0], [1.25, 0.35, 1.0]
    )
    meshes.append(
        {
            "name": "pedestal",
            "vertices": pedestal_vertices,
            "normals": pedestal_normals,
            "triangles": pedestal_indices,
            "material": 4,
        }
    )

    sphere_vertices, sphere_normals, sphere_indices = make_uv_sphere(
        [0.0, 1.65, 0.0], 0.86
    )
    meshes.append(
        {
            "name": "hero_orb",
            "vertices": sphere_vertices,
            "normals": sphere_normals,
            "triangles": sphere_indices,
            "material": 1,
        }
    )

    caster_vertices, caster_normals, caster_indices = make_cube(
        [7.3, 1.4, -0.8], [0.8, 1.4, 0.8]
    )
    meshes.append(
        {
            "name": "off_camera_shadow_caster",
            "vertices": caster_vertices,
            "normals": caster_normals,
            "triangles": caster_indices,
            "material": 3,
        }
    )

    return {
        "scene": {
            "camera": {
                "position": [0.0, 2.25, 7.5],
                "target": [0.0, 1.2, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_y_deg": 44.0,
                "near": 0.15,
                "far": 60.0,
            },
            "ambient": {"color": [0.035, 0.045, 0.065], "intensity": 1.0},
            "background": [0.006, 0.008, 0.012],
            "lights": [
                {
                    "name": "warm_key",
                    "type": "point",
                    "position": [4.2, 4.8, 3.2],
                    "color": [1.0, 0.76, 0.50],
                    "intensity": 72.0,
                    "range": 25.0,
                    "shadow": {
                        "enabled": True,
                        "map_size": 128,
                        "radius": 2.6,
                        "samples": 12,
                        "bias": 0.003,
                        "normal_bias": 0.025,
                    },
                },
                {
                    "name": "cool_fill",
                    "type": "point",
                    "position": [-4.0, 3.2, 2.6],
                    "color": [0.48, 0.68, 1.0],
                    "intensity": 48.0,
                    "range": 24.0,
                    "shadow": {
                        "enabled": True,
                        "map_size": 128,
                        "radius": 3.0,
                        "samples": 12,
                        "bias": 0.003,
                        "normal_bias": 0.025,
                    },
                },
                {
                    "name": "blue_rim",
                    "type": "point",
                    "position": [0.2, 3.8, -4.2],
                    "color": [0.35, 0.58, 1.0],
                    "intensity": 66.0,
                    "range": 24.0,
                    "shadow": {
                        "enabled": True,
                        "map_size": 128,
                        "radius": 2.2,
                        "samples": 12,
                        "bias": 0.003,
                        "normal_bias": 0.025,
                    },
                },
                {
                    "name": "soft_sun",
                    "type": "directional",
                    "direction": [-0.32, -1.0, -0.22],
                    "color": [1.0, 0.93, 0.82],
                    "intensity": 2.1,
                    "shadow": {
                        "enabled": True,
                        "map_size": 256,
                        "radius": 3.2,
                        "samples": 16,
                        "bias": 0.003,
                        "normal_bias": 0.025,
                    },
                },
            ],
            "materials": materials,
            "meshes": meshes,
        },
        "render": {
            "width": 320,
            "height": 240,
            "exposure": 1.0,
            "max_layers": 8,
        },
    }


def main() -> None:
    output = Path(__file__).with_name("studio_scene.json")
    payload = build_scene()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
