from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _box_mesh(
    name: str,
    center: Sequence[float],
    size: Sequence[float],
    material: int,
) -> Dict[str, Any]:
    center_array = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) * 0.5
    minimum = center_array - half
    maximum = center_array + half
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    faces = [
        (np.array([-1.0, 0.0, 0.0]), [(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)]),
        (np.array([1.0, 0.0, 0.0]), [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)]),
        (np.array([0.0, -1.0, 0.0]), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        (np.array([0.0, 1.0, 0.0]), [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)]),
        (np.array([0.0, 0.0, -1.0]), [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
        (np.array([0.0, 0.0, 1.0]), [(x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)]),
    ]
    vertices: List[Dict[str, Any]] = []
    triangles: List[Dict[str, Any]] = []
    for normal, corners in faces:
        base = len(vertices)
        for uv, corner in zip(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), corners):
            vertices.append(
                {
                    "position": [float(value) for value in corner],
                    "normal": [float(value) for value in normal],
                    "uv": [float(uv[0]), float(uv[1])],
                }
            )
        first = np.asarray(corners[0])
        second = np.asarray(corners[1])
        third = np.asarray(corners[2])
        winding = np.cross(second - first, third - first)
        if float(np.dot(winding, normal)) < 0.0:
            triangles.append({"indices": [base, base + 2, base + 1], "material": material})
            triangles.append({"indices": [base, base + 3, base + 2], "material": material})
        else:
            triangles.append({"indices": [base, base + 1, base + 2], "material": material})
            triangles.append({"indices": [base, base + 2, base + 3], "material": material})
    return {"name": name, "vertices": vertices, "triangles": triangles}


def _uv_sphere_mesh(
    name: str,
    center: Sequence[float],
    radius: float,
    material: int,
    longitude_segments: int = 32,
    latitude_segments: int = 18,
) -> Dict[str, Any]:
    cx, cy, cz = [float(value) for value in center]
    vertices: List[Dict[str, Any]] = []
    for latitude in range(latitude_segments + 1):
        theta = math.pi * latitude / latitude_segments
        ring_radius = radius * math.sin(theta)
        y = cy + radius * math.cos(theta)
        for longitude in range(longitude_segments):
            phi = 2.0 * math.pi * longitude / longitude_segments
            vertices.append(
                {
                    "position": [
                        cx + ring_radius * math.sin(phi),
                        y,
                        cz + ring_radius * math.cos(phi),
                    ],
                    "uv": [
                        longitude / longitude_segments,
                        1.0 - latitude / latitude_segments,
                    ],
                }
            )
    triangles: List[Dict[str, Any]] = []
    for latitude in range(latitude_segments):
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            a = latitude * longitude_segments + longitude
            b = (latitude + 1) * longitude_segments + longitude
            c = (latitude + 1) * longitude_segments + next_longitude
            d = latitude * longitude_segments + next_longitude
            triangles.append({"indices": [a, b, d], "material": material})
            triangles.append({"indices": [b, c, d], "material": material})
    return {"name": name, "vertices": vertices, "triangles": triangles}


def build_indoor_scene(width: int = 320, height: int = 240) -> Dict[str, Any]:
    floor = {
        "name": "oak_floor",
        "vertices": [
            {"position": [-5.0, 0.0, 5.0], "uv": [0.0, 0.0]},
            {"position": [5.0, 0.0, 5.0], "uv": [2.0, 0.0]},
            {"position": [5.0, 0.0, -3.0], "uv": [2.0, 2.0]},
            {"position": [-5.0, 0.0, -3.0], "uv": [0.0, 2.0]},
        ],
        "triangles": [
            {"indices": [0, 1, 2], "material": 0},
            {"indices": [0, 2, 3], "material": 0},
        ],
    }
    back_wall = {
        "name": "plaster_back_wall",
        "vertices": [
            {"position": [-5.0, 0.0, -3.0], "uv": [0.0, 0.0]},
            {"position": [5.0, 0.0, -3.0], "uv": [3.0, 0.0]},
            {"position": [5.0, 5.0, -3.0], "uv": [3.0, 2.0]},
            {"position": [-5.0, 5.0, -3.0], "uv": [0.0, 2.0]},
        ],
        "triangles": [
            {"indices": [0, 1, 2], "material": 1},
            {"indices": [0, 2, 3], "material": 1},
        ],
    }
    left_wall = {
        "name": "plaster_left_wall",
        "vertices": [
            {"position": [-5.0, 0.0, 5.0], "uv": [0.0, 0.0]},
            {"position": [-5.0, 0.0, -3.0], "uv": [3.0, 0.0]},
            {"position": [-5.0, 5.0, -3.0], "uv": [3.0, 2.0]},
            {"position": [-5.0, 5.0, 5.0], "uv": [0.0, 2.0]},
        ],
        "triangles": [
            {"indices": [0, 1, 2], "material": 1},
            {"indices": [0, 2, 3], "material": 1},
        ],
    }

    return {
        "version": 1,
        "color_space": "srgb",
        "image": {"width": int(width), "height": int(height)},
        "camera": {
            "position": [0.0, 2.15, 7.2],
            "target": [0.0, 1.05, -0.4],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 42.0,
            "near": 0.05,
            "far": 100.0,
        },
        "background": [0.015, 0.018, 0.022],
        "ambient": [0.025, 0.03, 0.04],
        "materials": [
            {
                "id": 0,
                "name": "oak_floor",
                "base_color": [0.42, 0.25, 0.11],
                "roughness": 0.58,
                "metallic": 0.0,
            },
            {
                "id": 1,
                "name": "warm_plaster",
                "base_color": [0.72, 0.67, 0.57],
                "roughness": 0.82,
                "metallic": 0.0,
            },
            {
                "id": 2,
                "name": "red_lacquer",
                "base_color": [0.58, 0.055, 0.035],
                "roughness": 0.24,
                "metallic": 0.0,
            },
            {
                "id": 3,
                "name": "brushed_metal",
                "base_color": [0.78, 0.69, 0.47],
                "roughness": 0.23,
                "metallic": 1.0,
            },
            {
                "id": 4,
                "name": "blue_pedestal",
                "base_color": [0.055, 0.18, 0.38],
                "roughness": 0.38,
                "metallic": 0.1,
            },
        ],
        "meshes": [
            floor,
            back_wall,
            left_wall,
            _box_mesh("red_cube", [-1.35, 0.62, 0.55], [1.24, 1.24, 1.24], 2),
            _box_mesh("blue_pedestal", [1.35, 0.72, 0.15], [0.9, 1.44, 0.9], 4),
            _uv_sphere_mesh("metal_sphere", [1.35, 2.08, 0.15], 0.62, 3),
        ],
        "lights": [
            {
                "name": "warm_point",
                "type": "point",
                "position": [-2.75, 3.4, 2.2],
                "color": [1.0, 0.69, 0.39],
                "intensity": 92.0,
                "attenuation": {
                    "constant": 1.0,
                    "linear": 0.12,
                    "quadratic": 0.42,
                    "range": 13.0,
                },
                "shadow_bias": 0.0012,
            },
            {
                "name": "cool_spot",
                "type": "spot",
                "position": [3.1, 4.15, 3.65],
                "target": [0.2, 0.9, 0.0],
                "color": [0.45, 0.72, 1.0],
                "intensity": 145.0,
                "inner_angle_deg": 18.0,
                "outer_angle_deg": 33.0,
                "attenuation": {
                    "constant": 1.0,
                    "linear": 0.1,
                    "quadratic": 0.35,
                    "range": 15.0,
                },
                "shadow_bias": 0.0011,
            },
        ],
    }


def write_indoor_scene(path: str, width: int = 320, height: int = 240) -> Dict[str, Any]:
    scene = build_indoor_scene(width, height)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(scene, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return scene
