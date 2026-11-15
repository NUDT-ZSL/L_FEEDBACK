"""Deterministic procedural scene construction.

No external assets are used.  The generated scene contains a parent/child
hierarchy (an orbiting cube carrying a child sphere), an opaque ground plane,
alpha-blended and additive quads, a directional light and a point light.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Sequence, Tuple


def _cube(side: float = 1.0) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    h = side / 2.0
    corners = [
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
    ]
    faces = [
        (0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7),
        (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0),
    ]
    positions: List[List[float]] = []
    normals: List[List[float]] = []
    indices: List[int] = []
    for face in faces:
        p0, p1, p2 = corners[face[0]], corners[face[1]], corners[face[2]]
        ux = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        vx = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        nx = (ux[1] * vx[2] - ux[2] * vx[1],
              ux[2] * vx[0] - ux[0] * vx[2],
              ux[0] * vx[1] - ux[1] * vx[0])
        length = math.sqrt(sum(c * c for c in nx)) or 1.0
        normal = [c / length for c in nx]
        base = len(positions)
        for corner_index in face:
            positions.append(list(corners[corner_index]))
            normals.append(normal)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return positions, normals, indices


def _sphere(radius: float = 0.5, segments: int = 12, rings: int = 8):
    positions: List[List[float]] = []
    normals: List[List[float]] = []
    indices: List[int] = []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        for s in range(segments + 1):
            phi = 2.0 * math.pi * s / segments
            nx = math.sin(theta) * math.cos(phi)
            ny = math.cos(theta)
            nz = math.sin(theta) * math.sin(phi)
            normals.append([nx, ny, nz])
            positions.append([radius * nx, radius * ny, radius * nz])
    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a = r * row + s
            b = a + row
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    return positions, normals, indices


def _plane(width: float, height: float) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    hw, hh = width / 2.0, height / 2.0
    positions = [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]]
    normals = [[0.0, 0.0, 1.0]] * 4
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, indices


def build_sample_scene(width: int = 320, height: int = 240) -> Dict[str, Any]:
    cube_p, cube_n, cube_i = _cube(1.0)
    sph_p, sph_n, sph_i = _sphere()
    plane_p, plane_n, plane_i = _plane(3.0, 2.0)
    return {
        "width": width,
        "height": height,
        "background": [0.02, 0.03, 0.06, 1.0],
        "ambient": [0.08, 0.08, 0.10],
        "camera": {
            "eye": [3.2, 2.6, 4.5],
            "target": [0.0, 0.2, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y": math.radians(55.0),
            "near": 0.1,
            "far": 50.0,
        },
        "lights": [
            {"type": "directional", "direction": [-0.4, -1.0, -0.3],
             "color": [1.0, 0.95, 0.85], "intensity": 1.0},
            {"type": "point", "position": [-2.0, 2.2, 2.0],
             "color": [0.4, 0.7, 1.0], "intensity": 8.0, "range": 12.0},
        ],
        "materials": {
            "red_metal": {"base_color": [0.85, 0.18, 0.16, 1.0],
                          "roughness": 0.35, "metallic": 0.6, "blend": "opaque"},
            "blue_matte": {"base_color": [0.18, 0.45, 0.85, 1.0],
                           "roughness": 0.7, "metallic": 0.0, "blend": "opaque"},
            "floor": {"base_color": [0.35, 0.38, 0.42, 1.0],
                      "roughness": 0.9, "metallic": 0.0, "blend": "opaque"},
            "glass": {"base_color": [0.3, 0.9, 0.9, 0.45],
                      "roughness": 0.15, "metallic": 0.0, "blend": "alpha"},
            "glow": {"base_color": [1.0, 0.7, 0.2, 0.6],
                     "roughness": 0.4, "metallic": 0.0, "blend": "additive"},
        },
        "meshes": {
            "cube": {"positions": cube_p, "normals": cube_n, "indices": cube_i},
            "sphere": {"positions": sph_p, "normals": sph_n, "indices": sph_i},
            "quad": {"positions": plane_p, "normals": plane_n, "indices": plane_i},
        },
        "nodes": [
            {
                "name": "floor",
                "mesh": "cube", "material": "floor",
                "translate": [0.0, -1.2, 0.0],
                "rotate": [0.0, 0.0, 0.0],
                "scale": [7.0, 0.1, 7.0],
            },
            {
                "name": "orbit_root",
                "translate": [0.0, 0.3, 0.0],
                "orbit": 0.8,
                "children": [
                    {
                        "name": "parent_cube",
                        "mesh": "cube", "material": "red_metal",
                        "translate": [1.6, 0.0, 0.0],
                        "rotate": [0.3, 0.4, 0.0],
                        "scale": [0.9, 0.9, 0.9],
                        "spin": [0.0, 1.4, 0.0],
                        "children": [
                            {
                                "name": "child_sphere",
                                "mesh": "sphere", "material": "blue_matte",
                                "translate": [0.0, 0.95, 0.0],
                                "scale": [0.7, 0.7, 0.7],
                                "spin": [0.0, 0.0, 1.8],
                            }
                        ],
                    }
                ],
            },
            {
                "name": "alpha_quad",
                "mesh": "quad", "material": "glass",
                "translate": [-1.1, 0.4, 1.4],
                "rotate": [0.0, 0.6, 0.0],
            },
            {
                "name": "additive_quad",
                "mesh": "quad", "material": "glow",
                "translate": [1.2, 0.7, -1.2],
                "rotate": [0.0, -0.5, 0.0],
            },
        ],
    }


def write_sample_scene(path: str, width: int = 320, height: int = 240) -> str:
    scene = build_sample_scene(width, height)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2)
    return path
