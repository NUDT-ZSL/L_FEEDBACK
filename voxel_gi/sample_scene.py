"""Deterministic local sample-scene generator (no external assets)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _quad(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> List[List[List[float]]]:
    p0 = [float(value) for value in a]
    p1 = [float(value) for value in b]
    p2 = [float(value) for value in c]
    p3 = [float(value) for value in d]
    return [[p0, p1, p2], [p0, p2, p3]]


def _room_triangles() -> Dict[str, List[List[List[float]]]]:
    x0, x1 = 0.0, 8.0
    y0, y1 = 0.0, 5.0
    z0, z1 = 0.0, 6.0
    floor = _quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1])
    ceiling = _quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0])
    back = _quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0])
    left = _quad([x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0])
    right = _quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1])
    return {
        "floor": floor,
        "ceiling": ceiling,
        "back_wall": back,
        "left_wall": left,
        "right_wall": right,
    }


def _box_triangles(
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> List[List[List[float]]]:
    x0, y0, z0 = [float(value) for value in minimum]
    x1, y1, z1 = [float(value) for value in maximum]
    vertices = {
        "a": [x0, y0, z0],
        "b": [x1, y0, z0],
        "c": [x1, y0, z1],
        "d": [x0, y0, z1],
        "e": [x0, y1, z0],
        "f": [x1, y1, z0],
        "g": [x1, y1, z1],
        "h": [x0, y1, z1],
    }
    result: List[List[List[float]]] = []
    for face in (
        ("a", "b", "c", "d"),
        ("e", "h", "g", "f"),
        ("a", "e", "f", "b"),
        ("d", "c", "g", "h"),
        ("a", "d", "h", "e"),
        ("b", "f", "g", "c"),
    ):
        result.extend(
            _quad(
                vertices[face[0]],
                vertices[face[1]],
                vertices[face[2]],
                vertices[face[3]],
            )
        )
    return result


def _icosphere(
    center: Sequence[float],
    radius: float,
    subdivisions: int = 2,
) -> List[List[List[float]]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_vertices = [
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    vertices = [np.asarray(vertex, dtype=np.float64) for vertex in base_vertices]
    vertices = [vertex / np.linalg.norm(vertex) for vertex in vertices]
    for _ in range(int(subdivisions)):
        cache: Dict[Tuple[int, int], int] = {}
        new_faces: List[List[int]] = []

        def midpoint(left: int, right: int) -> int:
            key = (min(left, right), max(left, right))
            if key in cache:
                return cache[key]
            value = vertices[left] + vertices[right]
            value = value / np.linalg.norm(value)
            vertices.append(value)
            cache[key] = len(vertices) - 1
            return cache[key]

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend(
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            )
        faces = new_faces
    origin = np.asarray(center, dtype=np.float64)
    return [
        [
            [round(float(value), 12) for value in origin + radius * vertices[index]]
            for index in face
        ]
        for face in faces
    ]


def generate_sample_scene(seed: int = 20261115) -> Dict[str, object]:
    room = _room_triangles()
    occluder = _box_triangles([2.35, 0.0, 2.15], [4.35, 1.55, 4.15])
    sphere = _icosphere([6.05, 1.05, 4.35], 0.62, subdivisions=2)
    return {
        "schema_version": 1,
        "scene_id": "deterministic_room_v1",
        "seed": int(seed),
        "materials": [
            {"id": "floor", "albedo": [0.78, 0.66, 0.54]},
            {"id": "wall", "albedo": [0.56, 0.60, 0.68]},
            {"id": "occluder", "albedo": [0.82, 0.32, 0.24]},
            {"id": "sphere", "albedo": [0.25, 0.42, 0.82]},
        ],
        "meshes": [
            {
                "name": "floor",
                "material": "floor",
                "occluder": True,
                "triangles": room["floor"],
            },
            {
                "name": "ceiling",
                "material": "wall",
                "occluder": True,
                "triangles": room["ceiling"],
            },
            {
                "name": "back_wall",
                "material": "wall",
                "occluder": True,
                "triangles": room["back_wall"],
            },
            {
                "name": "left_wall",
                "material": "wall",
                "occluder": True,
                "triangles": room["left_wall"],
            },
            {
                "name": "right_wall",
                "material": "wall",
                "occluder": True,
                "triangles": room["right_wall"],
            },
            {
                "name": "central_occluder",
                "material": "occluder",
                "occluder": True,
                "triangles": occluder,
            },
            {
                "name": "blue_sphere",
                "material": "sphere",
                "occluder": True,
                "triangles": sphere,
            },
        ],
        "lights": [
            {
                "id": "warm_area",
                "type": "area",
                "position": [2.35, 4.25, 2.45],
                "normal": [0.0, -1.0, 0.0],
                "radius": 0.55,
                "radiant_power": 6.0,
                "color": [1.0, 0.82, 0.61],
                "samples": 3,
            },
            {
                "id": "cool_area",
                "type": "area",
                "position": [6.20, 4.25, 4.30],
                "normal": [0.0, -1.0, 0.0],
                "radius": 0.48,
                "radiant_power": 4.0,
                "color": [0.56, 0.73, 1.0],
                "samples": 3,
            },
        ],
        "probe_layout": {
            "bounds_min": [1.5, 2.0, 1.5],
            "bounds_max": [6.5, 4.0, 4.5],
            "spacing": 1.6,
            "up": [0.0, 1.0, 0.0],
        },
        "bake": {
            "voxel_resolution": 36,
            "padding_voxels": 1,
            "ray_count": 256,
            "threads": 4,
            "seed": int(seed),
            "ambient_radiance": [0.025, 0.030, 0.040],
        },
    }


def write_sample_scene(path: Path, seed: int = 20261115) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(generate_sample_scene(seed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic local sample scene.")
    parser.add_argument("output", type=Path, help="Destination scene JSON")
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args(argv)
    write_sample_scene(args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
