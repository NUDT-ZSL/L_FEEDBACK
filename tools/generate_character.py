#!/usr/bin/env python3
"""Generate a deterministic, self-describing character model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from character_preview.math3d import (
    compose,
    rotation_y,
    rotation_z,
    scaling,
    to_column_major,
    translation,
)


def _round_values(values: np.ndarray, digits: int = 7) -> list[float]:
    return [float(round(float(value), digits)) for value in np.asarray(values).reshape(-1)]


def _round_vertices(values: np.ndarray, digits: int = 7) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    return [[float(round(float(component), digits)) for component in row] for row in array]


def box(width: float, height: float, depth: float) -> dict:
    hx, hy, hz = width / 2, height / 2, depth / 2
    faces = [
        ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
    ]
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for normal, corners in faces:
        base = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return {
        "positions": _round_vertices(np.asarray(positions)),
        "normals": _round_vertices(np.asarray(normals)),
        "indices": indices,
    }


def sphere(radius: float, segments: int = 32, rings: int = 20) -> dict:
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[int] = []
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        sin_theta, cos_theta = math.sin(theta), math.cos(theta)
        for segment in range(segments + 1):
            phi = 2.0 * math.pi * segment / segments
            n = np.array(
                [sin_theta * math.cos(phi), cos_theta, sin_theta * math.sin(phi)],
                dtype=np.float64,
            )
            positions.append(list(n * radius))
            normals.append(list(n))
    row = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            a = ring * row + segment
            b = a + row
            indices.extend([a, a + 1, b, a + 1, b + 1, b])
    return {
        "positions": _round_vertices(np.asarray(positions)),
        "normals": _round_vertices(np.asarray(normals)),
        "indices": indices,
    }


def cylinder(radius: float, height: float, segments: int = 24) -> dict:
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[int] = []
    half = height / 2
    for segment in range(segments + 1):
        angle = 2.0 * math.pi * segment / segments
        c, s = math.cos(angle), math.sin(angle)
        for y in (-half, half):
            positions.append([radius * c, y, radius * s])
            normals.append([c, 0.0, s])
    for segment in range(segments):
        a = segment * 2
        indices.extend([a, a + 1, a + 2, a + 1, a + 3, a + 2])
    for y, normal in ((half, (0.0, 1.0, 0.0)), (-half, (0.0, -1.0, 0.0))):
        center = len(positions)
        positions.append([0.0, y, 0.0])
        normals.append(list(normal))
        for segment in range(segments + 1):
            angle = 2.0 * math.pi * segment / segments
            positions.append([radius * math.cos(angle), y, radius * math.sin(angle)])
            normals.append(list(normal))
        for segment in range(segments):
            if normal[1] > 0.0:
                indices.extend([center, center + segment + 2, center + segment + 1])
            else:
                indices.extend([center, center + segment + 1, center + segment + 2])
    return {
        "positions": _round_vertices(np.asarray(positions)),
        "normals": _round_vertices(np.asarray(normals)),
        "indices": indices,
    }


def _node(name: str, mesh: dict, material: str, transform: np.ndarray) -> dict:
    return {
        "name": name,
        "material": material,
        "transform": _round_values(np.asarray(to_column_major(transform)), 8),
        "mesh": mesh,
    }


def build_model() -> dict:
    materials = [
        {
            "name": "suit_blue",
            "diffuse": [0.035, 0.12, 0.34],
            "specular": [0.85, 0.9, 1.0],
            "roughness": 0.32,
            "metallic": 0.18,
        },
        {
            "name": "suit_dark",
            "diffuse": [0.018, 0.028, 0.055],
            "specular": [0.7, 0.8, 1.0],
            "roughness": 0.48,
            "metallic": 0.08,
        },
        {
            "name": "skin",
            "diffuse": [0.56, 0.28, 0.17],
            "specular": [1.0, 0.82, 0.7],
            "roughness": 0.53,
            "metallic": 0.0,
        },
        {
            "name": "gold_trim",
            "diffuse": [0.82, 0.55, 0.13],
            "specular": [1.0, 0.9, 0.55],
            "roughness": 0.2,
            "metallic": 1.0,
        },
        {
            "name": "visor",
            "diffuse": [0.018, 0.06, 0.085],
            "specular": [0.65, 0.9, 1.0],
            "roughness": 0.08,
            "metallic": 0.72,
        },
        {
            "name": "floor",
            "diffuse": [0.09, 0.1, 0.12],
            "specular": [0.45, 0.45, 0.5],
            "roughness": 0.78,
            "metallic": 0.0,
        },
    ]

    nodes = [
        _node(
            "floor",
            box(5.0, 0.08, 5.0),
            "floor",
            translation(0.0, -0.04, 0.0),
        ),
        _node(
            "pelvis",
            box(0.46, 0.24, 0.3),
            "suit_dark",
            translation(0.0, 0.91, 0.0),
        ),
        _node(
            "torso",
            box(0.62, 0.68, 0.36),
            "suit_blue",
            translation(0.0, 1.39, 0.0),
        ),
        _node(
            "chest_trim",
            box(0.66, 0.09, 0.39),
            "gold_trim",
            translation(0.0, 1.18, 0.0),
        ),
        _node(
            "head",
            sphere(0.24, 32, 20),
            "skin",
            translation(0.0, 1.98, 0.0),
        ),
        _node(
            "visor",
            sphere(0.245, 32, 20),
            "visor",
            compose(translation(0.0, 1.99, 0.035), scaling(0.92, 0.44, 0.9)),
        ),
        _node(
            "arm_left",
            cylinder(0.085, 0.68, 24),
            "suit_blue",
            compose(translation(-0.42, 1.28, 0.0), rotation_z(-12.0)),
        ),
        _node(
            "arm_right",
            cylinder(0.085, 0.68, 24),
            "suit_blue",
            compose(translation(0.42, 1.28, 0.0), rotation_z(12.0)),
        ),
        _node(
            "leg_left",
            cylinder(0.11, 0.82, 24),
            "suit_dark",
            compose(translation(-0.17, 0.46, 0.0), rotation_z(-2.0)),
        ),
        _node(
            "leg_right",
            cylinder(0.11, 0.82, 24),
            "suit_dark",
            compose(translation(0.17, 0.46, 0.0), rotation_z(2.0)),
        ),
        _node(
            "shoulder_left",
            sphere(0.13, 24, 14),
            "gold_trim",
            translation(-0.31, 1.67, 0.0),
        ),
        _node(
            "shoulder_right",
            sphere(0.13, 24, 14),
            "gold_trim",
            translation(0.31, 1.67, 0.0),
        ),
    ]

    return {
        "schema_version": 1,
        "generator": "tools/generate_character.py",
        "units": "meters",
        "color_space": "linear",
        "coordinate_system": {
            "handedness": "right",
            "up": "+Y",
            "forward": "-Z",
            "matrix_layout": "column-major",
        },
        "materials": materials,
        "nodes": nodes,
        "scene": {
            "ambient": [0.035, 0.045, 0.07],
            "directional": {
                "name": "key_sun",
                "direction": [-0.45, -1.0, -0.35],
                "color": [1.0, 0.94, 0.84],
                "intensity": 3.0,
                "cast_shadow": True,
            },
            "point_lights": [
                {
                    "name": "warm_key_left",
                    "position": [-1.8, 2.55, 2.0],
                    "color": [1.0, 0.56, 0.28],
                    "intensity": 24.0,
                    "range": 9.0,
                    "cast_shadow": True,
                },
                {
                    "name": "cool_fill_right",
                    "position": [2.0, 1.65, 1.35],
                    "color": [0.3, 0.58, 1.0],
                    "intensity": 19.0,
                    "range": 8.0,
                    "cast_shadow": True,
                },
                {
                    "name": "violet_rim_back",
                    "position": [0.0, 2.35, -2.2],
                    "color": [0.58, 0.3, 1.0],
                    "intensity": 27.0,
                    "range": 9.0,
                    "cast_shadow": True,
                },
                {
                    "name": "low_weak",
                    "position": [-2.4, 0.55, -1.0],
                    "color": [1.0, 0.35, 0.18],
                    "intensity": 3.0,
                    "range": 6.0,
                    "cast_shadow": True,
                },
                {
                    "name": "far_weak",
                    "position": [4.0, 3.0, -4.0],
                    "color": [0.4, 1.0, 0.65],
                    "intensity": 1.0,
                    "range": 5.0,
                    "cast_shadow": True,
                },
            ],
            "camera": {
                "position": [3.1, 2.05, 4.15],
                "target": [0.0, 1.08, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_deg": 38.0,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="samples/character.json", help="output JSON path")
    parser.add_argument("--compact", action="store_true", help="write compact JSON")
    args = parser.parse_args()
    model = build_model()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        model,
        sort_keys=True,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    )
    output.write_text(payload + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(model['nodes'])} nodes, {len(model['materials'])} materials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

