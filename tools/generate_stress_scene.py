#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


COLUMNS = 100
ROWS = 50


def height(x: float, z: float) -> float:
    return 0.10 * math.sin(1.2 * x) * math.cos(1.35 * z)


def normal(x: float, z: float) -> np.ndarray:
    dx = 0.12 * math.cos(1.2 * x) * math.cos(1.35 * z)
    dz = -0.135 * math.sin(1.2 * x) * math.sin(1.35 * z)
    vector = np.asarray([-dx, 1.0, -dz], dtype=np.float64)
    return vector / np.linalg.norm(vector)


def rotation_y(angle: float) -> list[float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = cosine
    matrix[0, 2] = sine
    matrix[2, 0] = -sine
    matrix[2, 2] = cosine
    return matrix.reshape(-1).tolist()


def build_scene(width: int, height_px: int, shadow_resolution: int) -> dict:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    skin_indices: list[list[int]] = []
    skin_weights: list[list[float]] = []
    for row in range(ROWS):
        z = -3.0 + 6.0 * row / (ROWS - 1)
        for column in range(COLUMNS):
            x = -4.0 + 8.0 * column / (COLUMNS - 1)
            vertices.append([x, height(x, z), z])
            normals.append(normal(x, z).tolist())
            blend = min(1.0, max(0.0, (x + 4.0) / 8.0))
            skin_indices.append([0, 1, 0, 0])
            skin_weights.append([1.0 - blend, blend, 0.0, 0.0])
    triangles: list[dict] = []
    for row in range(ROWS - 1):
        for column in range(COLUMNS - 1):
            a = row * COLUMNS + column
            b = a + 1
            c = a + COLUMNS
            d = c + 1
            material = (row // 8 + column // 12) % 2
            triangles.append({"indices": [a, b, d], "material": material})
            triangles.append({"indices": [a, d, c], "material": material})
    identity = np.eye(4, dtype=np.float64).reshape(-1).tolist()
    animated = rotation_y(0.13)
    return {
        "schema_version": 1,
        "scene_id": "max_5000_vertex_stress",
        "output": {
            "width": width,
            "height": height_px,
            "background": [0.02, 0.025, 0.035],
            "frame_pattern": "frame_{frame:04d}.png",
            "stats_pattern": "frame_{frame:04d}.json",
            "manifest_name": "manifest.json",
        },
        "camera": {
            "position": [0.0, 6.8, 9.2],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_degrees": 48.0,
            "near": 0.1,
            "far": 30.0,
        },
        "lights": {
            "ambient": [0.08, 0.09, 0.11],
            "directional": {
                "direction": [0.4, 0.9, 0.25],
                "color": [1.0, 0.95, 0.86],
                "intensity": 1.1,
                "cast_shadow": True,
            },
            "point": {
                "position": [0.0, 6.8, 9.2],
                "color": [0.55, 0.68, 1.0],
                "intensity": 1.5,
                "falloff": 0.035,
            },
        },
        "shadow": {
            "resolution": shadow_resolution,
            "splits": [7.0, 13.0, 25.0],
            "bias": 0.001,
            "slope_bias": 0.004,
            "normal_offset": 0.012,
            "extent_padding": 1.15,
            "depth_padding": 2.5,
        },
        "materials": [
            {
                "name": "indigo",
                "diffuse": [0.16, 0.28, 0.68],
                "specular": [0.45, 0.50, 0.62],
                "shininess": 42.0,
            },
            {
                "name": "sand",
                "diffuse": [0.68, 0.43, 0.18],
                "specular": [0.35, 0.28, 0.18],
                "shininess": 28.0,
            },
        ],
        "skeleton": {
            "bones": [
                {
                    "name": "root",
                    "parent": -1,
                    "inverse_bind_matrix": identity,
                    "local_frames": [{"frame": 0, "matrix": identity}],
                },
                {
                    "name": "twist",
                    "parent": 0,
                    "inverse_bind_matrix": identity,
                    "local_frames": [{"frame": 0, "matrix": animated}],
                },
            ]
        },
        "animation": {"fps": 24, "frames": [0]},
        "mesh": {
            "vertices": vertices,
            "normals": normals,
            "skin_indices": skin_indices,
            "skin_weights": skin_weights,
            "triangles": triangles,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic 5000-vertex resource-bound scene."
    )
    parser.add_argument("--output", default="examples/stress_5000.json")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--shadow-resolution", type=int, default=1024)
    args = parser.parse_args(argv)
    scene = build_scene(args.width, args.height, args.shadow_resolution)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scene, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "vertices": len(scene["mesh"]["vertices"]),
                "triangles": len(scene["mesh"]["triangles"]),
                "width": args.width,
                "height": args.height,
                "shadow_resolution": args.shadow_resolution,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
