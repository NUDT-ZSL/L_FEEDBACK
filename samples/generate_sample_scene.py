"""Generate the deterministic sample scene used by the command-line demo."""

from __future__ import annotations

import json
from pathlib import Path


def triangle(a, b, c, normal, material):
    return {
        "positions": [list(a), list(b), list(c)],
        "normals": [list(normal), list(normal), list(normal)],
        "material": material,
    }


def quad(a, b, c, d, normal, material):
    return [
        triangle(a, b, c, normal, material),
        triangle(a, c, d, normal, material),
    ]


def cube(cx, cy, cz, sx, sy, sz, material):
    x0, x1 = cx - sx * 0.5, cx + sx * 0.5
    y0, y1 = cy, cy + sy
    z0, z1 = cz - sz * 0.5, cz + sz * 0.5
    triangles = []
    triangles += quad((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (0, -1, 0), material)
    triangles += quad((x0, y1, z1), (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (0, 1, 0), material)
    triangles += quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (0, 0, 1), material)
    triangles += quad((x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0), (0, 0, -1), material)
    triangles += quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1), (-1, 0, 0), material)
    triangles += quad((x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (x1, y0, z0), (1, 0, 0), material)
    return triangles


def build_scene():
    triangles = []
    triangles += quad(
        (-4.5, 0.0, -3.0),
        (-4.5, 0.0, 3.2),
        (4.5, 0.0, 3.2),
        (4.5, 0.0, -3.0),
        (0.0, 1.0, 0.0),
        "floor",
    )
    triangles += quad(
        (-4.5, 0.0, -2.2),
        (-4.5, 3.2, -2.2),
        (4.5, 3.2, -2.2),
        (4.5, 0.0, -2.2),
        (0.0, 0.0, 1.0),
        "wall",
    )
    triangles += cube(-1.15, 0.0, -0.35, 1.45, 1.65, 1.45, "red")
    triangles += cube(1.15, 0.0, -0.6, 1.1, 1.15, 1.1, "blue")
    triangles += cube(-0.25, 0.0, 1.15, 0.75, 0.7, 0.75, "green")
    return {
        "scene_id": "deterministic-deferred-sample",
        "version": 1,
        "frame": {"width": 128, "height": 96},
        "tile": {"width": 32, "height": 24},
        "camera": {
            "position": [3.25, 2.55, 5.35],
            "target": [0.0, 0.8, -0.25],
            "up": [0.0, 1.0, 0.0],
            "fov_y_degrees": 52.0,
            "near": 0.05,
            "far": 50.0,
        },
        "background": [0.018, 0.024, 0.038],
        "ambient": 0.055,
        "materials": [
            {
                "id": "floor",
                "albedo": [0.72, 0.68, 0.58],
                "roughness": 0.7,
                "specular": 0.15,
            },
            {
                "id": "wall",
                "albedo": [0.42, 0.52, 0.66],
                "roughness": 0.85,
                "specular": 0.08,
            },
            {
                "id": "red",
                "albedo": [0.78, 0.16, 0.12],
                "roughness": 0.45,
                "specular": 0.3,
            },
            {
                "id": "blue",
                "albedo": [0.12, 0.34, 0.82],
                "roughness": 0.4,
                "specular": 0.4,
            },
            {
                "id": "green",
                "albedo": [0.18, 0.72, 0.3],
                "roughness": 0.55,
                "specular": 0.2,
            },
        ],
        "triangles": triangles,
        "lights": [
            {
                "id": "sun",
                "type": "directional",
                "direction": [-0.45, -1.0, -0.32],
                "color": [1.0, 0.84, 0.62],
                "intensity": 1.15,
                "resolution": 192,
                "pcf_kernel": 5,
                "bias": 0.003,
                "normal_bias": 0.025,
            },
            {
                "id": "lamp",
                "type": "point",
                "position": [1.25, 2.35, -0.55],
                "color": [0.38, 0.72, 1.0],
                "intensity": 3.2,
                "range": 7.0,
                "resolution": 96,
                "bias": 0.002,
                "normal_bias": 0.02,
            },
        ],
        "quality": "high",
        "shadow": {
            "quality": "high",
            "pcf_kernel": 5,
            "bias": 0.003,
            "normal_bias": 0.025,
            "depth_epsilon": 0.00001,
        },
        "memory_limit_bytes": 67108864,
    }


def main():
    scene = build_scene()
    destination = Path(__file__).resolve().parents[1] / "sample_scene.json"
    destination.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
