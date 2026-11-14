"""Deterministic scene generation, camera math and mesh construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    f = (np.asarray(target, dtype=np.float32) - eye)
    f /= np.linalg.norm(f)
    s = np.cross(f, np.asarray(up, dtype=np.float32))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    # Standard gluLookAt view matrix (column-major storage via explicit rows).
    return np.array([
        [s[0], s[1], s[2], -np.dot(s, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def make_unit_sphere(segments: int = 32, rings: int = 24) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    indices: list[int] = []
    for r in range(rings + 1):
        theta = np.pi * r / rings
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        for s in range(segments + 1):
            phi = 2.0 * np.pi * s / segments
            normal = np.array([
                sin_t * np.cos(phi),
                cos_t,
                sin_t * np.sin(phi),
            ], dtype=np.float32)
            vertices.append([normal[0], normal[1], normal[2],
                             normal[0], normal[1], normal[2]])
    for r in range(rings):
        for s in range(segments):
            first = r * (segments + 1) + s
            second = first + segments + 1
            indices.extend([first, second, first + 1,
                            second, second + 1, first + 1])
    data = np.asarray(vertices, dtype=np.float32).reshape(-1, 6)
    return data, np.asarray(indices, dtype=np.uint32)


def make_ground_plane(size: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    h = size / 2.0
    # Vertices are listed counter-clockwise when viewed from above (+Y), so
    # the upward-facing normal side survives GL_BACK culling with CCW front.
    data = np.asarray([
        [-h, 0.0, -h, 0.0, 1.0, 0.0],
        [-h, 0.0, h, 0.0, 1.0, 0.0],
        [h, 0.0, -h, 0.0, 1.0, 0.0],
        [-h, 0.0, -h, 0.0, 1.0, 0.0],
        [h, 0.0, h, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return data, indices


@dataclass
class Mesh:
    vao: int
    vbo: int
    ebo: int
    index_count: int


def build_scene(seed: int) -> dict[str, Any]:
    """Generate the full scene deterministically from a single integer seed."""
    rng = np.random.default_rng(seed)

    material_count = 5
    materials = []
    for idx in range(material_count):
        materials.append({
            "id": idx,
            "name": f"material_{idx}",
            "albedo": [round(float(v), 5) for v in rng.uniform(0.08, 0.9, 3)],
            "roughness": round(float(rng.uniform(0.08, 0.95)), 5),
            "metallic": round(float(rng.uniform(0.0, 1.0)), 5),
            "emissive": [0.0, 0.0, 0.0],
        })

    sphere_count = 5
    spheres = []
    for idx in range(sphere_count):
        spheres.append({
            "id": idx,
            "material_id": int(rng.integers(0, material_count)),
            "radius": round(float(rng.uniform(0.45, 0.9)), 5),
            "orbit_radius": round(float(rng.uniform(1.4, 3.4)), 5),
            "phase": round(float(rng.uniform(0.0, 2.0 * np.pi)), 5),
            "angular_speed": round(float(rng.uniform(0.15, 0.45)), 5),
            "height": round(float(rng.uniform(0.6, 1.4)), 5),
        })

    point_lights = []
    for idx in range(2):
        point_lights.append({
            "id": idx,
            "position": [round(float(v), 5) for v in rng.uniform(-4.0, 4.0, 3)],
            "color": [round(float(v), 5) for v in rng.uniform(0.4, 1.0, 3)],
            "intensity": round(float(rng.uniform(12.0, 28.0)), 5),
            "range": round(float(rng.uniform(8.0, 14.0)), 5),
        })

    dir_direction = rng.uniform(-1.0, 1.0, 3)
    dir_direction[1] = -abs(dir_direction[1]) - 0.4
    dir_direction /= np.linalg.norm(dir_direction)

    scene = {
        "seed": seed,
        "rng": "numpy.random.default_rng",
        "camera": {
            "position": [0.0, 3.2, 7.5],
            "target": [0.0, 0.7, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 50.0,
            "near": 0.1,
            "far": 100.0,
        },
        "ground": {
            "size": 20.0,
            "material_id": material_count - 1,
        },
        "materials": materials,
        "spheres": spheres,
        "point_lights": point_lights,
        "directional_light": {
            "direction": [round(float(v), 6) for v in dir_direction],
            "color": [1.0, 0.96, 0.9],
            "intensity": 2.2,
        },
    }
    # Make the ground matte/dielectric and give one sphere an emissive material.
    scene["materials"][-1]["roughness"] = 0.85
    scene["materials"][-1]["metallic"] = 0.0
    scene["materials"][-1]["albedo"] = [0.32, 0.33, 0.36]
    scene["materials"][0]["emissive"] = [0.35, 0.12, 0.05]
    return scene


def write_scene(scene: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_scene(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sphere_model(sphere: dict[str, Any], frame_index: int) -> np.ndarray:
    angle = sphere["phase"] + sphere["angular_speed"] * frame_index
    x = sphere["orbit_radius"] * np.cos(angle)
    z = sphere["orbit_radius"] * np.sin(angle)
    y = sphere["height"]
    radius = sphere["radius"]
    model = np.eye(4, dtype=np.float32)
    model[0, 0] = radius
    model[1, 1] = radius
    model[2, 2] = radius
    model[0, 3] = x
    model[1, 3] = y
    model[2, 3] = z
    return model
