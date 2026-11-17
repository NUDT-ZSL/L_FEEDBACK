#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix


def rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[1:3, 1:3] = ((cosine, -sine), (sine, cosine))
    return matrix


def rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = cosine
    matrix[0, 2] = sine
    matrix[2, 0] = -sine
    matrix[2, 2] = cosine
    return matrix


def rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0:2, 0:2] = ((cosine, -sine), (sine, cosine))
    return matrix


class MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[list[float]] = []
        self.normals: list[list[float]] = []
        self.skin_indices: list[list[int]] = []
        self.skin_weights: list[list[float]] = []
        self.triangles: list[dict] = []

    def vertex(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        influences: list[tuple[int, float]],
    ) -> int:
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-12:
            raise ValueError("zero normal")
        weights = [0.0, 0.0, 0.0, 0.0]
        indices = [0, 0, 0, 0]
        total = sum(weight for _, weight in influences)
        for slot, (bone, weight) in enumerate(influences[:4]):
            indices[slot] = int(bone)
            weights[slot] = float(weight / total)
        self.vertices.append([float(value) for value in position])
        self.normals.append(
            [float(value) for value in normal / normal_length]
        )
        self.skin_indices.append(indices)
        self.skin_weights.append(weights)
        return len(self.vertices) - 1

    def triangle(self, a: int, b: int, c: int, material: int) -> None:
        self.triangles.append(
            {"indices": [int(a), int(b), int(c)], "material": int(material)}
        )


def add_floor(mesh: MeshBuilder) -> None:
    extent = 5.0
    normal = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    corners = [
        mesh.vertex(np.asarray([-extent, 0.0, -extent]), normal, [(0, 1.0)]),
        mesh.vertex(np.asarray([extent, 0.0, -extent]), normal, [(0, 1.0)]),
        mesh.vertex(np.asarray([extent, 0.0, extent]), normal, [(0, 1.0)]),
        mesh.vertex(np.asarray([-extent, 0.0, extent]), normal, [(0, 1.0)]),
    ]
    mesh.triangle(corners[0], corners[1], corners[2], 0)
    mesh.triangle(corners[0], corners[2], corners[3], 0)


def add_trunk(mesh: MeshBuilder) -> None:
    base = len(mesh.vertices)
    segments = 12
    ring_count = 10
    y_start = 0.18
    y_end = 1.92
    for ring in range(ring_count):
        t = ring / (ring_count - 1)
        y = y_start + (y_end - y_start) * t
        radius = 0.56 * (1.0 - 0.52 * t) + 0.04 * math.sin(t * math.pi * 3.0)
        if t < 0.34:
            spine_weight = 0.0
        elif t < 0.68:
            spine_weight = (t - 0.34) / 0.34
        else:
            spine_weight = 1.0
        influences = [(1, 1.0 - spine_weight), (2, spine_weight)]
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            normal = np.asarray(
                [math.cos(angle), 0.08, math.sin(angle)], dtype=np.float64
            )
            position = np.asarray(
                [
                    radius * math.cos(angle),
                    y,
                    radius * math.sin(angle),
                ],
                dtype=np.float64,
            )
            mesh.vertex(position, normal, influences)
    for ring in range(ring_count - 1):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = base + ring * segments + segment
            b = base + ring * segments + next_segment
            c = base + (ring + 1) * segments + next_segment
            d = base + (ring + 1) * segments + segment
            material = 1 if ring < 4 else 2
            mesh.triangle(a, b, c, material)
            mesh.triangle(a, c, d, material)
    top_center = mesh.vertex(
        np.asarray([0.0, y_end, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        [(2, 1.0)],
    )
    top_ring = base + (ring_count - 1) * segments
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        mesh.triangle(
            top_ring + segment,
            top_ring + next_segment,
            top_center,
            2,
        )
    bottom_center = mesh.vertex(
        np.asarray([0.0, y_start, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
        [(1, 1.0)],
    )
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        mesh.triangle(
            base + segment,
            bottom_center,
            base + next_segment,
            1,
        )


def add_arm(mesh: MeshBuilder) -> None:
    base = len(mesh.vertices)
    segments = 8
    ring_count = 8
    x_start = 0.28
    x_end = 1.82
    y_center = 1.48
    for ring in range(ring_count):
        t = ring / (ring_count - 1)
        x = x_start + (x_end - x_start) * t
        radius = 0.22 * (1.0 - 0.55 * t)
        arm_weight = min(1.0, max(0.0, (t - 0.25) / 0.5))
        influences = [(2, 1.0 - arm_weight), (3, arm_weight)]
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            normal = np.asarray(
                [0.0, math.cos(angle), math.sin(angle)], dtype=np.float64
            )
            position = np.asarray(
                [
                    x,
                    y_center + radius * math.cos(angle),
                    radius * math.sin(angle),
                ],
                dtype=np.float64,
            )
            mesh.vertex(position, normal, influences)
    for ring in range(ring_count - 1):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = base + ring * segments + segment
            b = base + ring * segments + next_segment
            c = base + (ring + 1) * segments + next_segment
            d = base + (ring + 1) * segments + segment
            mesh.triangle(a, b, c, 3)
            mesh.triangle(a, c, d, 3)
    end_center = mesh.vertex(
        np.asarray([x_end, y_center, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        [(3, 1.0)],
    )
    end_ring = base + (ring_count - 1) * segments
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        mesh.triangle(end_ring + segment, end_ring + next_segment, end_center, 3)
    start_center = mesh.vertex(
        np.asarray([x_start, y_center, 0.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        [(2, 1.0)],
    )
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        mesh.triangle(base + segment, start_center, base + next_segment, 3)


def bind_world_matrices() -> list[np.ndarray]:
    ground = np.eye(4, dtype=np.float64)
    root = np.eye(4, dtype=np.float64)
    spine = translation(0.0, 0.9, 0.0)
    arm = translation(0.35, 1.48, 0.0)
    return [ground, root, spine, arm]


def local_rest_matrices(bind_world: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.eye(4, dtype=np.float64),
        np.eye(4, dtype=np.float64),
        np.linalg.inv(bind_world[1]) @ bind_world[2],
        np.linalg.inv(bind_world[2]) @ bind_world[3],
    ]


def frame_locals(
    frame: int, local_rest: list[np.ndarray]
) -> list[np.ndarray]:
    root_angle = 0.08 * math.sin(frame * 0.82)
    spine_angle_z = 0.22 * math.sin(frame * 0.82 + 0.35)
    spine_angle_x = 0.10 * math.sin(frame * 0.55 + 0.8)
    arm_angle = 0.48 * math.sin(frame * 0.82 + 0.9)
    return [
        np.eye(4, dtype=np.float64),
        local_rest[1] @ rotation_z(root_angle),
        local_rest[2] @ rotation_z(spine_angle_z) @ rotation_x(spine_angle_x),
        local_rest[3] @ rotation_z(arm_angle),
    ]


def build_scene(
    frame_numbers: list[int],
    width: int,
    height: int,
    shadow_resolution: int,
) -> dict:
    bind_world = bind_world_matrices()
    local_rest = local_rest_matrices(bind_world)
    inverse_bind = [np.linalg.inv(matrix) for matrix in bind_world]
    mesh = MeshBuilder()
    add_floor(mesh)
    add_trunk(mesh)
    add_arm(mesh)
    bone_names = ["ground", "root", "spine", "arm"]
    parents = [-1, -1, 1, 2]
    bones = []
    for index, name in enumerate(bone_names):
        local_frames = []
        for frame in frame_numbers:
            matrices = frame_locals(frame, local_rest)
            local_frames.append(
                {
                    "frame": frame,
                    "matrix": matrices[index].reshape(-1).tolist(),
                }
            )
        bones.append(
            {
                "name": name,
                "parent": parents[index],
                "inverse_bind_matrix": inverse_bind[index].reshape(-1).tolist(),
                "local_frames": local_frames,
            }
        )
    return {
        "schema_version": 1,
        "scene_id": "deterministic_skinned_demo",
        "output": {
            "width": width,
            "height": height,
            "background": [0.025, 0.03, 0.045],
            "frame_pattern": "frame_{frame:04d}.png",
            "stats_pattern": "frame_{frame:04d}.json",
            "manifest_name": "manifest.json",
        },
        "camera": {
            "position": [4.6, 2.75, 6.1],
            "target": [0.0, 1.05, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_degrees": 43.0,
            "near": 0.1,
            "far": 30.0,
        },
        "lights": {
            "ambient": [0.11, 0.12, 0.15],
            "directional": {
                "direction": [0.42, 0.88, 0.32],
                "color": [1.0, 0.93, 0.82],
                "intensity": 1.15,
                "cast_shadow": True,
            },
            "point": {
                "position": [4.6, 2.75, 6.1],
                "color": [0.55, 0.68, 1.0],
                "intensity": 2.0,
                "falloff": 0.045,
            },
        },
        "shadow": {
            "resolution": shadow_resolution,
            "splits": [6.0, 11.0, 24.0],
            "bias": 0.001,
            "slope_bias": 0.004,
            "normal_offset": 0.014,
            "extent_padding": 1.14,
            "depth_padding": 2.5,
        },
        "materials": [
            {
                "name": "floor",
                "diffuse": [0.42, 0.47, 0.52],
                "specular": [0.08, 0.08, 0.09],
                "shininess": 24.0,
            },
            {
                "name": "ember",
                "diffuse": [0.72, 0.18, 0.10],
                "specular": [0.55, 0.35, 0.28],
                "shininess": 42.0,
            },
            {
                "name": "gold",
                "diffuse": [0.78, 0.48, 0.09],
                "specular": [0.62, 0.50, 0.22],
                "shininess": 36.0,
            },
            {
                "name": "teal",
                "diffuse": [0.10, 0.55, 0.62],
                "specular": [0.42, 0.65, 0.70],
                "shininess": 60.0,
            },
        ],
        "skeleton": {"bones": bones},
        "animation": {"fps": 24, "frames": frame_numbers},
        "mesh": {
            "vertices": mesh.vertices,
            "normals": mesh.normals,
            "skin_indices": mesh.skin_indices,
            "skin_weights": mesh.skin_weights,
            "triangles": mesh.triangles,
        },
    }


def parse_frames(args: argparse.Namespace) -> list[int]:
    if args.frame_numbers:
        values = [int(item.strip()) for item in args.frame_numbers.split(",")]
        if len(set(values)) != len(values) or any(value < 0 for value in values):
            raise ValueError("frame numbers must be unique non-negative integers")
        return sorted(values)
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    return list(range(args.frames))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic skinned scene JSON file."
    )
    parser.add_argument(
        "--output", default="examples/demo_scene.json", help="destination JSON path"
    )
    parser.add_argument(
        "--frames", type=int, default=4, help="number of frames 0..N-1"
    )
    parser.add_argument(
        "--frame-numbers",
        default=None,
        help="comma-separated explicit frame numbers (overrides --frames)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--shadow-resolution", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        frames = parse_frames(args)
        scene = build_scene(
            frames,
            args.width,
            args.height,
            args.shadow_resolution,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scene, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": frames,
                "vertices": len(scene["mesh"]["vertices"]),
                "triangles": len(scene["mesh"]["triangles"]),
                "materials": len(scene["materials"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
