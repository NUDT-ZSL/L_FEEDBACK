from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .jsonio import atomic_write_json


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_sample(
    *,
    frame_count: int = 240,
    width: int = 128,
    height: int = 72,
    samples_per_axis: int = 4,
    seed: int = 0x5EED1234,
) -> dict:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    spheres = [
        {
            "id": "center",
            "center": [0.0, 0.05, 0.0],
            "radius": 1.05,
            "albedo": [0.82, 0.20, 0.12],
            "specular": 0.65,
            "shininess": 80.0,
        },
        {
            "id": "teal",
            "center": [-1.75, -0.35, -0.7],
            "radius": 0.70,
            "albedo": [0.08, 0.58, 0.62],
            "specular": 0.45,
            "shininess": 44.0,
        },
        {
            "id": "gold",
            "center": [1.65, -0.25, -0.9],
            "radius": 0.78,
            "albedo": [0.90, 0.58, 0.10],
            "specular": 0.55,
            "shininess": 60.0,
        },
        {
            "id": "violet",
            "center": [0.7, 0.95, -1.8],
            "radius": 0.48,
            "albedo": [0.38, 0.16, 0.78],
            "specular": 0.35,
            "shininess": 28.0,
        },
        {
            "id": "pale",
            "center": [-0.85, 1.35, -2.0],
            "radius": 0.38,
            "albedo": [0.72, 0.78, 0.84],
            "specular": 0.75,
            "shininess": 110.0,
        },
        {
            "id": "rear",
            "center": [2.35, 1.25, 1.25],
            "radius": 0.55,
            "albedo": [0.24, 0.58, 0.24],
            "specular": 0.30,
            "shininess": 24.0,
        },
    ]
    frames = []
    for frame in range(frame_count):
        phase = 2.0 * math.pi * frame / frame_count
        radius = 6.4 + 0.35 * math.sin(phase * 2.0)
        eye = [
            radius * math.cos(phase),
            1.15 + 0.42 * math.sin(phase * 3.0),
            radius * math.sin(phase),
        ]
        target = [
            0.28 * math.sin(phase * 1.5),
            0.18 + 0.12 * math.cos(phase * 2.0),
            -0.15 * math.cos(phase),
        ]
        frames.append(
            {
                "frame": frame,
                "position": eye,
                "target": target,
                "up": [0.0, 1.0, 0.0],
                "seed": splitmix64(seed + frame * 0x9E3779B1),
            }
        )
    return {
        "schema_version": 1,
        "metadata": {
            "name": "deterministic-ray-sphere-demo",
            "description": (
                "240-frame pan/rotation path with six spheres and a fixed "
                "Blinn-Phong point light."
            ),
        },
        "scene": {
            "spheres": spheres,
            "light": {
                "position": [4.6, 6.2, 4.0],
                "color": [1.0, 0.94, 0.86],
                "intensity": 1.15,
            },
            "camera": {
                "width": width,
                "height": height,
                "fov_y_deg": 43.0,
                "position": [6.4, 1.15, 0.0],
                "target": [0.0, 0.18, 0.0],
                "up": [0.0, 1.0, 0.0],
            },
            "ambient": [0.055, 0.065, 0.085],
            "background": [0.012, 0.018, 0.032],
        },
        "animation": {
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "frames": frames,
        },
        "sampling": {
            "samples_per_axis": samples_per_axis,
            "jitter": True,
            "seed_algorithm": "pcg64-grid-jitter-v1",
        },
    }


def write_sample(
    path: str | Path,
    *,
    frame_count: int = 240,
    width: int = 128,
    height: int = 72,
    samples_per_axis: int = 4,
    seed: int = 0x5EED1234,
) -> Path:
    target = Path(path)
    atomic_write_json(
        target,
        build_sample(
            frame_count=frame_count,
            width=width,
            height=height,
            samples_per_axis=samples_per_axis,
            seed=seed,
        ),
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic render scene")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--samples-per-axis", type=int, default=4)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5EED1234)
    args = parser.parse_args(argv)
    write_sample(
        args.output,
        frame_count=args.frames,
        width=args.width,
        height=args.height,
        samples_per_axis=args.samples_per_axis,
        seed=args.seed,
    )
    print(json.dumps({"status": "ok", "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

