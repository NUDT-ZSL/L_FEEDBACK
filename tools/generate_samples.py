#!/usr/bin/env python3
"""Generate deterministic PNG/TGA/scene fixtures for bake_material.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from material_baker.color import linear_to_srgb  # noqa: E402
from material_baker.pngio import write_png, write_tga  # noqa: E402
from material_baker.scene import load_scene  # noqa: E402
from material_baker.tangent import build_tangent_field  # noqa: E402
from material_baker.util import stable_json_text  # noqa: E402


def _world_coordinates(
    width: int, height: int, seam_x: int
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:height, 0:width]
    world_x = np.empty((height, width), dtype=np.float64)
    left_width = seam_x
    right_width = width - seam_x

    left_mask = x < seam_x
    left_u = x[left_mask] / max(left_width - 1, 1)
    world_x[left_mask] = (left_u - 0.5) * 2.0

    right_mask = ~left_mask
    right_u = 1.0 - (x[right_mask] - seam_x) / max(right_width - 1, 1)
    world_x[right_mask] = (right_u - 0.5) * 2.0
    world_y = (0.5 - y / max(height - 1, 1)) * 2.0
    return world_x, world_y


def _height_and_derivatives(
    world_x: np.ndarray, world_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_x = np.pi * world_x
    phase_y = np.pi * world_y
    height = (
        0.40 * np.sin(1.10 * phase_x + 0.35) * np.cos(0.95 * phase_y - 0.25)
        + 0.16 * np.sin(2.25 * phase_x + 1.10 * phase_y + 0.4)
        + 0.10 * np.cos(2.80 * phase_y - 0.7 * phase_x)
        + 0.045 * np.sin(7.0 * phase_x + 5.0 * phase_y)
    )
    derivative_x = (
        0.40
        * 1.10
        * np.pi
        * np.cos(1.10 * phase_x + 0.35)
        * np.cos(0.95 * phase_y - 0.25)
        + 0.16 * 2.25 * np.pi * np.cos(2.25 * phase_x + 1.10 * phase_y + 0.4)
        - 0.10 * 0.7 * np.pi * np.sin(2.80 * phase_y - 0.7 * phase_x)
        + 0.045 * 7.0 * np.pi * np.cos(7.0 * phase_x + 5.0 * phase_y)
    )
    derivative_y = (
        -0.40
        * 0.95
        * np.pi
        * np.sin(1.10 * phase_x + 0.35)
        * np.sin(0.95 * phase_y - 0.25)
        + 0.16 * 1.10 * np.pi * np.cos(2.25 * phase_x + 1.10 * phase_y + 0.4)
        - 0.10 * 2.80 * np.pi * np.sin(2.80 * phase_y - 0.7 * phase_x)
        + 0.045 * 5.0 * np.pi * np.cos(7.0 * phase_x + 5.0 * phase_y)
    )
    return height, derivative_x, derivative_y


def generate(
    out_dir: Path,
    *,
    width: int,
    height: int,
    variant: str,
    seed: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    textures = out_dir / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    seam_x = width // 2
    world_x, world_y = _world_coordinates(width, height, seam_x)
    height_field, dh_dx, dh_dy = _height_and_derivatives(world_x, world_y)
    normal_scale = 0.105
    object_normal = np.stack(
        [-normal_scale * dh_dx, -normal_scale * dh_dy, np.ones_like(dh_dx)],
        axis=2,
    )
    object_normal /= np.linalg.norm(object_normal, axis=2, keepdims=True)

    normalized_height = np.clip((height_field + 0.62) / 1.24, 0.0, 1.0)
    fine = (
        0.5
        + 0.5
        * np.sin(11.0 * np.pi * world_x + 3.0 * world_y)
        * np.cos(9.0 * np.pi * world_y - 2.0 * world_x)
    )
    base_linear = np.stack(
        [
            0.055 + 0.46 * normalized_height + 0.035 * fine,
            0.090 + 0.31 * normalized_height + 0.022 * fine,
            0.045 + 0.15 * normalized_height + 0.012 * fine,
        ],
        axis=2,
    )
    base_linear = np.clip(base_linear, 0.0, 1.0).astype(np.float32)
    if variant == "hard-edge":
        base_linear[:, seam_x:, :] = np.clip(
            base_linear[:, seam_x:, :] * 1.65 + np.float32(0.045), 0.0, 1.0
        )
    base_srgb = linear_to_srgb(base_linear)

    scene_payload = {
        "schema_version": 1,
        "name": f"deterministic-{variant}",
        "assets": {
            "base_color": {
                "path": "textures/basecolor.png",
                "color_space": "srgb",
            },
            "normal": {
                "path": "textures/normal.tga",
                "space": "tangent",
                "green_sign": 1,
            },
            "roughness": {
                "path": "textures/roughness.png",
                "channel": 0,
                "min": 0.04,
                "max": 1.0,
            },
            "metallic": {
                "path": "textures/metallic.png",
                "channel": 0,
                "min": 0.0,
                "max": 1.0,
            },
            "defaults": {"roughness": 0.48, "metallic": 0.05},
        },
        "surface": {
            "geometry_normal": [0.0, 0.0, 1.0],
            "uv_axis_u": [1.0, 0.0, 0.0],
            "uv_axis_v": [0.0, -1.0, 0.0],
            "uv_origin": [0.0, 0.0, 0.0],
            "material_size": [2.0, 2.0],
            "normal_green_sign": 1.0,
            "seam_smoothing_radius": 1,
            "islands": [
                {
                    "id": 0,
                    "bounds": [0, 0, seam_x, height],
                    "flip_u": False,
                    "flip_v": False,
                },
                {
                    "id": 1,
                    "bounds": [seam_x, 0, width - seam_x, height],
                    "flip_u": variant != "hard-edge",
                    "flip_v": False,
                },
            ],
        },
        "lights": [
            {
                "id": "key",
                "direction": [-0.38, -0.28, 0.88],
                "color": [1.0, 0.91, 0.78],
                "intensity": 3.6,
            },
            {
                "id": "fill",
                "direction": [0.55, -0.18, 0.82],
                "color": [0.62, 0.78, 1.0],
                "intensity": 1.1,
            },
            {
                "id": "rim",
                "direction": [0.10, 0.75, 0.65],
                "color": [1.0, 0.72, 0.50],
                "intensity": 0.65,
            },
        ],
        "environment": {"intensity": 0.30, "color": [0.68, 0.76, 0.94]},
        "camera": {
            "projection": "perspective",
            "position": [0.0, 0.0, 3.15],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_degrees": 38.0,
            "ortho_scale": 2.25,
            "resolution": [384, 288],
        },
        "baking": {
            "chunk_rows": 24,
            "view_direction": [0.0, 0.0, 1.0],
            "exposure": 1.05,
            "normal_strength": 1.0,
        },
        "validation": {
            "seed": seed,
            "max_samples": 4096,
            "stratified": True,
            "normal_length_tolerance": 0.035,
            "angle_threshold_deg": 34.0,
            "seam_angle_threshold_deg": 10.0,
            "reference_gain": 0.85,
            "reference_min_gradient": 0.008,
            "reference_tilt_min": 0.04,
            "seam_width": 2,
            "tile_size": 64,
            "hard_edge_color_threshold": 0.22,
            "max_entries": 512,
        },
        "output": {
            "bit_depth": 8,
            "preview_resolution": [384, 288],
            "filename_pattern": "{kind}.png",
            "background": [0.018, 0.023, 0.032],
        },
    }
    scene_path = out_dir / "scene.json"
    scene_path.write_text(stable_json_text(scene_payload), encoding="utf-8")
    scene = load_scene(scene_path)
    tangent = build_tangent_field(scene, (height, width))

    n = object_normal.astype(np.float32)
    nx = np.sum(n * tangent.tangent, axis=2)
    ny = np.sum(n * tangent.bitangent, axis=2)
    nz = np.sum(n * tangent.geometry_normal[None, None, :], axis=2)
    if variant == "seam-flip":
        nx[:, seam_x:] *= -1.0
    elif variant == "hard-edge":
        nx[:, seam_x:] *= -1.0
    elif variant == "uv-flip":
        ny[:, seam_x:] *= -1.0
    elif variant == "space-mismatch":
        nx = n[:, :, 0]
        ny = n[:, :, 1]
        nz = n[:, :, 2]
    elif variant not in {"clean", "non-unit"}:
        raise ValueError(f"unknown sample variant: {variant}")
    tangent_normal = np.stack([nx, ny, nz], axis=2)
    tangent_normal /= np.maximum(
        np.linalg.norm(tangent_normal, axis=2, keepdims=True), 1e-8
    )
    if variant == "non-unit":
        tangent_normal = np.full_like(tangent_normal, 0.25)
    normal_rgb = np.clip(
        np.rint((tangent_normal * 0.5 + 0.5) * 255.0), 0, 255
    ).astype(np.uint8)

    texture_phase = np.sin(10.0 * np.pi * world_x) * np.cos(
        8.0 * np.pi * world_y
    )
    roughness = np.clip(
        0.30 + 0.28 * normalized_height + 0.06 * texture_phase, 0.04, 1.0
    )
    metal_mask = (
        np.cos(7.0 * world_x + 3.0 * world_y) > 0.55
    ).astype(np.float32)
    metallic = np.clip(0.08 + 0.72 * metal_mask, 0.0, 1.0)

    write_png(textures / "basecolor.png", base_srgb, 8)
    write_tga(textures / "normal.tga", normal_rgb, 8)
    write_png(textures / "roughness.png", roughness[:, :, None], 8)
    write_png(textures / "metallic.png", metallic[:, :, None], 8)
    return scene_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument(
        "--variant",
        choices=[
            "clean",
            "seam-flip",
            "uv-flip",
            "space-mismatch",
            "hard-edge",
            "non-unit",
        ],
        default="clean",
    )
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if args.width < 8 or args.height < 8:
        parser.error("--width and --height must be at least 8")
    path = generate(
        args.out.resolve(),
        width=args.width,
        height=args.height,
        variant=args.variant,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
