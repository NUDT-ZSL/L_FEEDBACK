#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def sample_scene() -> dict[str, Any]:
    mask_size = [512, 512]
    return {
        "schema_version": 1,
        "scene_id": "uid48_procedural_character",
        "uv_layout": {
            "width": 512,
            "height": 512,
            "mesh": {
                "u_segments": 96,
                "v_segments": 128,
                "radius_x": 0.40,
                "radius_z": 0.27,
                "height": 1.16,
                "profile_amplitude": 0.055,
                "profile_waves": 2.0,
            },
        },
        "material": {
            "color_space": "srgb",
            "base_color": [0.24, 0.31, 0.38],
            "base_roughness": 0.62,
            "base_metallic": 0.04,
            "layers": [
                {
                    "id": "woven_checker",
                    "channel": "albedo",
                    "blend": "multiply",
                    "opacity": 0.36,
                    "tiling": [2.0, 2.0],
                    "rotation_deg": 17.0,
                    "pattern": {
                        "type": "checker",
                        "frequency": [8.0, 8.0],
                        "color_a": [0.95, 0.91, 0.81],
                        "color_b": [0.20, 0.23, 0.27],
                    },
                },
                {
                    "id": "signal_stripes",
                    "channel": "albedo",
                    "blend": "screen",
                    "opacity": 0.18,
                    "tiling": [1.5, 3.2],
                    "rotation_deg": -22.0,
                    "pattern": {
                        "type": "stripes",
                        "frequency": 7.0,
                        "duty": 0.22,
                        "axis": "y",
                        "color_a": [0.02, 0.03, 0.04],
                        "color_b": [0.98, 0.52, 0.18],
                    },
                },
                {
                    "id": "soft_variation",
                    "channel": "albedo",
                    "blend": "overlay",
                    "opacity": 0.12,
                    "tiling": [3.0, 2.2],
                    "rotation_deg": 8.0,
                    "pattern": {
                        "type": "noise",
                        "frequency": 4,
                        "octaves": 3,
                        "seed": 73,
                        "color_a": [0.27, 0.34, 0.42],
                        "color_b": [0.83, 0.61, 0.40],
                    },
                },
                {
                    "id": "roughness_mottle",
                    "channel": "roughness",
                    "mask_size": mask_size,
                    "blend": "normal",
                    "opacity": 0.68,
                    "tiling": [2.5, 3.5],
                    "rotation_deg": 11.0,
                    "pattern": {
                        "type": "noise",
                        "frequency": 4,
                        "octaves": 3,
                        "seed": 118,
                        "value_a": 0.34,
                        "value_b": 0.88,
                    },
                },
                {
                    "id": "metal_insignia",
                    "channel": "metallic",
                    "mask_size": mask_size,
                    "blend": "normal",
                    "opacity": 0.62,
                    "tiling": [2.0, 2.0],
                    "rotation_deg": 37.0,
                    "pattern": {
                        "type": "noise",
                        "frequency": 3,
                        "octaves": 2,
                        "seed": 41,
                        "value_a": 0.02,
                        "value_b": 0.64,
                    },
                },
            ],
        },
        "cameras": [
            {
                "id": "front",
                "yaw_deg": 0.0,
                "ortho_height": 1.35,
                "resolution": [160, 200],
            },
            {
                "id": "side",
                "yaw_deg": 90.0,
                "ortho_height": 1.35,
                "resolution": [160, 200],
            },
        ],
        "lighting_presets": [
            {
                "id": "studio_aces",
                "key_light": {
                    "direction": [0.32, 0.58, 0.72],
                    "irradiance": [1.12, 1.08, 1.00],
                },
                "ambient_irradiance": [0.065, 0.076, 0.096],
                "tone_mapping": {
                    "operator": "aces",
                    "exposure": 1.0,
                    "white_point": 4.0,
                },
            },
            {
                "id": "dusk_reinhard",
                "key_light": {
                    "direction": [-0.46, 0.30, 0.82],
                    "irradiance": [0.94, 0.62, 0.42],
                },
                "ambient_irradiance": [0.052, 0.042, 0.071],
                "tone_mapping": {
                    "operator": "reinhard",
                    "exposure": 1.18,
                    "white_point": 3.5,
                },
            },
            {
                "id": "cool_rim_aces",
                "key_light": {
                    "direction": [0.72, 0.18, 0.67],
                    "irradiance": [0.62, 0.78, 1.10],
                },
                "ambient_irradiance": [0.030, 0.046, 0.083],
                "tone_mapping": {
                    "operator": "aces",
                    "exposure": 1.08,
                    "white_point": 4.0,
                },
            },
        ],
    }


def write_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sample_scene(), ensure_ascii=True, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the built-in deterministic scene sample.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples/character_card_sample.json"),
    )
    args = parser.parse_args()
    write_sample(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
