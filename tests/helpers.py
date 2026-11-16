from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def base_config() -> dict[str, Any]:
    with (ROOT / "sample_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["terrain"].update(
        {
            "rows": 40,
            "cols": 48,
            "world_width": 100.0,
            "world_depth": 80.0,
            "height_scale": 25.0,
            "tile_cells": 8,
            "noise": {
                "frequency": 3.0,
                "octaves": 5,
                "persistence": 0.52,
                "lacunarity": 2.0,
            },
        }
    )
    config["camera"].update(
        {
            "azimuth_deg": 35.0,
            "elevation_deg": 30.0,
            "distance": 160.0,
            "fov_y_deg": 50.0,
            "near": 0.1,
            "far": 600.0,
            "target": [0.5, 0.3, 0.5],
        }
    )
    config["materials"].update(
        {
            "elevation_thresholds": [0.20, 0.45, 0.70],
            "elevation_transition": 0.09,
            "slope_start": 0.14,
            "slope_end": 0.40,
            "slope_rock_strength": 0.85,
        }
    )
    config["light"]["shadow"].update(
        {
            "enabled": True,
            "resolution": 128,
            "depth_bias": 0.003,
            "normal_offset": 0.08,
            "pcf_samples": 8,
            "filter_radius": 1.5,
        }
    )
    config["render"].update(
        {
            "width": 192,
            "height": 144,
            "tile_size": 32,
            "write_debug_images": True,
            "exposure": 1.2,
        }
    )
    config["output"] = {
        "image": "out/preview.png",
        "normal_image": "out/normal.png",
        "shadow_image": "out/shadow.png",
        "terrain_file": "out/terrain.bin",
        "tile_report": "out/tiles.json",
        "cache_dir": "out/cache",
    }
    config["limits"] = {
        "max_memory_mb": 64.0,
        "max_total_samples": 20_000_000,
        "max_terrain_cells": 500_000,
    }
    return config


def write_config(directory: Path, config: dict[str, Any], name: str = "config.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def small_config() -> dict[str, Any]:
    return copy.deepcopy(base_config())

