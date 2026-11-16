from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def generate_textures(
    texture_directory: Path,
    size: int = 64,
) -> dict[str, Path]:
    """Create deterministic base-color, normal, and roughness textures."""
    texture_directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(size, dtype=np.float64)[None, :]
    y = np.arange(size, dtype=np.float64)[:, None]
    u = (x + 0.5) / size
    v = (y + 0.5) / size
    checker = ((np.floor(u * 8.0) + np.floor(v * 8.0)) % 2.0)

    base = np.empty((size, size, 3), dtype=np.float64)
    base[..., 0] = np.clip(0.12 + 0.78 * u + 0.05 * checker, 0.0, 1.0)
    base[..., 1] = np.clip(0.10 + 0.72 * v + 0.08 * (1.0 - checker), 0.0, 1.0)
    base[..., 2] = np.clip(0.30 + 0.35 * checker + 0.15 * np.sin(4.0 * np.pi * u), 0.0, 1.0)

    derivative_u = np.broadcast_to(
        0.55
        * np.cos(2.0 * np.pi * u * 4.0)
        * np.cos(2.0 * np.pi * v * 2.0),
        (size, size),
    )
    derivative_v = np.broadcast_to(
        -0.55
        * np.sin(2.0 * np.pi * u * 4.0)
        * np.sin(2.0 * np.pi * v * 2.0),
        (size, size),
    )
    normal = np.stack(
        (
            derivative_u,
            derivative_v,
            np.ones((size, size), dtype=np.float64) * 0.85,
        ),
        axis=-1,
    )
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    encoded_normal = np.rint((normal * 0.5 + 0.5) * 255.0).astype(np.uint8)

    roughness = np.clip(
        0.18
        + 0.52 * checker
        + 0.12 * np.sin(2.0 * np.pi * (u + v)),
        0.05,
        0.95,
    )
    encoded_roughness = np.rint(roughness * 255.0).astype(np.uint8)

    paths = {
        "base_color": texture_directory / "base_color.png",
        "normal": texture_directory / "normal.png",
        "roughness": texture_directory / "roughness.png",
    }
    Image.fromarray(np.rint(base * 255.0).astype(np.uint8), mode="RGB").save(
        paths["base_color"],
        format="PNG",
    )
    Image.fromarray(encoded_normal, mode="RGB").save(
        paths["normal"],
        format="PNG",
    )
    Image.fromarray(encoded_roughness, mode="L").save(
        paths["roughness"],
        format="PNG",
    )
    return paths


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(component * component for component in vector))
    return [component / length for component in vector]


def _scene_dict(
    texture_paths: dict[str, Path],
    demo_root: Path,
    *,
    width: int,
    height: int,
    texture_filter: str,
    block_size: int,
    block_axis: str,
    block_order: str,
    output_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "camera": {
            "position": [0.0, 0.15, 3.8],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "vertical_fov_degrees": 42.0,
            "near": 0.05,
            "far": 100.0,
        },
        "light": {
            "direction": _normalize([-0.45, 0.7, 0.6]),
            "color": [1.0, 0.96, 0.88],
            "intensity": 2.0,
        },
        "geometry": {
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": 1.0,
        },
        "material": {
            "base_color_texture": str(texture_paths["base_color"].relative_to(demo_root)),
            "normal_texture": str(texture_paths["normal"].relative_to(demo_root)),
            "roughness_texture": str(
                texture_paths["roughness"].relative_to(demo_root)
            ),
            "base_color_tint": [1.0, 1.0, 1.0],
            "specular_strength": 0.35,
            "normal_strength": 0.9,
        },
        "environment": {
            "sky_color": [0.36, 0.46, 0.62],
            "ground_color": [0.10, 0.09, 0.08],
            "intensity": 0.9,
        },
        "output": {
            "width": width,
            "height": height,
            "background_color": [0.015, 0.02, 0.03],
            "texture_filter": texture_filter,
            "block_size": block_size,
            "block_axis": block_axis,
            "block_order": block_order,
            "directory": f"outputs/{output_name}",
            "weights": {
                "base_color": 0.25,
                "direct": 0.55,
                "ambient": 0.2,
            },
        },
    }


def write_demo(
    demo_root: Path,
    *,
    size: int = 128,
    texture_size: int = 64,
) -> list[Path]:
    demo_root = demo_root.resolve()
    texture_paths = generate_textures(demo_root / "textures", texture_size)
    configurations = (
        ("bilinear_32_row_forward", "bilinear", 32, "row", "forward"),
        ("nearest_32_row_forward", "nearest", 32, "row", "forward"),
        ("bilinear_16_column_reverse", "bilinear", 16, "column", "reverse"),
        ("bilinear_64_row_forward", "bilinear", 64, "row", "forward"),
        (f"bilinear_{size}_whole", "bilinear", size, "row", "forward"),
    )
    scene_paths: list[Path] = []
    for name, filter_mode, block_size, axis, order in configurations:
        payload = _scene_dict(
            texture_paths,
            demo_root,
            width=size,
            height=size,
            texture_filter=filter_mode,
            block_size=block_size,
            block_axis=axis,
            block_order=order,
            output_name=name,
        )
        scene_path = demo_root / f"scene_{name}.json"
        scene_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        scene_paths.append(scene_path)
    return scene_paths
