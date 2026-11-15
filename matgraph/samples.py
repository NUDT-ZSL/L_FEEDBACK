from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .outputs import write_atomic_file
from .pngio import encode_rgb8


def sample_graph() -> Dict[str, Any]:
    return {
        "schema": "material-graph/v1",
        "name": "deterministic-checker-normal-rim",
        "textures": {
            "checker": {
                "path": "textures/checker.png",
                "color_space": "srgb",
                "wrap": "repeat",
                "filter": "bilinear",
            },
            "bumps": {
                "path": "textures/bumps.png",
                "color_space": "linear",
                "wrap": "repeat",
                "filter": "bilinear",
            },
        },
        "nodes": [
            {
                "id": "uv",
                "type": "uv",
            },
            {
                "id": "base_tex",
                "type": "texture_sample",
                "texture": "checker",
                "inputs": {"uv": "uv"},
            },
            {
                "id": "tint",
                "type": "constant",
                "value": [0.92, 0.97, 1.0],
            },
            {
                "id": "tinted_base",
                "type": "multiply",
                "inputs": {"a": "base_tex", "b": "tint"},
            },
            {
                "id": "base_boost",
                "type": "constant",
                "value": [0.01, 0.012, 0.016],
            },
            {
                "id": "base_color",
                "type": "add",
                "inputs": {"a": "tinted_base", "b": "base_boost"},
            },
            {
                "id": "normal_tex",
                "type": "texture_sample",
                "texture": "bumps",
                "inputs": {"uv": "uv"},
            },
            {
                "id": "normal_strength",
                "type": "constant",
                "value": 0.85,
            },
            {
                "id": "world_normal",
                "type": "normal_map",
                "inputs": {
                    "normal": "normal_tex",
                    "strength": "normal_strength",
                },
            },
            {
                "id": "ior",
                "type": "constant",
                "value": 1.45,
            },
            {
                "id": "fresnel",
                "type": "fresnel",
                "inputs": {"normal": "world_normal", "ior": "ior"},
            },
            {
                "id": "black",
                "type": "constant",
                "value": [0.0, 0.0, 0.0],
            },
            {
                "id": "rim_color",
                "type": "constant",
                "value": [0.035, 0.10, 0.18],
            },
            {
                "id": "emissive",
                "type": "mix",
                "inputs": {"a": "black", "b": "rim_color", "factor": "fresnel"},
            },
            {
                "id": "roughness",
                "type": "constant",
                "value": 0.34,
            },
            {
                "id": "metallic",
                "type": "constant",
                "value": 0.08,
            },
        ],
        "outputs": {
            "base_color": "base_color",
            "normal": "world_normal",
            "roughness": "roughness",
            "metallic": "metallic",
            "emissive": "emissive",
        },
    }


def sample_scene() -> Dict[str, Any]:
    return {
        "schema": "scene/v1",
        "name": "sphere-on-plane",
        "camera": {
            "position": [0.0, 1.38, 3.45],
            "target": [0.0, 0.70, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_degrees": 43.0,
        },
        "render": {
            "width": 320,
            "height": 240,
            "exposure": 1.1,
            "background": [0.012, 0.017, 0.028],
            "ambient": {
                "sky": [0.18, 0.23, 0.32],
                "ground": [0.03, 0.025, 0.02],
            },
        },
        "lights": [
            {
                "type": "directional",
                "direction": [0.45, 0.90, 0.35],
                "color": [3.1, 2.8, 2.5],
            }
        ],
        "objects": [
            {
                "type": "sphere",
                "center": [0.0, 0.72, 0.0],
                "radius": 0.72,
            },
            {
                "type": "plane",
                "point": [0.0, 0.0, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "size": [8.0, 8.0],
            },
        ],
    }


def _checker_texture(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    cells = (x * 8 // size + y * 8 // size) % 2
    warm = np.asarray([0.83, 0.42, 0.16], dtype=np.float64)
    cool = np.asarray([0.07, 0.34, 0.41], dtype=np.float64)
    image = np.where(cells[..., None] == 0, warm[None, None, :], cool[None, None, :])
    grout = ((x % 8) == 0) | ((y % 8) == 0)
    image[grout] *= 0.58
    variation = 0.055 * np.sin(2.0 * np.pi * (x + 0.5) / size) * np.cos(
        2.0 * np.pi * (y + 0.5) / size
    )
    image += variation[..., None]
    return np.floor(np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _bump_texture(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    u = (x + 0.5) / size
    v = (y + 0.5) / size
    nx = 0.34 * np.sin(2.0 * np.pi * 4.0 * u) * np.cos(2.0 * np.pi * 3.0 * v)
    ny = 0.34 * np.cos(2.0 * np.pi * 5.0 * u) * np.sin(2.0 * np.pi * 4.0 * v)
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 1e-6))
    normal = np.stack((nx, ny, nz), axis=-1)
    encoded = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    return np.floor(encoded * 255.0 + 0.5).astype(np.uint8)


def _json_bytes(value: Dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def generate_samples(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    textures = output_dir / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    write_atomic_file(textures / "checker.png", encode_rgb8(_checker_texture().astype(np.float64) / 255.0))
    write_atomic_file(textures / "bumps.png", encode_rgb8(_bump_texture().astype(np.float64) / 255.0))
    write_atomic_file(output_dir / "material_graph.json", _json_bytes(sample_graph()))
    write_atomic_file(output_dir / "scene.json", _json_bytes(sample_scene()))
