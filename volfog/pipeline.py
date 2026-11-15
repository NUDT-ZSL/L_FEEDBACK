"""Tone mapping, sRGB transfer and file output orchestration."""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from .config import RenderConfig
from .density import generate_density_field, load_density_field
from .renderer import render_frame


def extended_reinhard(color_linear: np.ndarray) -> np.ndarray:
    """Extended Reinhard with white point set to the frame's max luminance."""
    luminance = np.maximum(
        color_linear[..., 0] * 0.2126
        + color_linear[..., 1] * 0.7152
        + color_linear[..., 2] * 0.0722,
        0.0,
    )
    white_sq = float(luminance.max()) ** 2
    if white_sq < 1e-12:
        return np.zeros_like(color_linear)
    scale = 1.0 + luminance / white_sq
    mapped = color_linear * scale[..., None] / (1.0 + luminance[..., None])
    return np.clip(mapped, 0.0, 1.0)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, 12.92 * values,
                    1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def to_uint8(values: np.ndarray) -> np.ndarray:
    return np.rint(values * 255.0).astype(np.uint8)


def save_outputs(
    config: RenderConfig,
    color_linear: np.ndarray,
    transmittance: np.ndarray,
    depth: np.ndarray,
) -> dict:
    """Write color/transmittance (8-bit PNG), depth (16-bit PNG) and raw .npy."""
    os.makedirs(config.output_dir, exist_ok=True)
    tag = f"{config.output_prefix}_{config.mode}"

    color_srgb = to_uint8(linear_to_srgb(extended_reinhard(color_linear)))
    color_path = os.path.join(config.output_dir, f"{tag}_color.png")
    Image.fromarray(color_srgb, mode="RGB").save(color_path)

    trans_path = os.path.join(config.output_dir, f"{tag}_transmittance.png")
    Image.fromarray(to_uint8(np.clip(transmittance, 0.0, 1.0)), mode="L").save(trans_path)

    # Depth is normalized against the renderer's configured maximum ray
    # distance (step_size * max_samples), the same sentinel used for misses.
    normalized_depth = np.clip(depth / config.max_distance, 0.0, 1.0)
    depth16 = np.rint(normalized_depth * 65535.0).astype(np.uint16)
    depth_path = os.path.join(config.output_dir, f"{tag}_depth.png")
    Image.fromarray(depth16).save(depth_path)

    raw_paths = {}
    for name, array in (
        ("color_linear", color_linear),
        ("transmittance", transmittance),
        ("depth", depth),
    ):
        path = os.path.join(config.output_dir, f"{tag}_{name}.npy")
        np.save(path, array)
        raw_paths[name] = path

    meta = {
        "mode": config.mode,
        "max_distance": config.max_distance,
        "depth_normalization": "depth / (step_size * max_samples)",
        "parallel": config.parallel,
        "files": {
            "color_png": color_path,
            "transmittance_png": trans_path,
            "depth_png": depth_path,
            **raw_paths,
        },
    }
    meta_path = os.path.join(config.output_dir, f"{tag}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    meta["files"]["meta"] = meta_path
    return meta


def render_to_files(config: RenderConfig) -> dict:
    config.validate()
    if config.density_file:
        density = load_density_field(config.density_file)
    else:
        density = generate_density_field(config.voxel_res, config.seed)
    color, transmittance, depth = render_frame(config, density)
    return save_outputs(config, color, transmittance, depth)
