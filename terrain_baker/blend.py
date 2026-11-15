"""Layer field generation and multi-layer blending.

Blend formulas (also recorded in the manifest):

    per-layer height   h = base + height_scale * (fbm - 0.5) * 2
    multiply height    h_m = 1 + height_scale * (fbm - 0.5) * 2
    per-layer albedo   a = material_color * (0.7 + 0.6 * fbm)
    alpha              alpha = clip(opacity * mask_noise, 0, 1)

    over:      H <- H * (1 - alpha) + h * alpha
               A <- A * (1 - alpha) + a * alpha
    add:       H <- H + h * alpha
               A <- A + a * alpha
    multiply:  H <- H * (1 - alpha + alpha * h_m)
               A <- A * (1 - alpha + alpha * a)

    weight contribution per layer:
        over:      alpha * remaining   (remaining *= 1 - alpha)
        add:       alpha
        multiply:  alpha * 0.5
    Weights are accumulated per material id and normalized per pixel.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import noise
from .scene import Layer, Scene

# Seed domain separators so each noise channel is independent.
_SEED_BASE = 0xB45E
_SEED_PERTURB = 0x9E37
_SEED_MASK = 0xA55A
_SEED_ALBEDO = 0xA1BE


def material_color(material_id: str) -> np.ndarray:
    """Deterministic base albedo color for a material id, in [0.35, 0.95]."""
    digest = hashlib.sha256(material_id.encode("utf-8")).digest()
    rgb = np.array([digest[0], digest[1], digest[2]], dtype=np.float64) / 255.0
    return 0.35 + 0.6 * rgb


def layer_seed(scene_seed: int, layer_index: int, domain: int) -> int:
    return (scene_seed ^ (domain * 0x100001B3) ^ (layer_index * 0x9E3779B1)) \
        & 0x7FFFFFFFFFFFFFFF


def generate_layer_fields(
    scene: Scene,
    layer: Layer,
    xs: np.ndarray,
    ys: np.ndarray,
) -> dict:
    """Generate height/albedo/alpha fields for one layer on a coord grid.

    ``xs``/``ys`` are integer pixel coordinates (already wrapped into the
    image domain by the caller for halo padding).
    """
    width, height = scene.width, scene.height
    period = max(1, int(round(layer.tile_scale)))

    # Base low-frequency height (no rotation), period 2.
    base_u = xs.astype(np.float64) / float(width) * 2.0
    base_v = ys.astype(np.float64) / float(height) * 2.0
    base = noise.value_noise(
        layer_seed(scene.seed, layer.index, _SEED_BASE), base_u, base_v, 2)

    # Rotated, tiled perturbation noise.
    ru, rv = noise.rotated_uv(xs, ys, width, height, layer.rotation_deg)
    perturb = noise.fbm(
        layer_seed(scene.seed, layer.index, _SEED_PERTURB),
        ru * period, rv * period, period)

    # Coverage mask noise (drives alpha variation across the terrain).
    mask = noise.value_noise(
        layer_seed(scene.seed, layer.index, _SEED_MASK),
        ru * max(1, period // 2), rv * max(1, period // 2),
        max(1, period // 2))
    mask = np.clip((mask - 0.35) / 0.3, 0.0, 1.0)

    height_field = base + layer.height_scale * (perturb - 0.5) * 2.0
    height_mul = 1.0 + layer.height_scale * (perturb - 0.5) * 2.0
    color = material_color(layer.material_id)
    shade = 0.7 + 0.6 * perturb
    albedo = color[None, None, :] * shade[:, :, None]
    alpha = np.clip(layer.opacity * mask, 0.0, 1.0)

    return {
        "height": height_field,
        "height_mul": height_mul,
        "albedo": albedo,
        "alpha": alpha,
    }


def blend_layer(accum: dict, fields: dict, layer: Layer) -> None:
    """Accumulate one layer into height/albedo/weight state, in place."""
    alpha = fields["alpha"]
    alpha3 = alpha[:, :, None]
    mode = layer.blend_mode
    if mode == "over":
        inv = 1.0 - alpha
        accum["height"] *= inv
        accum["height"] += fields["height"] * alpha
        accum["albedo"] *= (1.0 - alpha3)
        accum["albedo"] += fields["albedo"] * alpha3
        contribution = alpha * accum["remaining"]
        accum["remaining"] *= inv
    elif mode == "add":
        accum["height"] += fields["height"] * alpha
        accum["albedo"] += fields["albedo"] * alpha3
        contribution = alpha
    elif mode == "multiply":
        accum["height"] *= (1.0 - alpha + alpha * fields["height_mul"])
        accum["albedo"] *= (1.0 - alpha3 + alpha3 * fields["albedo"])
        contribution = alpha * 0.5
    else:  # pragma: no cover - validated at scene load
        raise ValueError(f"unknown blend mode {mode!r}")
    accum["weights"][layer.material_id] += contribution


def new_accumulator(shape: tuple, materials: tuple) -> dict:
    return {
        "height": np.zeros(shape, dtype=np.float64),
        "albedo": np.zeros(shape + (3,), dtype=np.float64),
        "remaining": np.ones(shape, dtype=np.float64),
        "weights": {m: np.zeros(shape, dtype=np.float64) for m in materials},
    }


def normalized_weights(accum: dict, materials: tuple) -> np.ndarray:
    """Stack per-material weights and normalize each pixel to sum to 1."""
    stack = np.stack([accum["weights"][m] for m in materials], axis=-1)
    total = stack.sum(axis=-1, keepdims=True)
    uniform = 1.0 / float(len(materials))
    safe = np.where(total > 1e-12, stack / np.maximum(total, 1e-12), uniform)
    return safe
