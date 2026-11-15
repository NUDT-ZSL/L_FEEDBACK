"""Bake orchestration: chunked pipeline, outputs, manifest, path safety."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import PurePath

import numpy as np
from PIL import Image

from . import MANIFEST_GENERATOR, blend, cache, errors, normals
from .scene import Scene

MANIFEST_NAME = "manifest.json"
CACHE_DIR_NAME = ".tbcache"
OUTPUT_FILES = {
    "diffuse": "diffuse.png",
    "normal": "normal.png",
    "weights": "weights.png",
}

BLEND_FORMULA_DOC = {
    "height": "h = base_noise + height_scale * (fbm_rotated - 0.5) * 2",
    "albedo": "a = material_color * (0.7 + 0.6 * fbm_rotated)",
    "alpha": "alpha = clip(opacity * mask_noise, 0, 1)",
    "over": "dst = dst*(1-alpha) + src*alpha; weight += alpha*remaining",
    "add": "dst = dst + src*alpha; weight += alpha",
    "multiply": "dst = dst*(1-alpha+alpha*src); weight += alpha*0.5",
    "weights": "per-material weights normalized per pixel to sum to 1",
    "normals": "n = normalize(-sobel_x*strength, -sobel_y*strength, 8)",
}


def validate_output_path(raw_path: str) -> str:
    """Normalize and validate the output directory path.

    Rejects parent-directory traversal and any symlinked component.
    """
    if not raw_path or not raw_path.strip():
        raise errors.path_unsafe("empty output path")
    pure = PurePath(raw_path)
    if any(part == ".." for part in pure.parts):
        raise errors.path_unsafe(f"path contains '..': {raw_path!r}")
    abs_path = os.path.abspath(raw_path)
    # Walk existing components; reject symlinks anywhere in the chain.
    current = os.path.sep
    for part in PurePath(abs_path).parts[1:]:
        current = os.path.join(current, part)
        if os.path.islink(current):
            raise errors.path_unsafe(
                f"path component is a symlink: {current!r}")
    return abs_path


def check_target_directory(abs_path: str) -> None:
    """Refuse to overwrite directories not created by this tool."""
    if os.path.isfile(abs_path):
        raise errors.target_exists(abs_path)
    if not os.path.isdir(abs_path):
        return
    if not os.listdir(abs_path):
        return
    manifest_path = os.path.join(abs_path, MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if str(marker.get("generator", "")).startswith("terrain-baker/"):
                return
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
    raise errors.target_exists(abs_path)


def _tile_ranges(length: int, tile_size: int):
    start = 0
    while start < length:
        yield start, min(start + tile_size, length)
        start += tile_size


def _wrapped_coords(start: int, stop: int, halo: int, length: int) -> np.ndarray:
    coords = np.arange(start - halo, stop + halo, dtype=np.int64)
    return coords % length


def _accumulate_height_tile(scene: Scene, xs: np.ndarray,
                            ys: np.ndarray) -> np.ndarray:
    """Accumulated height for a haloed coordinate grid (blend pass 1)."""
    grid = np.meshgrid(xs, ys)  # (W+halo*2,) and (H+halo*2,)
    height = np.zeros((ys.size, xs.size), dtype=np.float64)
    for layer in scene.layers:
        fields = blend.generate_layer_fields(scene, layer, grid[0], grid[1])
        alpha = fields["alpha"]
        if layer.blend_mode == "over":
            height = height * (1.0 - alpha) + fields["height"] * alpha
        elif layer.blend_mode == "add":
            height = height + fields["height"] * alpha
        else:  # multiply
            height = height * (1.0 - alpha + alpha * fields["height_mul"])
    return height


def _accumulate_surface_tile(scene: Scene, xs: np.ndarray, ys: np.ndarray,
                             layer_stats: list) -> tuple:
    """Albedo + weights for a coordinate grid (blend pass 2)."""
    grid = np.meshgrid(xs, ys)
    accum = blend.new_accumulator((ys.size, xs.size), scene.materials)
    for layer in scene.layers:
        fields = blend.generate_layer_fields(scene, layer, grid[0], grid[1])
        blend.blend_layer(accum, fields, layer)
        stats = layer_stats[layer.index]
        stats["min"] = min(stats["min"], float(fields["height"].min()))
        stats["max"] = max(stats["max"], float(fields["height"].max()))
        stats["albedo_sum"] += fields["albedo"].sum(axis=(0, 1))
        stats["pixels"] += fields["height"].size
    return accum["albedo"], accum["weights"]


def bake_arrays(scene: Scene) -> dict:
    """Run the full chunked bake; returns all output arrays and stats."""
    width, height = scene.width, scene.height
    tile = scene.tile_size
    halo = normals.HALO

    height_full = np.zeros((height, width), dtype=np.float64)
    normal_full = np.zeros((height, width, 3), dtype=np.uint8)
    diffuse_full = np.zeros((height, width, 3), dtype=np.uint8)
    weights_full = np.zeros((height, width, len(scene.materials)),
                            dtype=np.float64)
    layer_stats = [
        {"min": math.inf, "max": -math.inf,
         "albedo_sum": np.zeros(3, dtype=np.float64), "pixels": 0}
        for _ in scene.layers
    ]

    for y0, y1 in _tile_ranges(height, tile):
        for x0, x1 in _tile_ranges(width, tile):
            # Pass 1: height with wrap-around halo, then normals.
            hxs = _wrapped_coords(x0, x1, halo, width)
            hys = _wrapped_coords(y0, y1, halo, height)
            haloed = _accumulate_height_tile(scene, hxs, hys)
            tile_normals = normals.normals_from_height(
                haloed, scene.normal_strength)
            normal_full[y0:y1, x0:x1] = normals.encode_normals(tile_normals)
            height_full[y0:y1, x0:x1] = haloed[halo:-halo, halo:-halo]

            # Pass 2: albedo + weights (no halo required).
            xs = np.arange(x0, x1, dtype=np.int64)
            ys = np.arange(y0, y1, dtype=np.int64)
            albedo, weights = _accumulate_surface_tile(
                scene, xs, ys, layer_stats)
            diffuse_full[y0:y1, x0:x1] = np.clip(
                np.round(albedo * 255.0), 0, 255).astype(np.uint8)
            weights_full[y0:y1, x0:x1] = blend.normalized_weights(
                {"weights": weights}, scene.materials)

    weights_u8 = np.clip(np.round(weights_full * 255.0), 0, 255)
    # Pad to RGBA so every material id maps to a fixed channel.
    padded = np.zeros((height, width, 4), dtype=np.uint8)
    padded[:, :, :weights_u8.shape[2]] = weights_u8.astype(np.uint8)

    return {
        "height": height_full.astype(np.float32),
        "normal": normal_full,
        "diffuse": diffuse_full,
        "weights": padded,
        "layer_stats": layer_stats,
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_image_atomic(path: str, array: np.ndarray,
                        written: list) -> None:
    tmp_path = path + ".tmp"
    Image.fromarray(array).save(tmp_path, format="PNG")
    os.replace(tmp_path, path)
    written.append(path)


def run_bake(scene: Scene, output_dir: str) -> dict:
    """Full bake entry point. Returns the manifest dict."""
    abs_dir = validate_output_path(output_dir)
    check_target_directory(abs_dir)
    os.makedirs(abs_dir, exist_ok=True)
    cache_dir = os.path.join(abs_dir, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)

    params_hash = scene.params_hash()
    written: list = []
    try:
        # Try the intermediate cache first; rebuild on any invalidity.
        height_cache = os.path.join(cache_dir, "height.tbc")
        normal_cache = os.path.join(cache_dir, "normals.tbc")
        try:
            height = cache.read_cache(height_cache, cache.KIND_HEIGHT,
                                      params_hash)
            normal = cache.read_cache(normal_cache, cache.KIND_NORMALS,
                                      params_hash)
            result = None
        except cache.CacheMiss:
            cache.clear_cache_dir(cache_dir)
            result = bake_arrays(scene)
            height, normal = result["height"], result["normal"]
            cache.write_cache(height_cache, cache.KIND_HEIGHT, height,
                              params_hash)
            cache.write_cache(normal_cache, cache.KIND_NORMALS, normal,
                              params_hash)

        if result is None:
            # Cache hit: still need diffuse/weights, recompute pass 2 only.
            result = _surface_only(scene)
            result["height"] = height
            result["normal"] = normal

        correction = normals.normalization_correction(result["normal"])

        outputs = {}
        _write_image_atomic(os.path.join(abs_dir, OUTPUT_FILES["diffuse"]),
                            result["diffuse"], written)
        _write_image_atomic(os.path.join(abs_dir, OUTPUT_FILES["normal"]),
                            result["normal"], written)
        _write_image_atomic(os.path.join(abs_dir, OUTPUT_FILES["weights"]),
                            result["weights"], written)
        for key, filename in OUTPUT_FILES.items():
            outputs[key] = {
                "file": filename,
                "sha256": _sha256_file(os.path.join(abs_dir, filename)),
            }

        manifest = {
            "generator": MANIFEST_GENERATOR,
            "params_hash": params_hash,
            "params": scene.normalized(),
            "blend_formulas": BLEND_FORMULA_DOC,
            "layers": _layer_manifest(scene, result["layer_stats"]),
            "normal_correction_factor": round(correction, 9),
            "outputs": outputs,
            "cache": {"format": cache.MAGIC.decode("ascii"),
                      "version": cache.VERSION},
        }
        manifest_path = os.path.join(abs_dir, MANIFEST_NAME)
        tmp_manifest = manifest_path + ".tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_manifest, manifest_path)
        written.append(manifest_path)
        return manifest
    except BaseException:
        # Leave no partial outputs behind.
        for path in written:
            try:
                os.remove(path)
            except OSError:
                pass
        for candidate in list(OUTPUT_FILES.values()) + [MANIFEST_NAME]:
            tmp = os.path.join(abs_dir, candidate + ".tmp")
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        raise


def _surface_only(scene: Scene) -> dict:
    """Recompute diffuse/weights (and stats) without height/normals."""
    width, height = scene.width, scene.height
    diffuse_full = np.zeros((height, width, 3), dtype=np.uint8)
    weights_full = np.zeros((height, width, len(scene.materials)),
                            dtype=np.float64)
    layer_stats = [
        {"min": math.inf, "max": -math.inf,
         "albedo_sum": np.zeros(3, dtype=np.float64), "pixels": 0}
        for _ in scene.layers
    ]
    for y0, y1 in _tile_ranges(height, scene.tile_size):
        for x0, x1 in _tile_ranges(width, scene.tile_size):
            xs = np.arange(x0, x1, dtype=np.int64)
            ys = np.arange(y0, y1, dtype=np.int64)
            albedo, weights = _accumulate_surface_tile(
                scene, xs, ys, layer_stats)
            diffuse_full[y0:y1, x0:x1] = np.clip(
                np.round(albedo * 255.0), 0, 255).astype(np.uint8)
            weights_full[y0:y1, x0:x1] = blend.normalized_weights(
                {"weights": weights}, scene.materials)
    weights_u8 = np.clip(np.round(weights_full * 255.0), 0, 255)
    padded = np.zeros((height, width, 4), dtype=np.uint8)
    padded[:, :, :weights_u8.shape[2]] = weights_u8.astype(np.uint8)
    return {"diffuse": diffuse_full, "weights": padded,
            "layer_stats": layer_stats}


def _layer_manifest(scene: Scene, layer_stats: list) -> list:
    entries = []
    for layer, stats in zip(scene.layers, layer_stats):
        mean_albedo = stats["albedo_sum"] / max(1, stats["pixels"])
        entries.append({
            "index": layer.index,
            "material_id": layer.material_id,
            "blend_mode": layer.blend_mode,
            "min_height": round(stats["min"], 6),
            "max_height": round(stats["max"], 6),
            "mean_albedo": [round(float(c), 6) for c in mean_albedo],
        })
    return entries
