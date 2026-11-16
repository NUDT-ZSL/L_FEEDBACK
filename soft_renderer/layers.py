from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .errors import OutputError
from .imageio import AtomicOutputSet, downsample, read_pfm, write_pfm
from .renderer import LayerBuffer, RenderResult, compose_layers


MANIFEST_NAME = "layers.json"
FORMAT_NAME = "offline-soft-renderer-layers"


@dataclass
class LoadedLayers:
    layers: Dict[int, LayerBuffer]
    background: np.ndarray
    supersample: int
    output_width: int
    output_height: int
    render_width: int
    render_height: int
    block_size: int
    no_tiling: bool
    preset: str
    manifest: Dict[str, object]


def layer_file_names(material_id: int) -> Tuple[str, str]:
    return (
        "material_%d.pfm" % material_id,
        "material_%d_depth.pfm" % material_id,
    )


def build_manifest(result: RenderResult, scene_material_names: Dict[int, str]) -> Dict[str, object]:
    entries: List[Dict[str, object]] = []
    for material_id in sorted(result.layers):
        color_name, depth_name = layer_file_names(material_id)
        entries.append(
            {
                "material_id": material_id,
                "name": scene_material_names.get(material_id, "material_%d" % material_id),
                "color": color_name,
                "depth": depth_name,
            }
        )
    return {
        "format": FORMAT_NAME,
        "version": 1,
        "background_linear": [float(value) for value in result.background],
        "preset": result.preset.name,
        "block_size": result.block_size,
        "no_tiling": result.no_tiling,
        "supersample": result.preset.supersample,
        "pixel_result_sha256": result.pixel_hash,
        "output": {
            "width": result.width,
            "height": result.height,
        },
        "render": {
            "width": result.render_width,
            "height": result.render_height,
        },
        "layers": entries,
    }


def add_layers_to_output_set(
    output_set: AtomicOutputSet,
    result: RenderResult,
    layers_directory: str,
    material_names: Dict[int, str],
) -> None:
    for material_id in sorted(result.layers):
        layer = result.layers[material_id]
        color_name, depth_name = layer_file_names(material_id)
        color_path = os.path.join(layers_directory, color_name)
        depth_path = os.path.join(layers_directory, depth_name)
        output_set.add(
            color_path,
            lambda path, color=layer.color: write_pfm(path, color),
        )
        output_set.add(
            depth_path,
            lambda path, depth=layer.depth: write_pfm(path, depth),
        )
    manifest_path = os.path.join(layers_directory, MANIFEST_NAME)
    manifest = build_manifest(result, material_names)
    output_set.add(
        manifest_path,
        lambda path, data=manifest: _write_json(path, data),
    )


def _write_json(path: str, value: Dict[str, object]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_layers(directory: str) -> LoadedLayers:
    manifest_path = os.path.join(directory, MANIFEST_NAME)
    try:
        with open(manifest_path, "rb") as handle:
            manifest = json.loads(handle.read().decode("utf-8"))
    except OSError as error:
        raise OutputError("cannot read layer manifest %r: %s" % (manifest_path, error))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OutputError("invalid layer manifest %r: %s" % (manifest_path, error))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_NAME:
        raise OutputError("layer manifest has an unsupported format")
    if manifest.get("version") != 1:
        raise OutputError("layer manifest has an unsupported version")
    try:
        supersample = int(manifest["supersample"])
        output = manifest["output"]
        render = manifest["render"]
        output_width = int(output["width"])
        output_height = int(output["height"])
        render_width = int(render["width"])
        render_height = int(render["height"])
        background = np.asarray(manifest.get("background_linear", [0.0, 0.0, 0.0]), dtype=np.float64)
        entries = manifest["layers"]
        block_size = int(manifest.get("block_size", 1))
        no_tiling = bool(manifest.get("no_tiling", False))
        preset = str(manifest.get("preset", "unknown"))
    except (KeyError, TypeError, ValueError) as error:
        raise OutputError("invalid layer manifest fields: %s" % error)
    if supersample < 1 or output_width < 1 or output_height < 1:
        raise OutputError("invalid layer dimensions in manifest")
    if render_width != output_width * supersample or render_height != output_height * supersample:
        raise OutputError("manifest output/render dimensions are inconsistent")
    if background.shape != (3,) or not np.all(np.isfinite(background)):
        raise OutputError("manifest background_linear must contain three finite values")
    if not isinstance(entries, list) or not entries:
        raise OutputError("layer manifest contains no layers")

    layers: Dict[int, LayerBuffer] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise OutputError("layer manifest entry %d is invalid" % index)
        try:
            material_id = int(entry["material_id"])
            color_path = os.path.join(directory, str(entry["color"]))
            depth_path = os.path.join(directory, str(entry["depth"]))
        except (KeyError, TypeError, ValueError):
            raise OutputError("layer manifest entry %d has invalid fields" % index)
        if material_id in layers:
            raise OutputError("duplicate material layer ID %d" % material_id)
        color = read_pfm(color_path)
        depth = read_pfm(depth_path)
        if color.shape != (render_height, render_width, 3):
            raise OutputError(
                "layer %d color shape %r does not match %dx%d"
                % (material_id, color.shape, render_width, render_height)
            )
        if depth.shape != (render_height, render_width):
            raise OutputError(
                "layer %d depth shape %r does not match %dx%d"
                % (material_id, depth.shape, render_width, render_height)
            )
        layers[material_id] = LayerBuffer(material_id, color, depth)
    return LoadedLayers(
        layers=layers,
        background=background,
        supersample=supersample,
        output_width=output_width,
        output_height=output_height,
        render_width=render_width,
        render_height=render_height,
        block_size=block_size,
        no_tiling=no_tiling,
        preset=preset,
        manifest=manifest,
    )


def composite_layers_from_directory(directory: str) -> Tuple[np.ndarray, LoadedLayers, Dict[str, float]]:
    started = time.perf_counter()
    loaded = load_layers(directory)
    loaded_ms = (time.perf_counter() - started) * 1000.0
    compose_started = time.perf_counter()
    layered_frame = compose_layers(loaded.layers, loaded.background)
    compose_ms = (time.perf_counter() - compose_started) * 1000.0
    resolve_started = time.perf_counter()
    frame = downsample(layered_frame, loaded.supersample)
    resolve_ms = (time.perf_counter() - resolve_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    return frame, loaded, {
        "load_layers": loaded_ms,
        "layer_composite": compose_ms,
        "resolve_supersample": resolve_ms,
        "total": total_ms,
    }
