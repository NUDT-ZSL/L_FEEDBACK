"""Scene description parsing, validation and normalization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import errors

BLEND_MODES = ("over", "add", "multiply")
MAX_MATERIALS = 4  # weight map channels (RGBA)
DEFAULT_MAX_PIXELS = 4096 * 4096
DEFAULT_TILE_SIZE = 256


@dataclass(frozen=True)
class Layer:
    index: int
    material_id: str
    tile_scale: float
    rotation_deg: float
    blend_mode: str
    height_scale: float
    opacity: float

    def normalized(self) -> dict:
        return {
            "material_id": self.material_id,
            "tile_scale": self.tile_scale,
            "rotation_deg": self.rotation_deg,
            "blend_mode": self.blend_mode,
            "height_scale": self.height_scale,
            "opacity": self.opacity,
        }


@dataclass(frozen=True)
class Scene:
    seed: int
    width: int
    height: int
    tile_size: int
    normal_strength: float
    materials: tuple
    layers: tuple
    max_pixels: int = DEFAULT_MAX_PIXELS
    extra: dict = field(default_factory=dict)

    def normalized(self) -> dict:
        """Canonical parameter summary used for hashing and the manifest."""
        return {
            "seed": self.seed,
            "resolution": {"width": self.width, "height": self.height},
            "tile_size": self.tile_size,
            "normal_strength": self.normal_strength,
            "materials": list(self.materials),
            "layers": [layer.normalized() for layer in self.layers],
        }

    def params_hash(self) -> str:
        canonical = json.dumps(
            self.normalized(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require(cond: bool, detail: str) -> None:
    if not cond:
        raise errors.scene_invalid(detail)


def _number(value, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{name} must be a number")
    return float(value)


def load_scene(path: str) -> Scene:
    if not os.path.isfile(path):
        raise errors.scene_missing(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise errors.scene_invalid(f"JSON parse error: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise errors.scene_invalid(f"scene file is not valid UTF-8: {exc}")
    _require(isinstance(raw, dict), "top-level value must be a JSON object")

    seed = raw.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool)
             and 0 <= seed < 2**63, "seed must be an integer in [0, 2^63)")

    resolution = raw.get("resolution")
    _require(isinstance(resolution, dict), "resolution must be an object")
    width = resolution.get("width")
    height = resolution.get("height")
    _require(isinstance(width, int) and isinstance(height, int)
             and not isinstance(width, bool) and not isinstance(height, bool),
             "resolution.width/height must be integers")
    _require(8 <= width and 8 <= height,
             "resolution must be at least 8x8")

    max_pixels = raw.get("max_pixels", DEFAULT_MAX_PIXELS)
    _require(isinstance(max_pixels, int) and max_pixels > 0,
             "max_pixels must be a positive integer")
    if width * height > max_pixels:
        raise errors.resolution_limit(width, height, max_pixels)

    tile_size = raw.get("tile_size", DEFAULT_TILE_SIZE)
    _require(isinstance(tile_size, int) and tile_size >= 16,
             "tile_size must be an integer >= 16")

    normal_strength = _number(raw.get("normal_strength", 1.0),
                              "normal_strength")
    _require(normal_strength > 0, "normal_strength must be positive")

    materials = raw.get("materials")
    _require(isinstance(materials, list) and materials
             and all(isinstance(m, str) and m for m in materials),
             "materials must be a non-empty list of material id strings")
    _require(len(materials) <= MAX_MATERIALS,
             f"at most {MAX_MATERIALS} materials (weight map channels)")
    _require(len(set(materials)) == len(materials),
             "material ids must be unique")

    raw_layers = raw.get("layers")
    _require(isinstance(raw_layers, list) and raw_layers,
             "layers must be a non-empty list")

    layers = []
    for index, entry in enumerate(raw_layers):
        _require(isinstance(entry, dict), f"layer {index} must be an object")
        material_id = entry.get("material_id")
        if material_id not in materials:
            raise errors.unknown_material(str(material_id), index)
        blend_mode = entry.get("blend_mode", "over")
        _require(blend_mode in BLEND_MODES,
                 f"layer {index}: blend_mode must be one of {BLEND_MODES}")
        tile_scale = _number(entry.get("tile_scale", 4.0),
                             f"layer {index}: tile_scale")
        _require(tile_scale >= 1.0, f"layer {index}: tile_scale must be >= 1")
        rotation_deg = _number(entry.get("rotation_deg", 0.0),
                               f"layer {index}: rotation_deg")
        height_scale = _number(entry.get("height_scale", 1.0),
                               f"layer {index}: height_scale")
        _require(height_scale >= 0.0,
                 f"layer {index}: height_scale must be >= 0")
        opacity = _number(entry.get("opacity", 1.0),
                          f"layer {index}: opacity")
        _require(0.0 <= opacity <= 1.0,
                 f"layer {index}: opacity must be in [0, 1]")
        layers.append(Layer(
            index=index,
            material_id=material_id,
            tile_scale=tile_scale,
            rotation_deg=rotation_deg,
            blend_mode=blend_mode,
            height_scale=height_scale,
            opacity=opacity,
        ))

    return Scene(
        seed=seed,
        width=width,
        height=height,
        tile_size=tile_size,
        normal_strength=normal_strength,
        materials=tuple(materials),
        layers=tuple(layers),
        max_pixels=max_pixels,
    )
