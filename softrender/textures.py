"""Deterministic texture loading and sampling."""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import BusinessError
from .math3d import normalize_rows, srgb_to_linear


@dataclass(frozen=True)
class Texture:
    rgb: np.ndarray
    role: str
    wrap: str = "repeat"

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    return xx.astype(np.float64) / max(width, 1), yy.astype(np.float64) / max(height, 1)


def _panel_mask(u: np.ndarray, v: np.ndarray, cells: int, line_width: float) -> np.ndarray:
    fu = (u * cells) % 1.0
    fv = (v * cells) % 1.0
    edge = np.minimum(np.minimum(fu, 1.0 - fu), np.minimum(fv, 1.0 - fv))
    return np.clip(edge / max(line_width, 1e-6), 0.0, 1.0)


def _height_to_normal(height: np.ndarray, strength: float) -> np.ndarray:
    du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dv = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nx = -du * strength
    ny = -dv * strength
    nz = np.ones_like(height)
    normals = np.stack([nx, ny, nz], axis=-1)
    normals = normalize_rows(normals.reshape(-1, 3), fallback=np.array([0.0, 0.0, 1.0]))
    return normals.reshape(height.shape + (3,))


def _generated_texture(identifier: str, width: int, height: int, role: str) -> np.ndarray:
    u, v = _grid(width, height)
    cells = 8
    checker = ((np.floor(u * cells) + np.floor(v * cells)) % 2.0).astype(np.float64)
    grid = _panel_mask(u, v, cells, 0.055)
    wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * (u * 5.0 + v * 3.0))
    weave = 0.5 + 0.25 * (np.sin(2.0 * np.pi * u * 24.0) + np.sin(2.0 * np.pi * v * 24.0))

    if identifier == "armor_diffuse":
        base = np.array([0.58, 0.39, 0.22])
        accent = np.array([0.22, 0.42, 0.47])
        rgb = base[None, None, :] * (0.72 + 0.24 * checker[..., None])
        rgb = rgb * (0.58 + 0.42 * grid[..., None]) + accent[None, None, :] * (1.0 - grid[..., None]) * 0.32
        return np.clip(srgb_to_linear(rgb), 0.0, 1.0)
    if identifier == "cloth_diffuse":
        base = np.array([0.12, 0.20, 0.38])
        rgb = base[None, None, :] * (0.72 + 0.28 * weave[..., None])
        stripe = 0.85 + 0.15 * (np.sin(2.0 * np.pi * u * 12.0) > 0.0)
        rgb *= stripe[..., None]
        return np.clip(srgb_to_linear(rgb), 0.0, 1.0)
    if identifier == "skin_diffuse":
        base = np.array([0.68, 0.43, 0.31])
        variation = 0.92 + 0.08 * wave
        return np.clip(srgb_to_linear(base[None, None, :] * variation[..., None]), 0.0, 1.0)
    if identifier == "visor_diffuse":
        base = np.array([0.18, 0.70, 0.82])
        vertical = 0.72 + 0.28 * (0.5 + 0.5 * np.cos(2.0 * np.pi * v))
        return np.clip(srgb_to_linear(base[None, None, :] * vertical[..., None]), 0.0, 1.0)
    if identifier == "platform_diffuse":
        base = np.array([0.19, 0.21, 0.24])
        ring = 0.5 + 0.5 * np.sin(np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 90.0)
        rgb = base[None, None, :] * (0.72 + 0.20 * checker[..., None]) * (0.85 + 0.15 * ring[..., None])
        return np.clip(srgb_to_linear(rgb), 0.0, 1.0)

    if identifier == "armor_normal":
        height_field = 0.42 * grid + 0.25 * checker + 0.12 * wave
        return np.clip(_height_to_normal(height_field, 2.2) * 0.5 + 0.5, 0.0, 1.0)
    if identifier == "cloth_normal":
        height_field = 0.55 * weave + 0.12 * wave
        return np.clip(_height_to_normal(height_field, 1.4) * 0.5 + 0.5, 0.0, 1.0)
    if identifier == "skin_normal":
        pores = 0.5 + 0.15 * np.sin(2.0 * np.pi * u * 42.0) * np.sin(2.0 * np.pi * v * 37.0)
        return np.clip(_height_to_normal(pores, 0.65) * 0.5 + 0.5, 0.0, 1.0)
    if identifier == "visor_normal":
        height_field = 0.5 + 0.5 * np.sin(2.0 * np.pi * (u * 2.0 - v * 2.0))
        return np.clip(_height_to_normal(height_field, 0.3) * 0.5 + 0.5, 0.0, 1.0)
    if identifier == "platform_normal":
        height_field = 0.7 * grid + 0.08 * wave
        return np.clip(_height_to_normal(height_field, 2.8) * 0.5 + 0.5, 0.0, 1.0)

    if identifier == "armor_spec":
        value = 0.42 + 0.48 * grid + 0.10 * checker
        return np.repeat(np.clip(value[..., None], 0.0, 1.0), 3, axis=2)
    if identifier == "cloth_spec":
        value = 0.08 + 0.12 * weave
        return np.repeat(np.clip(value[..., None], 0.0, 1.0), 3, axis=2)
    if identifier == "skin_spec":
        value = 0.22 + 0.10 * wave
        return np.repeat(np.clip(value[..., None], 0.0, 1.0), 3, axis=2)
    if identifier == "visor_spec":
        value = 0.72 + 0.22 * (0.5 + 0.5 * np.cos(2.0 * np.pi * v))
        return np.repeat(np.clip(value[..., None], 0.0, 1.0), 3, axis=2)
    if identifier == "platform_spec":
        value = 0.25 + 0.30 * grid
        return np.repeat(np.clip(value[..., None], 0.0, 1.0), 3, axis=2)

    raise BusinessError("E_TEXTURE", f"unknown generated texture id: {identifier}")


def _procedural_texture(spec: dict[str, Any], role: str) -> np.ndarray:
    width = int(spec.get("width", 64))
    height = int(spec.get("height", width))
    if width < 1 or height < 1 or width > 1024 or height > 1024:
        raise BusinessError("E_VALUE", "procedural texture dimensions must be in [1, 1024]")
    pattern = spec.get("pattern")
    if pattern not in {"checker", "stripes", "waves"}:
        raise BusinessError("E_SCHEMA", "procedural texture pattern must be checker, stripes or waves")
    u, v = _grid(width, height)
    scale = float(spec.get("scale", 8.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise BusinessError("E_VALUE", "texture scale must be finite and positive")
    if pattern == "checker":
        field = (np.floor(u * scale) + np.floor(v * scale)) % 2.0
    elif pattern == "stripes":
        field = 0.5 + 0.5 * np.sin(2.0 * np.pi * u * scale)
    else:
        field = 0.5 + 0.5 * np.sin(2.0 * np.pi * (u * scale + v * 0.5 * scale))

    if role == "normal":
        return np.clip(_height_to_normal(field, 1.5) * 0.5 + 0.5, 0.0, 1.0)

    colors = spec.get("colors")
    if colors is None:
        colors = [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]
    if (
        not isinstance(colors, list)
        or len(colors) != 2
        or any(not isinstance(c, list) or len(c) != 3 for c in colors)
    ):
        raise BusinessError("E_SCHEMA", "texture colors must contain two RGB triples")
    color_a = np.asarray(colors[0], dtype=np.float64)
    color_b = np.asarray(colors[1], dtype=np.float64)
    if not (np.all(np.isfinite(color_a)) and np.all(np.isfinite(color_b))):
        raise BusinessError("E_VALUE", "texture colors must be finite")
    mix = field[..., None]
    rgb = color_a[None, None, :] * (1.0 - mix) + color_b[None, None, :] * mix
    if role == "diffuse" and bool(spec.get("srgb", True)):
        rgb = srgb_to_linear(rgb)
    return np.clip(rgb, 0.0, 1.0)


def _decode_base64_texture(spec: dict[str, Any], role: str, path: str) -> np.ndarray:
    encoded = spec.get("data")
    if not isinstance(encoded, str) or not encoded.strip():
        raise BusinessError("E_SCHEMA", "base64 texture is missing non-empty data", path)
    if encoded.startswith("data:"):
        _, _, encoded = encoded.partition(",")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BusinessError("E_TEXTURE", f"invalid base64 texture data: {exc}", path) from exc
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BusinessError("E_TEXTURE", f"cannot decode texture image: {exc}", path) from exc
    if role == "diffuse" and bool(spec.get("srgb", True)):
        rgb = srgb_to_linear(rgb)
    return np.clip(rgb, 0.0, 1.0)


def load_texture(spec: Any, role: str, path: str = "$") -> Texture:
    if role not in {"diffuse", "normal", "specular"}:
        raise ValueError(f"unknown texture role: {role}")
    if not isinstance(spec, dict):
        raise BusinessError("E_SCHEMA", "texture descriptor must be an object", path)
    kind = spec.get("kind")
    if kind == "generated":
        identifier = spec.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise BusinessError("E_SCHEMA", "generated texture needs a string id", f"{path}.id")
        width = int(spec.get("width", 96))
        height = int(spec.get("height", width))
        if width < 1 or height < 1 or width > 1024 or height > 1024:
            raise BusinessError("E_VALUE", "generated texture dimensions must be in [1, 1024]", path)
        rgb = _generated_texture(identifier, width, height, role)
    elif kind == "procedural":
        rgb = _procedural_texture(spec, role)
    elif kind == "base64":
        rgb = _decode_base64_texture(spec, role, path)
    else:
        raise BusinessError(
            "E_SCHEMA", "texture kind must be generated, procedural or base64", f"{path}.kind"
        )
    wrap = spec.get("wrap", "repeat")
    if wrap not in {"repeat", "clamp"}:
        raise BusinessError("E_VALUE", "texture wrap must be repeat or clamp", f"{path}.wrap")
    return Texture(rgb=rgb, role=role, wrap=wrap)


def sample_texture(texture: Texture, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if texture.wrap == "repeat":
        u = u - np.floor(u)
        v = v - np.floor(v)
    else:
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

    h, w = texture.rgb.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    if texture.wrap == "repeat":
        x0m = x0 % w
        y0m = y0 % h
        x1m = (x0 + 1) % w
        y1m = (y0 + 1) % h
    else:
        x0m = np.clip(x0, 0, w - 1)
        y0m = np.clip(y0, 0, h - 1)
        x1m = np.clip(x0 + 1, 0, w - 1)
        y1m = np.clip(y0 + 1, 0, h - 1)
    fx = fx[..., None]
    fy = fy[..., None]
    c00 = texture.rgb[y0m, x0m]
    c10 = texture.rgb[y0m, x1m]
    c01 = texture.rgb[y1m, x0m]
    c11 = texture.rgb[y1m, x1m]
    return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy
