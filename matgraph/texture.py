from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from PIL import Image

from .errors import graph_error
from .limits import Limits
from .mathutil import srgb_to_linear


@dataclass(frozen=True)
class TextureAsset:
    asset_id: str
    relative_path: str
    width: int
    height: int
    color_space: str
    wrap: str
    filter_mode: str
    data: np.ndarray

    def metadata(self) -> Dict[str, Any]:
        return {
            "path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "color_space": self.color_space,
            "wrap": self.wrap,
            "filter": self.filter_mode,
        }


def _fail(
    node_index: int,
    node_id: str,
    field: str,
    message: str,
    code: str = "E_TEXTURE",
) -> None:
    raise graph_error(
        code,
        message,
        node_index=node_index,
        node_id=node_id,
        field=field,
    )


def load_texture(
    asset_id: str,
    spec: Mapping[str, Any],
    base_dir: Path,
    limits: Limits,
    *,
    node_index: int,
    node_id: str,
) -> TextureAsset:
    path_value = spec.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        _fail(node_index, node_id, "texture.path", "texture path must be a non-empty string")

    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(
            node_index,
            node_id,
            "texture.path",
            "texture path must be relative to the material graph directory and may not contain '..'",
            "E_TEXTURE_PATH",
        )

    base_dir = base_dir.resolve()
    resolved = (base_dir / relative).resolve()
    if resolved != base_dir and base_dir not in resolved.parents:
        _fail(
            node_index,
            node_id,
            "texture.path",
            "texture path escapes the material graph directory",
            "E_TEXTURE_PATH",
        )
    if resolved.suffix.lower() != ".png":
        _fail(
            node_index,
            node_id,
            "texture.path",
            "only PNG textures are supported",
            "E_TEXTURE_FORMAT",
        )
    if not resolved.is_file():
        _fail(
            node_index,
            node_id,
            "texture.path",
            f"texture file does not exist: {relative.as_posix()}",
            "E_TEXTURE_IO",
        )

    size = resolved.stat().st_size
    if size > limits.max_texture_bytes:
        _fail(
            node_index,
            node_id,
            "texture.path",
            f"texture exceeds {limits.max_texture_bytes} bytes",
            "E_LIMIT",
        )

    color_space = spec.get("color_space", "linear")
    if color_space not in ("linear", "srgb"):
        _fail(
            node_index,
            node_id,
            "texture.color_space",
            "color_space must be 'linear' or 'srgb'",
            "E_TEXTURE_PARAM",
        )
    wrap = spec.get("wrap", "repeat")
    if wrap not in ("repeat", "clamp"):
        _fail(
            node_index,
            node_id,
            "texture.wrap",
            "wrap must be 'repeat' or 'clamp'",
            "E_TEXTURE_PARAM",
        )
    filter_mode = spec.get("filter", "bilinear")
    if filter_mode not in ("bilinear", "nearest"):
        _fail(
            node_index,
            node_id,
            "texture.filter",
            "filter must be 'bilinear' or 'nearest'",
            "E_TEXTURE_PARAM",
        )

    try:
        with Image.open(resolved) as image:
            if image.format != "PNG":
                _fail(
                    node_index,
                    node_id,
                    "texture.path",
                    "only PNG textures are supported",
                    "E_TEXTURE_FORMAT",
                )
            width, height = image.size
            if (
                width < 1
                or height < 1
                or width > limits.max_texture_dimension
                or height > limits.max_texture_dimension
            ):
                _fail(
                    node_index,
                    node_id,
                    "texture.path",
                    (
                        "texture dimensions must be within "
                        f"1..{limits.max_texture_dimension}"
                    ),
                    "E_LIMIT",
                )
            rgb = image.convert("RGB")
            data = np.asarray(rgb, dtype=np.float64) / 255.0
    except Exception as exc:
        if getattr(exc, "code", None) is not None:
            raise
        _fail(
            node_index,
            node_id,
            "texture.path",
            f"unable to decode PNG texture: {type(exc).__name__}",
            "E_TEXTURE_IO",
        )

    if color_space == "srgb":
        data = srgb_to_linear(data)
    return TextureAsset(
        asset_id=asset_id,
        relative_path=relative.as_posix(),
        width=width,
        height=height,
        color_space=color_space,
        wrap=wrap,
        filter_mode=filter_mode,
        data=np.ascontiguousarray(data, dtype=np.float64),
    )


def _indices(coord: np.ndarray, size: int, wrap: str) -> np.ndarray:
    indices = np.floor(coord).astype(np.int64)
    if wrap == "repeat":
        return np.mod(indices, size)
    return np.clip(indices, 0, size - 1)


def sample_texture(asset: TextureAsset, uv: np.ndarray) -> np.ndarray:
    """Sample a texture using the same conventions for compile-time and runtime."""
    points = np.asarray(uv, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError("texture UV must have a trailing dimension of 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("texture UV must be finite")

    width = asset.width
    height = asset.height
    x = points[..., 0] * width - 0.5
    y = points[..., 1] * height - 0.5

    if asset.filter_mode == "nearest":
        x_index = _indices(np.floor(x + 0.5), width, asset.wrap)
        y_index = _indices(np.floor(y + 0.5), height, asset.wrap)
        return asset.data[y_index, x_index]

    x0 = _indices(x, width, asset.wrap)
    y0 = _indices(y, height, asset.wrap)
    x1 = _indices(np.floor(x) + 1, width, asset.wrap)
    y1 = _indices(np.floor(y) + 1, height, asset.wrap)
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    c00 = asset.data[y0, x0]
    c10 = asset.data[y0, x1]
    c01 = asset.data[y1, x0]
    c11 = asset.data[y1, x1]
    bottom = c00 * (1.0 - fx)[..., None] + c10 * fx[..., None]
    top = c01 * (1.0 - fx)[..., None] + c11 * fx[..., None]
    return bottom * (1.0 - fy)[..., None] + top * fy[..., None]
