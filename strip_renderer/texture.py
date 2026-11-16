from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import RenderError, asset_error
from .scene import Material


MIN_TEXTURE_SIZE = 2
MAX_TEXTURE_SIZE = 4096


@dataclass(frozen=True)
class TextureSet:
    base_color: np.ndarray
    roughness: np.ndarray
    normal: np.ndarray | None
    base_color_size: tuple[int, int]
    roughness_size: tuple[int, int]
    normal_size: tuple[int, int] | None
    normal_texture_missing: bool


def _read_image(path: Path, label: str) -> Image.Image:
    if not path.exists():
        raise asset_error(
            "E_TEXTURE_READ",
            f"{label} texture does not exist: {path}",
            path=str(path),
        )
    if not path.is_file():
        raise asset_error(
            "E_TEXTURE_READ",
            f"{label} texture path is not a file: {path}",
            path=str(path),
        )
    try:
        with Image.open(path) as image:
            width, height = image.size
            if not (
                MIN_TEXTURE_SIZE <= width <= MAX_TEXTURE_SIZE
                and MIN_TEXTURE_SIZE <= height <= MAX_TEXTURE_SIZE
            ):
                raise asset_error(
                    "E_TEXTURE_SIZE",
                    (
                        f"{label} texture size {width}x{height} is outside "
                        f"{MIN_TEXTURE_SIZE}..{MAX_TEXTURE_SIZE}"
                    ),
                    path=str(path),
                    width=width,
                    height=height,
                )
            image.load()
            return image.copy()
    except RenderError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise asset_error(
            "E_TEXTURE_READ",
            f"cannot read {label} texture {path}: {exc}",
            path=str(path),
        ) from exc


def load_textures(material: Material) -> TextureSet:
    base_image = _read_image(material.base_color_texture, "base color")
    roughness_image = _read_image(material.roughness_texture, "roughness")

    base_color = np.asarray(base_image.convert("RGB"), dtype=np.float64) / 255.0
    roughness = np.asarray(roughness_image.convert("RGB"), dtype=np.float64)[
        :, :, 0
    ] / 255.0

    normal: np.ndarray | None = None
    normal_size: tuple[int, int] | None = None
    normal_missing = material.normal_texture is None
    if material.normal_texture is not None:
        if not material.normal_texture.exists():
            normal_missing = True
        else:
            normal_image = _read_image(material.normal_texture, "normal")
            normal = (
                np.asarray(normal_image.convert("RGB"), dtype=np.float64) / 255.0
            )
            normal_size = normal.shape[1], normal.shape[0]

    return TextureSet(
        base_color=np.ascontiguousarray(base_color),
        roughness=np.ascontiguousarray(roughness),
        normal=None if normal is None else np.ascontiguousarray(normal),
        base_color_size=(base_color.shape[1], base_color.shape[0]),
        roughness_size=(roughness.shape[1], roughness.shape[0]),
        normal_size=normal_size,
        normal_texture_missing=normal_missing,
    )
