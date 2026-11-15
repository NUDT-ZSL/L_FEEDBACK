"""Deterministic image encoding and metadata serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def encode_height16(heightfield: np.ndarray) -> Image.Image:
    quantized = np.rint(heightfield * 65535.0).clip(0, 65535).astype(np.uint16)
    # Pillow infers the 16-bit unsigned grayscale mode ("I;16") from dtype.
    return Image.fromarray(quantized)


def encode_material16(material: np.ndarray) -> Image.Image:
    return Image.fromarray(material.astype(np.uint16))


def encode_normal_rgb(normal_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(normal_rgb, mode="RGB")


def write_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed compression keeps the emitted bytes reproducible across runs.
    image.save(path, format="PNG", optimize=False, compress_level=6)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
