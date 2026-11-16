"""Deterministic image, report, and atomic file output helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def ppm_bytes(image_uint8: np.ndarray) -> bytes:
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("PPM image must have shape H x W x 3")
    height, width, _ = image_uint8.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + image_uint8.astype(np.uint8, copy=False).tobytes(order="C")


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.maximum(linear, 0.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )


def tonemap_ppm(linear_rgb: np.ndarray, exposure: float) -> bytes:
    exposed = np.maximum(linear_rgb, 0.0) * exposure
    mapped = exposed / (1.0 + exposed)
    encoded = linear_to_srgb(mapped)
    pixels = np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)
    return ppm_bytes(pixels)


def auxiliary_ppm(aux_contribution: np.ndarray) -> bytes:
    """Encode three independent class AOVs as R=direct, G=refract, B=TIR."""

    channels = np.maximum(aux_contribution, 0.0)
    maxima = np.max(channels, axis=(0, 1), keepdims=True)
    maxima = np.maximum(maxima, 1.0e-15)
    normalized = np.power(np.clip(channels / maxima, 0.0, 1.0), 1.0 / 2.2)
    pixels = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    return ppm_bytes(pixels)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def stable_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

