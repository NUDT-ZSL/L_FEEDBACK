"""PNG encoding with fixed 8-bit quantization and atomic writes."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import EXIT_OUTPUT_WRITE, OUTPUT_WRITE, BusinessError
from .util import pretty_json_bytes


def rgba16f_bytes_to_rgb8(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert bottom-up RGBA16F readback to top-down opaque RGB8.

    Quantization is explicitly fixed: ``floor(clamp(c, 0, 1) * 255 + 0.5)``.
    This avoids driver-dependent float-to-unorm rounding at output time.
    """

    expected = width * height * 4 * 2
    if len(data) != expected:
        raise ValueError(f"RGBA16F readback has {len(data)} bytes, expected {expected}")
    rgba = np.frombuffer(data, dtype="<f2").reshape(height, width, 4)
    rgba = np.flipud(rgba).astype(np.float32)
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    quantized = np.floor(rgb * 255.0 + 0.5).astype(np.uint8)
    return quantized


def rgb8_to_png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return buffer.getvalue()


def frame_png_name(frame_index: int) -> str:
    return f"frame_{frame_index:04d}.png"


def frame_json_name(frame_index: int) -> str:
    return f"frame_{frame_index:04d}.json"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BusinessError(
            OUTPUT_WRITE,
            f"could not write output file: {exc}",
            EXIT_OUTPUT_WRITE,
            {"path": str(path)},
        ) from exc


def write_frame_files(
    output_dir: str,
    frame_index: int,
    png_bytes: bytes,
    frame_stats: dict,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BusinessError(
            OUTPUT_WRITE,
            f"could not create output directory: {exc}",
            EXIT_OUTPUT_WRITE,
            {"path": str(directory)},
        ) from exc
    png_path = directory / frame_png_name(frame_index)
    json_path = directory / frame_json_name(frame_index)
    _atomic_write(png_path, png_bytes)
    _atomic_write(json_path, pretty_json_bytes(frame_stats))
    return png_path, json_path


def write_manifest(output_dir: str, manifest: dict) -> Path:
    path = Path(output_dir) / "stats.json"
    _atomic_write(path, pretty_json_bytes(manifest))
    return path

