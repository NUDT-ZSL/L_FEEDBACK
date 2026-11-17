"""Atomic 32-bit float image output."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .errors import RENDER_ERR_IO, RenderError


def write_image(path: str | Path, image: np.ndarray) -> None:
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".npy", ".pfm"}:
        raise RenderError(
            RENDER_ERR_IO,
            "unsupported output extension; use .npy or .pfm",
            {"path": str(destination)},
        )
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.float32:
        raise RenderError(
            RENDER_ERR_IO,
            "output image must be a float32 H x W x 3 array",
            {"path": str(destination)},
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            if suffix == ".npy":
                with temporary.open("wb") as handle:
                    np.save(handle, image, allow_pickle=False)
            else:
                _write_pfm(temporary, image)
            os.replace(temporary, destination)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
    except RenderError:
        raise
    except OSError as exc:
        raise RenderError(
            RENDER_ERR_IO,
            "failed to write output image",
            {"path": str(destination), "error": str(exc)},
        ) from exc


def _write_pfm(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    payload = np.ascontiguousarray(image[::-1], dtype="<f4")
    with path.open("wb") as handle:
        handle.write(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))
        handle.write(payload.tobytes(order="C"))
