"""Texture loading and declared/actual format verification."""

from __future__ import annotations

import numpy as np

from .errors import texture_error
from .materials import Material, REQUIRED_TEXTURE_FORMATS, TextureData


def _classify_array(array: np.ndarray) -> tuple[str | None, str]:
    shape = tuple(int(dim) for dim in array.shape)
    if array.ndim == 3 and array.shape[2] == 4 and array.dtype == np.uint8:
        return "RGBA8", f"RGBA8 (uint8, {shape[1]}x{shape[0]}x4)"
    if array.ndim == 3 and array.shape[2] == 4 and array.dtype == np.float32:
        return "RGBA32F", f"RGBA32F (float32, {shape[1]}x{shape[0]}x4)"
    channels = shape[2] if array.ndim == 3 else "?"
    layout = (
        f"{shape[1]}x{shape[0]}x{channels}" if array.ndim == 3 else "x".join(map(str, shape))
    )
    return None, f"{array.dtype.name} ({layout})"


def load_texture(material: Material) -> TextureData:
    spec = material.texture
    try:
        array = np.load(spec.path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise texture_error(
            f"material '{material.name}': cannot load texture '{spec.path}': {exc}",
            material=material.name,
            path=str(spec.path),
            expected=spec.format,
            actual="unreadable",
        ) from exc

    if not isinstance(array, np.ndarray):
        raise texture_error(
            f"material '{material.name}': texture did not load as a NumPy array",
            material=material.name,
            path=str(spec.path),
            expected=spec.format,
            actual=type(array).__name__,
        )

    actual_format, actual_description = _classify_array(array)
    if spec.format not in REQUIRED_TEXTURE_FORMATS:
        raise texture_error(
            f"material '{material.name}': unsupported declared texture format '{spec.format}'; "
            f"expected one of {sorted(REQUIRED_TEXTURE_FORMATS)} "
            f"(actual: {actual_description})",
            material=material.name,
            path=str(spec.path),
            expected=spec.format,
            actual=actual_description,
        )
    if actual_format != spec.format:
        raise texture_error(
            f"material '{material.name}': texture format mismatch; "
            f"expected {spec.format}, actual {actual_description}",
            material=material.name,
            path=str(spec.path),
            expected=spec.format,
            actual=actual_description,
        )
    if array.ndim != 3 or array.shape[2] != 4 or array.shape[0] < 1 or array.shape[1] < 1:
        raise texture_error(
            f"material '{material.name}': texture must have shape HxWx4, actual {array.shape}",
            material=material.name,
            path=str(spec.path),
            expected="HxWx4",
            actual=str(array.shape),
        )

    contiguous = np.ascontiguousarray(array)
    return TextureData(
        array=contiguous,
        declared_format=spec.format,
        actual_format=actual_format,
        actual_description=actual_description,
        width=int(contiguous.shape[1]),
        height=int(contiguous.shape[0]),
        dtype=str(contiguous.dtype),
    )

