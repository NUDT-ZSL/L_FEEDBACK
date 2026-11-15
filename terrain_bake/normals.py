"""Heightfield-to-tangent-space normal baking and material previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from .coordinates import (
    HEIGHT_SCALE,
    MATERIAL_COLORS,
    MATERIAL_NAMES,
    world_axis_step,
)
from .errors import E_INVALID_HEIGHTFIELD, BakeError


@dataclass(frozen=True)
class NormalMapResult:
    rgb: np.ndarray
    derivative_x: np.ndarray
    derivative_z: np.ndarray
    pixel_step: float
    height_scale: float


def _bilinear_height(
    heightfield: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    rows, cols = heightfield.shape
    px = np.clip(x, 0.0, 1.0) * (cols - 1)
    pz = np.clip(z, 0.0, 1.0) * (rows - 1)
    x0 = np.floor(px).astype(np.int64)
    z0 = np.floor(pz).astype(np.int64)
    x0 = np.minimum(x0, cols - 1)
    z0 = np.minimum(z0, rows - 1)
    x1 = np.minimum(x0 + 1, cols - 1)
    z1 = np.minimum(z0 + 1, rows - 1)
    tx = px - x0
    tz = pz - z0
    v00 = heightfield[z0, x0]
    v10 = heightfield[z0, x1]
    v01 = heightfield[z1, x0]
    v11 = heightfield[z1, x1]
    value_x0 = v00 * (1.0 - tx) + v10 * tx
    value_x1 = v01 * (1.0 - tx) + v11 * tx
    return value_x0 * (1.0 - tz) + value_x1 * tz


def _bilinear_weights(
    weights: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    rows, cols, channels = weights.shape
    px = np.clip(x, 0.0, 1.0) * (cols - 1)
    pz = np.clip(z, 0.0, 1.0) * (rows - 1)
    x0 = np.minimum(np.floor(px).astype(np.int64), cols - 1)
    z0 = np.minimum(np.floor(pz).astype(np.int64), rows - 1)
    x1 = np.minimum(x0 + 1, cols - 1)
    z1 = np.minimum(z0 + 1, rows - 1)
    tx = (px - x0)[..., None]
    tz = (pz - z0)[..., None]
    v00 = weights[z0, x0]
    v10 = weights[z0, x1]
    v01 = weights[z1, x0]
    v11 = weights[z1, x1]
    value_x0 = v00 * (1.0 - tx) + v10 * tx
    value_x1 = v01 * (1.0 - tx) + v11 * tx
    result = value_x0 * (1.0 - tz) + value_x1 * tz
    return result / np.sum(result, axis=2, keepdims=True)


def world_height_derivatives(
    heightfield: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Finite-difference derivatives using one output pixel of world XZ."""

    if resolution < 2:
        raise BakeError(
            E_INVALID_HEIGHTFIELD,
            "normal map resolution must be at least 2",
            {"resolution": int(resolution)},
        )
    if heightfield.ndim != 2 or min(heightfield.shape) < 2:
        raise BakeError(E_INVALID_HEIGHTFIELD, "invalid heightfield shape")
    if not np.all(np.isfinite(heightfield)):
        raise BakeError(E_INVALID_HEIGHTFIELD, "heightfield contains NaN or Inf")

    step = world_axis_step(resolution)
    x_axis = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    z_axis = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    x_grid, z_grid = np.meshgrid(x_axis, z_axis)
    half = step * 0.5
    x_minus = np.maximum(x_grid - half, 0.0)
    x_plus = np.minimum(x_grid + half, 1.0)
    z_minus = np.maximum(z_grid - half, 0.0)
    z_plus = np.minimum(z_grid + half, 1.0)

    h_x_minus = _bilinear_height(heightfield, x_minus, z_grid)
    h_x_plus = _bilinear_height(heightfield, x_plus, z_grid)
    h_z_minus = _bilinear_height(heightfield, x_grid, z_minus)
    h_z_plus = _bilinear_height(heightfield, x_grid, z_plus)
    derivative_x = HEIGHT_SCALE * (h_x_plus - h_x_minus) / (x_plus - x_minus)
    derivative_z = HEIGHT_SCALE * (h_z_plus - h_z_minus) / (z_plus - z_minus)
    return derivative_x, derivative_z, x_grid, z_grid, step


def bake_normal_map(heightfield: np.ndarray, resolution: int) -> NormalMapResult:
    derivative_x, derivative_z, _, _, step = world_height_derivatives(
        heightfield, resolution
    )
    # T=+X, B=cross(N,T)=-Z, N=+Y.  Decoding is
    # normalize(T*r + B*g + N*b).
    tangent_normal = np.stack(
        (-derivative_x, derivative_z, np.ones_like(derivative_x)),
        axis=2,
    )
    length = np.linalg.norm(tangent_normal, axis=2, keepdims=True)
    tangent_normal /= length
    rgb = np.rint((tangent_normal * 0.5 + 0.5) * 255.0).astype(np.uint8)
    return NormalMapResult(
        rgb=np.ascontiguousarray(rgb),
        derivative_x=derivative_x,
        derivative_z=derivative_z,
        pixel_step=step,
        height_scale=HEIGHT_SCALE,
    )


def _png_metadata(seed: int, **values: str) -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("seed", str(seed))
    info.add_text("coordinate_system", "X=cols,Z=rows,Y=up")
    info.add_text("tangent_axis", "+X")
    info.add_text("bitangent_axis", "-Z")
    info.add_text("normal_axis", "+Y")
    for key, value in values.items():
        info.add_text(key, value)
    return info


def save_normal_png(
    normal_map: NormalMapResult,
    path: Path,
    seed: int,
) -> None:
    metadata = _png_metadata(
        seed,
        pixel_step=f"{normal_map.pixel_step:.17g}",
        height_scale=f"{normal_map.height_scale:.17g}",
    )
    Image.fromarray(normal_map.rgb, mode="RGB").save(path, format="PNG", pnginfo=metadata)


def bake_material_preview(
    heightfield: np.ndarray,
    weights: np.ndarray,
    resolution: int,
) -> np.ndarray:
    derivative_x, derivative_z, x_grid, z_grid, _ = world_height_derivatives(
        heightfield, resolution
    )
    sampled_weights = _bilinear_weights(weights, x_grid, z_grid)
    colors = np.einsum("ijc,ck->ijk", sampled_weights, MATERIAL_COLORS)
    normal = np.stack(
        (-derivative_x, np.ones_like(derivative_x), -derivative_z),
        axis=2,
    )
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    light = np.array([0.38, 0.85, 0.36], dtype=np.float64)
    light /= np.linalg.norm(light)
    diffuse = np.clip(np.sum(normal * light, axis=2), 0.0, 1.0)
    shade = 0.52 + 0.58 * diffuse
    rgb = np.clip(colors * shade[..., None], 0.0, 1.0)
    return np.ascontiguousarray(np.rint(rgb * 255.0).astype(np.uint8))


def save_material_png(
    rgb: np.ndarray,
    path: Path,
    seed: int,
    resolution: int,
) -> None:
    metadata = _png_metadata(
        seed,
        material_order=",".join(MATERIAL_NAMES),
        resolution=f"{resolution}x{resolution}",
    )
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", pnginfo=metadata)
