"""Image asset decoding, linearization and channel validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .color import srgb_to_linear
from .errors import MaterialError
from .scene import Scene, TextureSpec
from .util import ASSET_SCHEMA, safe_relative_path, sha256_file

SUPPORTED_EXTENSIONS = {".png", ".tga"}


@dataclass
class LoadedAssets:
    base_color: np.ndarray
    normal_tangent: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    manifest: dict[str, Any]


def _round_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 9) for value in np.asarray(values).ravel()]


def _decode_image(path: Path, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise MaterialError(
            "E001", f"{label} texture does not exist: {path}", asset=label
        )
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise MaterialError(
            "E002",
            f"{label} texture format must be PNG or TGA, got {suffix!r}",
            asset=label,
        )
    try:
        source_hash = sha256_file(path)
        if suffix == ".tga":
            image = _decode_tga(path)
            tga_source_depth = int(path.read_bytes()[16])
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            tga_source_depth = None
    except OSError as exc:
        raise MaterialError(
            "E008", f"cannot read {label} texture: {exc}", asset=label
        ) from exc
    if image is None:
        raise MaterialError(
            "E002", f"OpenCV could not decode {label} texture: {path}", asset=label
        )
    if image.dtype not in {np.dtype("uint8"), np.dtype("uint16")}:
        raise MaterialError(
            "E002",
            f"{label} texture must be 8-bit or 16-bit integer data, got {image.dtype}",
            asset=label,
        )
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise MaterialError(
            "E003", f"{label} texture has unsupported dimensions", asset=label
        )
    source_channels = int(image.shape[2])
    if source_channels in {3, 4}:
        rgb = image[:, :, :3][:, :, ::-1]
        if source_channels == 4:
            image = np.concatenate([rgb, image[:, :, 3:4]], axis=2)
        else:
            image = rgb
    bits = int(image.dtype.itemsize * 8)
    if tga_source_depth is not None:
        bits = tga_source_depth
    metadata = {
        "path": str(path),
        "sha256": source_hash,
        "format": suffix.removeprefix("."),
        "source_bit_depth": bits,
        "decoded_bit_depth": int(image.dtype.itemsize * 8),
        "source_channels": source_channels,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    return np.ascontiguousarray(image), metadata


def _decode_tga(path: Path) -> np.ndarray:
    """Decode common TGA 1.0 variants without relying on an OpenCV codec."""

    payload = path.read_bytes()
    if len(payload) < 18:
        raise MaterialError("E002", f"TGA header is truncated: {path}", asset="tga")
    header = payload[:18]
    id_length = int(header[0])
    color_map_type = int(header[1])
    image_type = int(header[2])
    if color_map_type != 0 or image_type not in {2, 3, 10, 11}:
        raise MaterialError(
            "E002",
            f"unsupported TGA type/color-map combination: {image_type}/{color_map_type}",
            asset="tga",
        )
    width = int.from_bytes(header[12:14], "little")
    height = int.from_bytes(header[14:16], "little")
    pixel_depth = int(header[16])
    descriptor = int(header[17])
    if width <= 0 or height <= 0:
        raise MaterialError("E002", "TGA dimensions must be positive", asset="tga")
    if image_type in {2, 10}:
        if pixel_depth not in {16, 24, 32}:
            raise MaterialError(
                "E002", f"unsupported true-color TGA depth: {pixel_depth}", asset="tga"
            )
    elif pixel_depth != 8:
        raise MaterialError(
            "E002", f"unsupported grayscale TGA depth: {pixel_depth}", asset="tga"
        )

    bytes_per_pixel = pixel_depth // 8
    offset = 18 + id_length
    if offset > len(payload):
        raise MaterialError("E002", "TGA image ID extends past EOF", asset="tga")
    pixel_count = width * height
    raw_pixels = bytearray(pixel_count * bytes_per_pixel)

    if image_type in {2, 3}:
        needed = pixel_count * bytes_per_pixel
        if offset + needed > len(payload):
            raise MaterialError("E002", "TGA pixel data is truncated", asset="tga")
        raw_pixels[:] = payload[offset : offset + needed]
    else:
        read_offset = offset
        output_offset = 0
        while output_offset < len(raw_pixels):
            if read_offset >= len(payload):
                raise MaterialError("E002", "TGA RLE data is truncated", asset="tga")
            packet = int(payload[read_offset])
            read_offset += 1
            count = (packet & 0x7F) + 1
            byte_count = count * bytes_per_pixel
            if output_offset + byte_count > len(raw_pixels):
                raise MaterialError("E002", "TGA RLE packet overruns image", asset="tga")
            if packet & 0x80:
                if read_offset + bytes_per_pixel > len(payload):
                    raise MaterialError(
                        "E002", "TGA RLE pixel is truncated", asset="tga"
                    )
                source = payload[
                    read_offset : read_offset + bytes_per_pixel
                ]
                read_offset += bytes_per_pixel
                block = source * count
            else:
                if read_offset + byte_count > len(payload):
                    raise MaterialError(
                        "E002", "TGA RLE literal is truncated", asset="tga"
                    )
                block = payload[read_offset : read_offset + byte_count]
                read_offset += byte_count
            raw_pixels[output_offset : output_offset + byte_count] = block
            output_offset += byte_count

    pixels = np.frombuffer(bytes(raw_pixels), dtype=np.uint8)
    if image_type in {3, 11}:
        image = pixels.reshape(height, width, 1)
    elif pixel_depth == 24:
        image = pixels.reshape(height, width, 3)
    elif pixel_depth == 32:
        image = pixels.reshape(height, width, 4)
    else:
        packed = pixels.reshape(height, width, 2).astype(np.uint16)
        value = packed[:, :, 0] | (packed[:, :, 1] << 8)
        blue = ((value & 0x1F) * 255 // 31).astype(np.uint8)
        green = (((value >> 5) & 0x1F) * 255 // 31).astype(np.uint8)
        red = (((value >> 10) & 0x1F) * 255 // 31).astype(np.uint8)
        image = np.stack([blue, green, red], axis=2)

    if not (descriptor & 0x20):
        image = image[::-1, :, :]
    if descriptor & 0x10:
        image = image[:, ::-1, :]
    return np.ascontiguousarray(image)


def _as_float01(image: np.ndarray) -> np.ndarray:
    maximum = np.float32((1 << (8 * image.dtype.itemsize)) - 1)
    return image.astype(np.float32) / maximum


def _linearize_base(
    image: np.ndarray, metadata: dict[str, Any], spec: TextureSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    channels = int(image.shape[2])
    if channels not in {3, 4}:
        raise MaterialError(
            "E003",
            f"base color needs RGB or RGBA channels, got {channels}",
            asset="base_color",
        )
    values = _as_float01(image)
    if channels == 4:
        values = values[:, :, :3]
    color_space = str(spec.options.get("color_space", "srgb"))
    if color_space == "srgb":
        linear = srgb_to_linear(values)
        method = "srgb_to_linear_iec61966_2_1"
    elif color_space == "linear":
        linear = values.astype(np.float32)
        method = "linear_identity"
    else:
        raise MaterialError(
            "E004",
            "base color color_space must be srgb or linear",
            asset="base_color",
        )
    info = dict(metadata)
    info.update(
        {
            "linearization": method,
            "linear_channels": 3,
            "alpha_used": False,
            "linear_min": _round_list(linear.min(axis=(0, 1))),
            "linear_max": _round_list(linear.max(axis=(0, 1))),
            "linear_mean": _round_list(linear.mean(axis=(0, 1))),
        }
    )
    return linear.astype(np.float32), info


def _decode_normal(
    image: np.ndarray,
    metadata: dict[str, Any],
    spec: TextureSpec,
    surface_green_sign: float,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_channels = int(metadata["source_channels"])
    if source_channels != 3 or image.shape[2] != 3:
        raise MaterialError(
            "E003",
            f"normal map needs exactly RGB channels, got {source_channels}",
            asset="normal",
        )
    encoded = _as_float01(image)
    decoded = encoded * np.float32(2.0) - np.float32(1.0)
    asset_green_sign = float(spec.options.get("green_sign", 1.0))
    green_sign = asset_green_sign * float(surface_green_sign)
    if green_sign not in {-1.0, 1.0}:
        raise MaterialError(
            "E004", "normal green sign must resolve to -1 or 1", asset="normal"
        )
    if green_sign < 0.0:
        decoded[:, :, 1] *= -1.0

    lengths = np.linalg.norm(decoded.astype(np.float64), axis=2)
    deviation = np.abs(lengths - 1.0)
    max_deviation = float(deviation.max())
    bad_count = int(np.count_nonzero(deviation > tolerance))
    max_flat = int(np.argmax(deviation))
    y, x = np.unravel_index(max_flat, deviation.shape)
    if max_deviation > tolerance:
        raise MaterialError(
            "E005",
            (
                "normal map contains non-unit vectors: "
                f"max |length-1|={max_deviation:.8f}, tolerance={tolerance:.8f}"
            ),
            asset="normal",
            location={
                "x": int(x),
                "y": int(y),
                "length": round(float(lengths[y, x]), 9),
                "deviation": round(max_deviation, 9),
            },
            details={
                "tolerance": tolerance,
                "affected_pixels": bad_count,
                "pixel_count": int(deviation.size),
            },
        )
    normalized = decoded.astype(np.float32)
    normalized /= np.maximum(lengths[:, :, None], 1e-12).astype(np.float32)
    info = dict(metadata)
    info.update(
        {
            "linearization": "linear_2x_minus_1_then_unit_normalize",
            "linear_channels": 3,
            "normal_space": "tangent",
            "green_sign": green_sign,
            "length_min": round(float(lengths.min()), 9),
            "length_max": round(float(lengths.max()), 9),
            "length_max_abs_deviation": round(max_deviation, 9),
            "length_tolerance": tolerance,
        }
    )
    return normalized, info


def _load_scalar(
    spec: TextureSpec,
    *,
    default: float,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    label = spec.label
    if spec.path is None:
        values = np.full((target_shape[0], target_shape[1], 1), default, np.float32)
        return values, {
            "source": "scene_default",
            "value": round(float(default), 9),
            "linearization": "scene_default_fill",
            "linear_channels": 1,
            "width": int(target_shape[1]),
            "height": int(target_shape[0]),
        }

    image, metadata = _decode_image(spec.path, label)
    source_channels = int(metadata.get("source_channels", image.shape[2]))
    if source_channels == 1:
        values = _as_float01(image[:, :, 0])
        channel_used = 0
    elif "channel" in spec.options:
        channel = int(spec.options["channel"])
        if channel < 0 or channel >= source_channels:
            raise MaterialError(
                "E003",
                f"{label} channel {channel} is absent from a {source_channels}-channel map",
                asset=label,
            )
        values = _as_float01(image[:, :, channel])
        channel_used = channel
    else:
        raise MaterialError(
            "E003",
            (
                f"{label} map must be single-channel or declare an explicit "
                f"channel; got {source_channels} channels"
            ),
            asset=label,
        )
    if values.shape != target_shape:
        raise MaterialError(
            "E006",
            (
                f"{label} map size {values.shape[::-1]} does not match "
                f"{target_shape[::-1]}"
            ),
            asset=label,
        )
    lower = float(spec.options.get("min", 0.0))
    upper = float(spec.options.get("max", 1.0))
    values = np.clip(values, lower, upper).astype(np.float32)
    info = dict(metadata)
    info.update(
        {
            "linearization": "linear_single_channel_clamp",
            "linear_channels": 1,
            "channel_used": channel_used,
            "clamp_min": lower,
            "clamp_max": upper,
            "linear_min": round(float(values.min()), 9),
            "linear_max": round(float(values.max()), 9),
        }
    )
    return values[:, :, None], info


def load_assets(scene: Scene) -> LoadedAssets:
    base_spec = scene.assets["base_color"]
    assert base_spec.path is not None
    base_image, base_metadata = _decode_image(base_spec.path, "base_color")
    base_color, base_info = _linearize_base(base_image, base_metadata, base_spec)
    height, width = base_color.shape[:2]

    normal_spec = scene.assets["normal"]
    assert normal_spec.path is not None
    normal_image, normal_metadata = _decode_image(normal_spec.path, "normal")
    if (
        normal_metadata["width"] != width
        or normal_metadata["height"] != height
    ):
        raise MaterialError(
            "E006",
            (
                f"normal map size {normal_metadata['width']}x"
                f"{normal_metadata['height']} does not match base color "
                f"{width}x{height}"
            ),
            asset="normal",
        )
    tolerance = float(scene.validation["normal_length_tolerance"])
    normal, normal_info = _decode_normal(
        normal_image,
        normal_metadata,
        normal_spec,
        float(scene.surface["normal_green_sign"]),
        tolerance,
    )

    roughness, roughness_info = _load_scalar(
        scene.assets["roughness"],
        default=scene.defaults["roughness"],
        target_shape=(height, width),
    )
    metallic, metallic_info = _load_scalar(
        scene.assets["metallic"],
        default=scene.defaults["metallic"],
        target_shape=(height, width),
    )
    if roughness.shape[:2] != metallic.shape[:2]:
        raise MaterialError("E006", "roughness and metallic map sizes differ")

    for info in (base_info, normal_info, roughness_info, metallic_info):
        path_value = info.get("path")
        if path_value:
            info["path"] = safe_relative_path(Path(path_value), scene.base_dir)

    manifest = {
        "schema": ASSET_SCHEMA,
        "scene": scene.path.name,
        "resolution": [int(width), int(height)],
        "textures": {
            "base_color": base_info,
            "normal": normal_info,
            "roughness": roughness_info,
            "metallic": metallic_info,
        },
        "defaults": {
            "roughness": round(float(scene.defaults["roughness"]), 9),
            "metallic": round(float(scene.defaults["metallic"]), 9),
        },
    }
    return LoadedAssets(
        base_color=base_color,
        normal_tangent=normal,
        roughness=roughness,
        metallic=metallic,
        manifest=manifest,
    )
