from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import ErrorCode


_UINT32_MASK = np.uint64(0xFFFFFFFF)


def _hash_u32(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Counter-based integer hash; no process RNG or platform RNG is used."""

    xx = x.astype(np.int64).astype(np.uint64) & _UINT32_MASK
    yy = y.astype(np.int64).astype(np.uint64) & _UINT32_MASK
    ss = np.uint64(seed & 0xFFFFFFFF)
    h = (xx * np.uint64(0x9E3779B1)) ^ (yy * np.uint64(0x85EBCA77)) ^ (ss * np.uint64(0xC2B2AE3D))
    h ^= h >> np.uint64(16)
    h *= np.uint64(0x7FEB352D)
    h &= _UINT32_MASK
    h ^= h >> np.uint64(15)
    h *= np.uint64(0x846CA68B)
    h &= _UINT32_MASK
    h ^= h >> np.uint64(16)
    return (h & np.uint64(0x00FFFFFF)).astype(np.float64) / float(0x00FFFFFF)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def _value_noise(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    frequency: int,
    seed: int,
    octaves: int,
) -> np.ndarray:
    total = np.zeros_like(local_x, dtype=np.float64)
    amplitude = 1.0
    amplitude_sum = 0.0
    for octave in range(octaves):
        freq = frequency * (1 << octave)
        fx = local_x * freq
        fy = local_y * freq
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        tx = np.mod(ix, freq)
        ty = np.mod(iy, freq)
        tx1 = np.mod(ix + 1, freq)
        ty1 = np.mod(iy + 1, freq)
        sx = _smoothstep(fx - ix)
        sy = _smoothstep(fy - iy)
        origin_x = tile_x * freq
        origin_y = tile_y * freq
        octave_seed = seed + octave * 1013
        n00 = _hash_u32(origin_x + tx, origin_y + ty, octave_seed)
        n10 = _hash_u32(origin_x + tx1, origin_y + ty, octave_seed)
        n01 = _hash_u32(origin_x + tx, origin_y + ty1, octave_seed)
        n11 = _hash_u32(origin_x + tx1, origin_y + ty1, octave_seed)
        top = n00 * (1.0 - sx) + n10 * sx
        bottom = n01 * (1.0 - sx) + n11 * sx
        total += amplitude * (top * (1.0 - sy) + bottom * sy)
        amplitude_sum += amplitude
        amplitude *= 0.5
    return total / max(amplitude_sum, 1e-15)


def transformed_uv(
    uv: np.ndarray,
    tiling: tuple[float, float],
    rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tiled_x = np.asarray(uv[..., 0], dtype=np.float64) * tiling[0]
    tiled_y = np.asarray(uv[..., 1], dtype=np.float64) * tiling[1]
    tile_x = np.floor(tiled_x)
    tile_y = np.floor(tiled_y)
    local_x = tiled_x - tile_x
    local_y = tiled_y - tile_y
    angle = math.radians(rotation_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    centered_x = local_x - 0.5
    centered_y = local_y - 0.5
    rotated_x = centered_x * cosine - centered_y * sine + 0.5
    rotated_y = centered_x * sine + centered_y * cosine + 0.5
    local_x = np.mod(rotated_x, 1.0)
    local_y = np.mod(rotated_y, 1.0)
    return tile_x, tile_y, local_x, local_y


def _require_number(pattern: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in pattern:
        if default is not None:
            return default
        raise KeyError(key)
    value = pattern[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"pattern.{key} must be a finite number")
    return float(value)


def _require_pair(pattern: dict[str, Any], key: str, default: tuple[float, float] | None = None) -> tuple[float, float]:
    if key not in pattern:
        if default is not None:
            return default
        raise KeyError(key)
    value = pattern[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"pattern.{key} must contain two numbers")
    return (
        _require_number({key: value[0]}, key),
        _require_number({key: value[1]}, key),
    )


def _require_rgb(pattern: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = pattern[key]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"pattern.{key} must contain three numbers")
    result = tuple(_require_number({key: value[i]}, key) for i in range(3))
    return result  # type: ignore[return-value]


def evaluate_checker(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    pattern: dict[str, Any],
) -> np.ndarray:
    frequency = _require_pair(pattern, "frequency", (8.0, 8.0))
    if frequency[0] <= 0.0 or frequency[1] <= 0.0:
        raise ValueError("pattern.frequency must be positive")
    cx = np.floor(local_x * frequency[0]).astype(np.int64)
    cy = np.floor(local_y * frequency[1]).astype(np.int64)
    tile_parity = np.mod(tile_x.astype(np.int64) + tile_y.astype(np.int64), 2)
    checker = np.mod(cx + cy + tile_parity, 2).astype(np.float64)
    return checker


def evaluate_stripes(
    local_x: np.ndarray,
    local_y: np.ndarray,
    pattern: dict[str, Any],
) -> np.ndarray:
    frequency = _require_number(pattern, "frequency", 6.0)
    duty = _require_number(pattern, "duty", 0.5)
    axis = pattern.get("axis", "y")
    if frequency <= 0.0:
        raise ValueError("pattern.frequency must be positive")
    if not (0.0 < duty < 1.0):
        raise ValueError("pattern.duty must be in (0, 1)")
    coordinate = local_x if axis == "x" else local_y
    phase = np.mod(coordinate * frequency, 1.0)
    return (phase < duty).astype(np.float64)


def evaluate_pattern_rgb(
    uv: np.ndarray,
    pattern: dict[str, Any],
    tiling: tuple[float, float],
    rotation_deg: float,
) -> np.ndarray:
    tile_x, tile_y, local_x, local_y = transformed_uv(uv, tiling, rotation_deg)
    pattern_type = pattern.get("type")
    if pattern_type == "checker":
        mask = evaluate_checker(tile_x, tile_y, local_x, local_y, pattern)
        color_a = _require_rgb(pattern, "color_a")
        color_b = _require_rgb(pattern, "color_b")
        a = np.asarray(color_a, dtype=np.float64)
        b = np.asarray(color_b, dtype=np.float64)
        return a + (b - a) * mask[..., None]
    if pattern_type == "stripes":
        mask = evaluate_stripes(local_x, local_y, pattern)
        color_a = _require_rgb(pattern, "color_a")
        color_b = _require_rgb(pattern, "color_b")
        a = np.asarray(color_a, dtype=np.float64)
        b = np.asarray(color_b, dtype=np.float64)
        return a + (b - a) * mask[..., None]
    if pattern_type == "noise":
        frequency = _require_number(pattern, "frequency", 6.0)
        if frequency <= 0.0 or frequency != int(frequency):
            raise ValueError("pattern.frequency must be a positive integer for noise")
        octaves = int(_require_number(pattern, "octaves", 3.0))
        if octaves < 1 or octaves > 6:
            raise ValueError("pattern.octaves must be in [1, 6]")
        seed = int(_require_number(pattern, "seed", 1.0))
        noise = _value_noise(tile_x, tile_y, local_x, local_y, int(frequency), seed, octaves)
        color_a = np.asarray(_require_rgb(pattern, "color_a"), dtype=np.float64)
        color_b = np.asarray(_require_rgb(pattern, "color_b"), dtype=np.float64)
        return color_a + (color_b - color_a) * noise[..., None]
    raise ValueError(f"unsupported pattern type {pattern_type!r}")


def evaluate_pattern_scalar(
    uv: np.ndarray,
    pattern: dict[str, Any],
    tiling: tuple[float, float],
    rotation_deg: float,
) -> np.ndarray:
    tile_x, tile_y, local_x, local_y = transformed_uv(uv, tiling, rotation_deg)
    pattern_type = pattern.get("type")
    value_a = _require_number(pattern, "value_a", 0.0)
    value_b = _require_number(pattern, "value_b", 1.0)
    if pattern_type == "checker":
        mask = evaluate_checker(tile_x, tile_y, local_x, local_y, pattern)
        return value_a + (value_b - value_a) * mask
    if pattern_type == "stripes":
        mask = evaluate_stripes(local_x, local_y, pattern)
        return value_a + (value_b - value_a) * mask
    if pattern_type == "noise":
        frequency = _require_number(pattern, "frequency", 6.0)
        if frequency <= 0.0 or frequency != int(frequency):
            raise ValueError("pattern.frequency must be a positive integer for noise")
        octaves = int(_require_number(pattern, "octaves", 3.0))
        if octaves < 1 or octaves > 6:
            raise ValueError("pattern.octaves must be in [1, 6]")
        seed = int(_require_number(pattern, "seed", 1.0))
        noise = _value_noise(tile_x, tile_y, local_x, local_y, int(frequency), seed, octaves)
        return value_a + (value_b - value_a) * noise
    raise ValueError(f"unsupported pattern type {pattern_type!r}")


def pattern_error(exc: Exception, *, path: str) -> tuple[str, str]:
    message = str(exc)
    if message.startswith("unsupported pattern type"):
        return ErrorCode.MAT_PATTERN_UNSUPPORTED, message
    return ErrorCode.MAT_PATTERN_FIELD_MISSING, message
