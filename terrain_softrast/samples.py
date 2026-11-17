from __future__ import annotations

import numpy as np

from .errors import ValidationError


def _bilinear_lattice(lattice: np.ndarray, size: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, lattice.shape[0], dtype=np.float64)
    target = np.linspace(0.0, 1.0, size, dtype=np.float64)
    first = np.empty((size, lattice.shape[1]), dtype=np.float64)
    for column in range(lattice.shape[1]):
        first[:, column] = np.interp(target, source, lattice[:, column])
    result = np.empty((size, size), dtype=np.float64)
    for row in range(size):
        result[row] = np.interp(target, source, first[row])
    return result


def fractal_value_noise(size: int, seed: int, octaves: int = 6) -> np.ndarray:
    if size < 2:
        return np.zeros((size, size), dtype=np.float64)
    rng = np.random.default_rng(np.uint64(seed))
    result = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    for octave in range(octaves):
        resolution = min(65, 2 ** (octave + 1) + 1)
        lattice = rng.random((resolution, resolution), dtype=np.float64)
        layer = _bilinear_lattice(lattice, size)
        result += amplitude * layer
        total += amplitude
        amplitude *= 0.5
    result /= total
    return result


def generate_heightfield(size: int, seed: int = 1337) -> np.ndarray:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 3:
        raise ValidationError("高度场规模必须为不小于 3 的整数")
    if size > 2049:
        raise ValidationError("高度场规模不能超过 2049")
    size = int(size)
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15))
    broad = fractal_value_noise(size, seed, octaves=5) * 2.0 - 1.0
    detail = fractal_value_noise(size, seed + 101, octaves=7) * 2.0 - 1.0
    x = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    xx, zz = np.meshgrid(x, x)
    basin = -0.62 * np.exp(-((xx + 0.20) ** 2 / 0.16 + (zz - 0.15) ** 2 / 0.30))
    ridge = 0.52 * np.exp(-((xx - 0.34) ** 2 / 0.09 + (zz + 0.24) ** 2 / 0.14))
    tilt = 0.15 * xx - 0.10 * zz
    height = 0.72 * broad + 0.18 * detail + basin + ridge + tilt
    low = float(np.min(height))
    high = float(np.max(height))
    if high - low <= 1e-12:
        return np.zeros((size, size), dtype=np.float64)
    height = (height - low) / (high - low)
    height = height + 0.015 * (rng.random((size, size), dtype=np.float64) - 0.5)
    return np.asarray(height, dtype=np.float64)


def generate_weight_maps(height: np.ndarray, layer_count: int = 4, seed: int = 4242) -> np.ndarray:
    height = np.asarray(height, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValidationError("高度场必须是至少 2x2 的二维数组")
    if isinstance(layer_count, bool) or not isinstance(layer_count, (int, np.integer)) or layer_count <= 0:
        raise ValidationError("材质图层数必须为正整数")
    layer_count = int(layer_count)
    normalized_height = (height - float(np.min(height))) / max(float(np.max(height) - np.min(height)), 1e-12)
    grad_z, grad_x = np.gradient(normalized_height)
    slope = np.sqrt(grad_x * grad_x + grad_z * grad_z)
    noise = fractal_value_noise(height.shape[0], seed, octaves=5)
    xx, zz = np.meshgrid(
        np.linspace(0.0, 1.0, height.shape[1], dtype=np.float64),
        np.linspace(0.0, 1.0, height.shape[0], dtype=np.float64),
    )
    weight_maps = np.empty((layer_count, *height.shape), dtype=np.float64)
    for layer in range(layer_count):
        phase = layer * 1.731
        wave = 0.5 + 0.5 * np.sin((xx * (2.0 + layer) + zz * (1.3 + 0.4 * layer)) * np.pi * 2.0 + phase)
        altitude = np.clip(1.0 - np.abs(normalized_height - (0.16 + 0.23 * layer)), 0.0, 1.0)
        rocky = np.clip(slope * (1.0 + 0.3 * layer) - 0.02 * layer, 0.0, 1.0)
        material_noise = 0.15 + 0.70 * np.mod(noise + layer * 0.237, 1.0)
        weight_maps[layer] = (0.16 + altitude) * (0.25 + material_noise) * (0.55 + 0.9 * rocky)
    weight_maps = np.maximum(weight_maps, 1e-8)
    weight_maps /= np.sum(weight_maps, axis=0, keepdims=True)
    return np.asarray(weight_maps, dtype=np.float64)
