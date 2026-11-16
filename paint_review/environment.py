from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def direction_from_lat_lon(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    cos_lat = np.cos(latitude)
    return np.stack(
        (
            cos_lat * np.sin(longitude),
            np.sin(latitude),
            cos_lat * np.cos(longitude),
        ),
        axis=-1,
    )


def direction_to_uv(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    u = lon / (2.0 * math.pi) + 0.5
    v = 0.5 - lat / math.pi
    return u, v


def procedural_environment(width: int = 512, height: int = 256) -> np.ndarray:
    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    v = (np.arange(height, dtype=np.float64) + 0.5) / height
    uu, vv = np.meshgrid(u, v, indexing="xy")
    lon = (uu - 0.5) * 2.0 * math.pi
    lat = (0.5 - vv) * math.pi
    dirs = direction_from_lat_lon(lat, lon)

    sky_mix = np.clip(dirs[..., 1] * 2.1 + 0.46, 0.0, 1.0)
    horizon = np.exp(-np.abs(dirs[..., 1]) / 0.16)
    floor = np.exp(-np.maximum(-dirs[..., 1], 0.0) / 0.32)
    base = (
        np.array((0.050, 0.058, 0.078))
        + np.array((0.150, 0.195, 0.270)) * sky_mix[..., None]
        + np.array((0.360, 0.345, 0.310)) * horizon[..., None] * 0.55
        + np.array((0.048, 0.052, 0.064)) * floor[..., None]
    )

    def angular_spot(lon0: float, lat0: float, sigma: float, peak: float):
        center = direction_from_lat_lon(
            np.asarray(lat0), np.asarray(lon0)
        )
        dot = np.clip(np.sum(dirs * center, axis=-1), -1.0, 1.0)
        angle = np.arccos(dot)
        return np.exp(-0.5 * (angle / sigma) ** 2) * peak

    sun = angular_spot(0.62, 0.48, 0.032, 42.0)
    sun_glow = angular_spot(0.62, 0.48, 0.27, 4.2)
    strip_a = angular_spot(-1.18, 0.22, 0.11, 8.5)
    strip_b = angular_spot(1.72, 0.17, 0.085, 6.0)
    rear_fill = angular_spot(-2.52, 0.05, 0.19, 2.8)

    warm = np.array((1.00, 0.77, 0.50))
    cool = np.array((0.58, 0.76, 1.00))
    base += (sun + sun_glow)[..., None] * warm
    base += strip_a[..., None] * cool
    base += strip_b[..., None] * np.array((0.82, 0.91, 1.00))
    base += rear_fill[..., None] * np.array((0.50, 0.65, 1.00))

    vignette = 0.88 + 0.12 * np.cos(np.clip(dirs[..., 1], -1.0, 1.0) * 1.2) ** 2
    return np.ascontiguousarray(base * vignette[..., None], dtype=np.float64)


def _blur_axis_wrap(image: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma <= 1e-6:
        return image.copy()
    radius = max(1, int(math.ceil(sigma * 2.5)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.zeros_like(image)
    if axis == 1:
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            result += weight * np.roll(image, offset, axis=1)
    else:
        padded = np.pad(image, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        for index, weight in enumerate(kernel):
            result += weight * padded[index : index + image.shape[0], :, :]
    return result


def _blur_equirect(image: np.ndarray, sigma: float) -> np.ndarray:
    return _blur_axis_wrap(_blur_axis_wrap(image, sigma, axis=1), sigma, axis=0)


def _downsample_equirect(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    h2 = max(2, h // 2)
    w2 = max(4, w // 2)
    hs = h // h2
    ws = w // w2
    cropped = image[: h2 * hs, : w2 * ws]
    return cropped.reshape(h2, hs, w2, ws, 3).mean(axis=(1, 3))


def _sample_texture(tex: np.ndarray, direction: np.ndarray) -> np.ndarray:
    u, v = direction_to_uv(direction)
    h, w = tex.shape[:2]
    x = (u % 1.0) * w - 0.5
    y = np.clip(v, 0.0, 1.0) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0)[..., None]
    ty = (y - y0)[..., None]
    x0m = np.mod(x0, w)
    x1m = np.mod(x0 + 1, w)
    y0m = np.clip(y0, 0, h - 1)
    y1m = np.clip(y0 + 1, 0, h - 1)
    a = tex[y0m, x0m]
    b = tex[y0m, x1m]
    c = tex[y1m, x0m]
    d = tex[y1m, x1m]
    top = a * (1.0 - tx) + b * tx
    bottom = c * (1.0 - tx) + d * tx
    return top * (1.0 - ty) + bottom * ty


@dataclass
class HdrEnvironment:
    levels: Tuple[np.ndarray, ...]
    roughness_centers: Tuple[float, ...]
    source_width: int
    source_height: int
    prefilter_ms: float
    digest: str

    @classmethod
    def build(
        cls, width: int = 512, height: int = 256, level_count: int = 8
    ) -> "HdrEnvironment":
        started = time.perf_counter()
        base = procedural_environment(width, height)
        levels: List[np.ndarray] = [_blur_equirect(base, 0.55)]
        for _ in range(1, level_count):
            down = _downsample_equirect(levels[-1])
            levels.append(_blur_equirect(down, 0.72))
        centers = tuple(
            min(0.98, max(0.01, i / max(1, level_count - 1) * 0.91))
            for i in range(level_count)
        )
        digest = hashlib.sha256()
        for image in levels:
            digest.update(np.ascontiguousarray(image, dtype=np.float64).tobytes())
        elapsed = (time.perf_counter() - started) * 1000.0
        return cls(
            levels=tuple(np.ascontiguousarray(x, dtype=np.float64) for x in levels),
            roughness_centers=centers,
            source_width=int(width),
            source_height=int(height),
            prefilter_ms=elapsed,
            digest=digest.hexdigest(),
        )

    @property
    def level_count(self) -> int:
        return len(self.levels)

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((int(x.shape[1]), int(x.shape[0])) for x in self.levels)

    def level_for_roughness(self, roughness: float) -> int:
        r = min(0.999, max(0.001, float(roughness)))
        index = int(round(r * (self.level_count - 1)))
        return min(self.level_count - 1, max(0, index))

    def sample(self, direction: np.ndarray, level: int = 0) -> np.ndarray:
        index = min(self.level_count - 1, max(0, int(level)))
        return _sample_texture(self.levels[index], direction)

    def sample_rough(self, direction: np.ndarray, roughness: float) -> np.ndarray:
        return self.sample(direction, self.level_for_roughness(roughness))

    def clear(self) -> None:
        self.levels = tuple()

