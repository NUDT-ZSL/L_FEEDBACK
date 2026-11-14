from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ConfigurationError


@dataclass(frozen=True)
class KernelSpec:
    blades: int
    radius_px: float
    rotation_deg: float = 0.0
    supersample: int = 8

    def validate(self) -> None:
        if self.blades < 0:
            raise ConfigurationError("aperture blades must be >= 0")
        if not math.isfinite(self.radius_px) or self.radius_px < 0.0:
            raise ConfigurationError("aperture radius must be a finite value >= 0")
        if not math.isfinite(self.rotation_deg):
            raise ConfigurationError("aperture rotation must be finite")
        if self.supersample < 1:
            raise ConfigurationError("aperture supersample must be >= 1")


@dataclass(frozen=True)
class ApertureKernel:
    weights: np.ndarray
    coverage: np.ndarray
    half_size: int
    radius_px: float
    blades: int
    raw_area_px2: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights.shape

    def metadata(self) -> dict[str, object]:
        return {
            "blades": self.blades,
            "radius_px": float(self.radius_px),
            "matrix_shape": [int(self.weights.shape[0]), int(self.weights.shape[1])],
            "half_size": int(self.half_size),
            "weight_sum": float(np.sum(self.weights, dtype=np.float64)),
            "raw_support_area_px2": float(self.raw_area_px2),
            "normalized_effective_area": 1.0,
        }


def _shape_coverage(
    x: np.ndarray,
    y: np.ndarray,
    radius_px: float,
    blades: int,
    rotation_deg: float,
) -> np.ndarray:
    if blades == 0:
        return (x * x + y * y) <= radius_px * radius_px

    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x) - math.radians(rotation_deg)
    sector = 2.0 * math.pi / blades
    # A regular polygon with one vertex on the positive x axis.
    wrapped = np.mod(theta + sector * 0.5, sector) - sector * 0.5
    boundary = radius_px * math.cos(math.pi / blades) / np.cos(wrapped)
    return rho <= boundary


def build_aperture_kernel(spec: KernelSpec) -> ApertureKernel:
    spec.validate()
    radius = float(spec.radius_px)
    if radius == 0.0:
        weights = np.ones((1, 1), dtype=np.float64)
        return ApertureKernel(
            weights=weights,
            coverage=weights.copy(),
            half_size=0,
            radius_px=0.0,
            blades=spec.blades,
            raw_area_px2=1.0,
        )

    half = math.ceil(radius)
    size = 2 * half + 1
    supersample = spec.supersample
    substep = 1.0 / supersample
    base = -half - 0.5 + substep * 0.5
    samples = base + substep * np.arange(size * supersample, dtype=np.float64)
    gx, gy = np.meshgrid(samples, samples, indexing="xy")
    inside = _shape_coverage(
        gx, gy, radius, spec.blades, spec.rotation_deg
    ).astype(np.float64)

    coverage = inside.reshape(
        size, supersample, size, supersample
    ).mean(axis=(1, 3))
    total = float(np.sum(coverage, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ConfigurationError(
            "aperture shape produced an empty kernel; reduce radius or supersample"
        )
    weights = coverage / total
    # Coverage is already area in pixel^2 (each cell contributes its fractional
    # occupancy); raw_area is reported before normalization.
    raw_area = total
    return ApertureKernel(
        weights=weights.astype(np.float64, copy=False),
        coverage=coverage.astype(np.float64, copy=False),
        half_size=half,
        radius_px=radius,
        blades=spec.blades,
        raw_area_px2=float(raw_area),
    )


def quantized_kernel_specs(
    blades: int,
    max_radius_px: float,
    rotation_deg: float,
    supersample: int,
    quantization_px: float = 0.5,
) -> dict[float, KernelSpec]:
    if max_radius_px < 0.0:
        raise ConfigurationError("max radius must be >= 0")
    steps = math.floor(max_radius_px / quantization_px + 1e-9)
    radii = [0.0] + [(index + 1) * quantization_px for index in range(steps)]
    if max_radius_px > 0.0 and not math.isclose(radii[-1], max_radius_px):
        radii.append(max_radius_px)
    return {
        radius: KernelSpec(
            blades=blades,
            radius_px=radius,
            rotation_deg=rotation_deg,
            supersample=supersample,
        )
        for radius in radii
    }
