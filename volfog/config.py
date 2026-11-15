"""Render configuration and unified upfront input validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .errors import RenderError


def _finite_float(name: str, value) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise RenderError(name, "value must be a finite number")
    return value


def _positive_int(name: str, value, minimum: int = 1) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise RenderError(name, "value must be an integer")
    if ivalue < minimum:
        raise RenderError(name, f"value must be >= {minimum}")
    return ivalue


@dataclass
class RenderConfig:
    """All tunables for one render. Distances are in world units."""

    mode: str = "single"
    width: int = 128
    height: int = 128
    voxel_res: int = 64
    seed: int = 1234
    density_file: Optional[str] = None
    output_dir: str = "output"
    output_prefix: str = "fog"
    parallel: bool = False
    threads: int = 0

    camera_pos: Tuple[float, float, float] = (0.0, 0.0, -2.2)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    focal_length: float = 1.0

    light_pos: Tuple[float, float, float] = (0.7, 0.8, -0.6)
    light_color: Tuple[float, float, float] = (1.0, 0.95, 0.85)
    light_intensity: float = 6.0

    step_size: float = 0.02
    max_samples: int = 220
    terminate_threshold: float = 0.005
    depth_threshold: float = 0.01
    extinction: float = 2.5
    albedo: float = 0.9
    phase_g: float = 0.0
    light_samples: int = 8
    multiple_strength: float = 1.0

    @property
    def max_distance(self) -> float:
        return self.step_size * self.max_samples

    def validate(self) -> None:
        """Validate every scalar/vector parameter before any heavy work."""
        if self.mode not in ("single", "multiple"):
            raise RenderError("mode", "must be 'single' or 'multiple'")

        self.width = _positive_int("width", self.width)
        self.height = _positive_int("height", self.height)
        self.voxel_res = _positive_int("voxel_res", self.voxel_res)
        self.max_samples = _positive_int("max_samples", self.max_samples)
        self.light_samples = _positive_int("light_samples", self.light_samples)
        if int(self.threads) < 0:
            raise RenderError("threads", "value must be >= 0")

        for name in (
            "focal_length",
            "step_size",
            "terminate_threshold",
            "depth_threshold",
            "extinction",
            "albedo",
            "phase_g",
            "light_intensity",
            "multiple_strength",
        ):
            setattr(self, name, _finite_float(name, getattr(self, name)))

        if self.focal_length <= 0.0:
            raise RenderError("focal_length", "value must be positive")
        if self.step_size <= 0.0:
            raise RenderError("step_size", "value must be positive")
        if not 0.0 < self.terminate_threshold <= 1.0:
            raise RenderError("terminate_threshold", "must be within (0, 1]")
        if not 0.0 < self.depth_threshold <= 1.0:
            raise RenderError("depth_threshold", "must be within (0, 1]")
        if self.terminate_threshold >= self.depth_threshold:
            raise RenderError(
                "terminate_threshold",
                "must be smaller than depth_threshold so the depth event is observable",
            )
        if self.extinction <= 0.0:
            raise RenderError("extinction", "value must be positive")
        if not 0.0 <= self.albedo <= 1.0:
            raise RenderError("albedo", "must be within [0, 1]")
        if not -1.0 < self.phase_g < 1.0:
            raise RenderError("phase_g", "must be within (-1, 1)")
        if self.light_intensity < 0.0:
            raise RenderError("light_intensity", "value must be non-negative")
        if self.multiple_strength < 0.0:
            raise RenderError("multiple_strength", "value must be non-negative")

        for vec_name in ("camera_pos", "camera_target", "light_pos", "light_color"):
            vec = getattr(self, vec_name)
            if len(vec) != 3:
                raise RenderError(vec_name, "must contain exactly 3 components")
            checked = tuple(_finite_float(vec_name, component) for component in vec)
            setattr(self, vec_name, checked)

        if any(component < 0.0 for component in self.light_color):
            raise RenderError("light_color", "components must be non-negative")
        if np.allclose(np.asarray(self.camera_pos), np.asarray(self.camera_target)):
            raise RenderError("camera_target", "must differ from camera_pos")


def validate_density_array(density: np.ndarray, param: str = "density") -> np.ndarray:
    """Shared checks for procedural and externally loaded density fields."""
    if not isinstance(density, np.ndarray):
        raise RenderError(param, "density field must be a numpy array")
    if density.ndim != 3:
        raise RenderError(param, f"density field must be 3D, got {density.ndim}D")
    if not np.issubdtype(density.dtype, np.floating):
        density = density.astype(np.float64)
    if not np.all(np.isfinite(density)):
        raise RenderError(param, "density field contains NaN or infinity")
    if np.any(density < 0.0):
        raise RenderError(param, "density field contains negative values")
    return density


def normalize_density(density: np.ndarray) -> np.ndarray:
    """Min-max normalize a valid density field into [0, 1]."""
    dmin = float(density.min())
    dmax = float(density.max())
    if dmax - dmin < 1e-12:
        return np.zeros_like(density, dtype=np.float64)
    normalized = (density - dmin) / (dmax - dmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float64)
