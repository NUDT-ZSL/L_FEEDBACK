from __future__ import annotations

import math

import numpy as np

from .config import LensConfig
from .errors import ConfigurationError


class ThinLensModel:
    """Object-space thin-lens model with pixel-domain blur clamping."""

    def __init__(self, config: LensConfig, image_width: int):
        if image_width <= 0:
            raise ConfigurationError("image width must be positive")
        self.config = config
        self.image_width = int(image_width)
        self.focal_m = config.focal_length_mm / 1000.0
        self.sensor_width_m = config.sensor_width_mm / 1000.0
        self.pixel_pitch_m = self.sensor_width_m / self.image_width

    def coc_diameter_px(self, depth_m: np.ndarray, focus_distance_m: float) -> np.ndarray:
        focus = float(focus_distance_m)
        if not math.isfinite(focus):
            raise ConfigurationError("focus distance must be finite")
        if focus <= self.focal_m:
            raise ConfigurationError(
                f"focus distance {focus:.9g} m must be greater than focal length "
                f"{self.focal_m:.9g} m"
            )
        depth = np.asarray(depth_m, dtype=np.float64)
        if not np.all(np.isfinite(depth)):
            raise ConfigurationError("depth passed to the lens model must be finite")
        if np.any(depth <= 0.0):
            raise ConfigurationError("depth passed to the lens model must be > 0")

        # c = f^2 |s - s_f| / (N s (s_f - f))
        numerator = self.focal_m * self.focal_m * np.abs(depth - focus)
        denominator = self.config.f_number * depth * (focus - self.focal_m)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            diameter_m = numerator / denominator
        diameter_px = diameter_m / self.pixel_pitch_m
        maximum_diameter = 2.0 * self.config.max_blur_radius_px
        diameter_px = np.nan_to_num(
            diameter_px,
            nan=0.0,
            posinf=maximum_diameter,
            neginf=0.0,
        )
        return np.clip(diameter_px, 0.0, maximum_diameter)

    def coc_radius_px(self, depth_m: np.ndarray, focus_distance_m: float) -> np.ndarray:
        radius = 0.5 * self.coc_diameter_px(depth_m, focus_distance_m)
        return np.clip(radius, 0.0, self.config.max_blur_radius_px)
