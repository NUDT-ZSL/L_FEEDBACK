"""Deterministic CPU terrain renderer."""

from .config import (
    Camera,
    CascadeSettings,
    DirectionalLight,
    Material,
    PointLight,
    RenderSettings,
    ToneSettings,
)
from .errors import RenderError, ValidationError
from .renderer import RenderCache, RenderResult, incremental_render, render_scene
from .samples import generate_heightfield, generate_weight_maps

__all__ = [
    "Camera",
    "CascadeSettings",
    "DirectionalLight",
    "Material",
    "PointLight",
    "RenderCache",
    "RenderError",
    "RenderResult",
    "RenderSettings",
    "ToneSettings",
    "ValidationError",
    "generate_heightfield",
    "generate_weight_maps",
    "incremental_render",
    "render_scene",
]

__version__ = "1.0.0"
