"""Deterministic CPU-only layered soft rasterizer."""

from .errors import RenderError, SceneError, UsageError
from .renderer import RenderResult, render_scene
from .scene import load_scene

__all__ = [
    "RenderError",
    "RenderResult",
    "SceneError",
    "UsageError",
    "load_scene",
    "render_scene",
]

__version__ = "1.0.0"
