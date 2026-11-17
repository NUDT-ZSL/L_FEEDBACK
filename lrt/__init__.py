"""Deterministic, CPU-only layered-material ray tracer."""

from .errors import SceneError
from .scene import load_scene, parse_scene, create_default_scene
from .renderer import Renderer, RenderResult

__all__ = [
    "SceneError",
    "load_scene",
    "parse_scene",
    "create_default_scene",
    "Renderer",
    "RenderResult",
]

