"""Deterministic CPU software renderer."""

from .errors import BusinessError
from .renderer import RenderResult, render_scene_file

__all__ = ["BusinessError", "RenderResult", "render_scene_file"]
__version__ = "1.0.0"
