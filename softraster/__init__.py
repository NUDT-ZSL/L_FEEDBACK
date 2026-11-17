"""NumPy CPU renderer for layered materials and soft shadows."""

from .renderer import render_path, render_postprocessed, render_scene
from .scene import load_scene, parse_scene

__all__ = [
    "load_scene",
    "parse_scene",
    "render_path",
    "render_postprocessed",
    "render_scene",
]
