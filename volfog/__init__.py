"""Offline volumetric fog renderer.

Deterministic procedural density fields, single-point-light ray marching with
Beer-Lambert transmittance, and single / multiple scattering modes.
"""

from .config import RenderConfig
from .density import generate_density_field, load_density_field
from .errors import RenderError
from .pipeline import render_to_files
from .renderer import render_frame

__all__ = [
    "RenderConfig",
    "RenderError",
    "generate_density_field",
    "load_density_field",
    "render_frame",
    "render_to_files",
]
