"""Deterministic first-person spatial audio renderer."""

from .errors import ParameterError, RenderRuntimeError, TrackLoadError
from .scene import SceneSpec, load_scene

__all__ = [
    "ParameterError",
    "RenderRuntimeError",
    "TrackLoadError",
    "SceneSpec",
    "load_scene",
]

__version__ = "1.0.0"
