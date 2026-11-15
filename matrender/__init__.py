"""Deterministic offline material-sphere renderer."""

from .scene import Scene, SceneValidationError, load_scene, parse_scene

__all__ = ["Scene", "SceneValidationError", "load_scene", "parse_scene"]
__version__ = "1.0.0"

