"""Render backend abstractions and implementations."""

from .base import RenderBackend
from .software import SoftwareBackend

__all__ = ["RenderBackend", "SoftwareBackend"]
