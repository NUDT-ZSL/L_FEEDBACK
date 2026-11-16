"""Offline OpenGL deferred-render validation toolkit."""

__version__ = "1.0.0"

from .errors import AppError, ConfigError, RenderError, VerificationError

__all__ = ["AppError", "ConfigError", "RenderError", "VerificationError", "__version__"]
