"""Deterministic offline material preview renderer.

The package intentionally depends only on NumPy and the Python standard
library at runtime. Pillow is used by the test-suite as an independent PNG
decoder.
"""

from .errors import ErrorCode, PipelineError

__all__ = ["ErrorCode", "PipelineError"]
