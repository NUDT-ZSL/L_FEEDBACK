"""Offline, deterministic terrain tile rendering pipeline prototype.

All heavy modules are import-safe and perform no work at import time so that
the pipeline (and its tests) run deterministically in a single thread.
"""

from .errors import PipelineError, error_code

__all__ = ["PipelineError", "error_code"]
