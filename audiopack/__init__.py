"""Offline audio packaging toolkit for esports replay event voice-over.

The package turns a deterministic event manifest plus per-language WAV voice
clips into 48kHz/16bit mono tracks, CUE indexes and a JSON report, with
content-addressed snapshots that allow interrupted runs to resume.
"""

from .config import Config
from .errors import Issue, ErrorCode
from .pipeline import run_pipeline, PipelineResult

__all__ = [
    "Config",
    "Issue",
    "ErrorCode",
    "run_pipeline",
    "PipelineResult",
]

__version__ = "1.0.0"
