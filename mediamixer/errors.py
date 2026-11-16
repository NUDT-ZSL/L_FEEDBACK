"""Typed errors and JSON error reporting.

Validation failures exit with code 2; any runtime/processing failure
(subprocess failure, IO failure, malformed decoded data) exits with code 1.
Both write a single JSON object containing ``code`` and ``message`` to stderr.
"""

from __future__ import annotations

import json
import sys
from typing import Any


class MixerError(Exception):
    """Base class for all expected tool errors."""

    exit_code = 1
    code = "PROCESSING_ERROR"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ValidationError(MixerError):
    """Raised when the project description fails validation."""

    exit_code = 2
    code = "VALIDATION_ERROR"


class FFmpegError(MixerError):
    """Raised when an ffmpeg/ffprobe subprocess fails."""

    exit_code = 1
    code = "FFMPEG_ERROR"


class ProcessError(MixerError):
    """Raised for IO errors or malformed decoded data."""

    exit_code = 1
    code = "PROCESSING_ERROR"


def emit_error(exc: BaseException) -> int:
    """Write a JSON error object to stderr and return the exit code."""
    if isinstance(exc, MixerError):
        payload: dict[str, Any] = {"code": exc.code, "message": str(exc)}
        exit_code = exc.exit_code
    else:
        payload = {
            "code": "PROCESSING_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 1
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()
    return exit_code
