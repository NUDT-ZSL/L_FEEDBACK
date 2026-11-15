"""Structured business errors used by every pipeline entry point."""

from __future__ import annotations

from typing import Any, Dict, Optional


INPUT_MISSING = "INPUT_MISSING"
TIMELINE_INVALID = "TIMELINE_INVALID"
TIMELINE_OUT_OF_RANGE = "TIMELINE_OUT_OF_RANGE"
STRETCH_UNREACHABLE = "STRETCH_UNREACHABLE"
AUDIO_UNSUPPORTED_FORMAT = "AUDIO_UNSUPPORTED_FORMAT"
BATCH_INVALID = "BATCH_INVALID"
PROCESSING_FAILED = "PROCESSING_FAILED"
BOUNDARY_MISALIGNED = "BOUNDARY_MISALIGNED"
LOUDNESS_UNREACHABLE = "LOUDNESS_UNREACHABLE"


EXIT_CODES = {
    INPUT_MISSING: 10,
    TIMELINE_INVALID: 11,
    TIMELINE_OUT_OF_RANGE: 12,
    STRETCH_UNREACHABLE: 13,
    AUDIO_UNSUPPORTED_FORMAT: 14,
    BATCH_INVALID: 15,
    BOUNDARY_MISALIGNED: 16,
    LOUDNESS_UNREACHABLE: 17,
    PROCESSING_FAILED: 20,
}


class RepairError(Exception):
    """A non-retryable, caller-visible processing error."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "status": "error",
            "error_code": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result

