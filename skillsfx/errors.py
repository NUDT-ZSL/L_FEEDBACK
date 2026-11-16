"""Stable business error codes and error serialization."""

from __future__ import annotations

from typing import Any


ERROR_CODES: dict[str, str] = {
    "SFX-E1001": "recipe file is missing, unreadable, or not valid JSON",
    "SFX-E1002": "recipe schema is missing a field or contains an invalid type",
    "SFX-E1003": "recipe value is outside the accepted range",
    "SFX-E2001": "source WAV is missing or unreadable",
    "SFX-E2002": "source WAV is malformed or uses an unsupported encoding",
    "SFX-E2003": "event offset is negative or outside the output timeline",
    "SFX-E2004": "source WAV cannot be mapped to the requested output channels",
    "SFX-E3001": "bus, compressor, normalization, or limiter parameter is invalid",
    "SFX-E4001": "WAV encoding or post-write header verification failed",
    "SFX-E4002": "generated audio contains a clipping sample",
    "SFX-E5001": "manifest replay consistency check failed",
    "SFX-E6001": "batch generation failed and staged output was removed",
    "SFX-E9001": "unexpected internal failure",
}


class SkillSFXError(Exception):
    """A diagnosable, user-facing business failure."""

    def __init__(
        self,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown business error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error_code": self.code,
            "category": ERROR_CODES[self.code],
            "message": self.message,
            "context": self.context,
        }
