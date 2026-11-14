"""Business error codes and the exception carrying a structured JSON payload."""

from __future__ import annotations

from typing import Any


class ErrorCode:
    MISSING_ASSET = "E_MISSING_ASSET"
    UNSUPPORTED_FORMAT = "E_UNSUPPORTED_FORMAT"
    OUT_OF_RANGE = "E_OUT_OF_RANGE"
    PEAK_LIMIT = "E_PEAK_LIMIT"
    INVALID_CONFIG = "E_INVALID_CONFIG"
    INTERNAL = "E_INTERNAL"


class MixerError(Exception):
    """An error that maps 1:1 to a business error code in the JSON report.

    ``details`` is JSON-serialisable and embedded verbatim in the report so
    that the failure cause can be audited without the WAV ever being written.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "error_code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload
