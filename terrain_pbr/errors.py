"""Error types and structured error serialization."""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base error carrying a stable business code and process exit code."""

    code = "E_INTERNAL_ERROR"
    exit_code = 1

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        if self.details:
            error["details"] = self.details
        return {"ok": False, "error": error}


class InputValidationError(PipelineError):
    """Invalid CLI arguments or scene JSON."""

    code = "E_INVALID_INPUT"
    exit_code = 2


class OutputError(PipelineError):
    """An output path could not be written or replaced."""

    code = "E_IO_ERROR"
    exit_code = 3
