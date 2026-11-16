"""Business errors used by the command-line interface."""

from __future__ import annotations

from typing import Any, Dict, Optional


class LipAlignError(Exception):
    """An expected, user-facing failure with a stable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        exit_code: int = 2,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


def error_payload(error: LipAlignError) -> Dict[str, Any]:
    """Return the single-line JSON object written to stderr."""

    payload = error.as_dict()
    payload["ok"] = False
    return payload

