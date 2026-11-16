from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class BusinessError(Exception):
    """An invalid command or manifest that can be rejected before processing."""

    code: int
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


@dataclass
class ProcessingError(Exception):
    """A recoverable media/processing failure."""

    code: int
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


def error_object(code: int, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": int(code), "message": str(message)}
    if details:
        payload["details"] = details
    return payload


def emit_error(error: BusinessError | ProcessingError) -> None:
    import sys

    payload = error_object(error.code, error.message, error.details)
    sys.stderr.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stderr.flush()

