from __future__ import annotations

import json
import sys
import threading
from typing import Any


EXIT_OK = 0
EXIT_TIMELINE = 2
EXIT_EXISTS = 3
EXIT_PROCESSING = 4
EXIT_MUX = 5
EXIT_INTERRUPTED = 130

_ERROR_WRITE_LOCK = threading.Lock()


class LocalizerError(Exception):
    """An expected, machine-readable business error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "exit_code": self.exit_code,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class OperationCancelled(Exception):
    """Raised cooperatively after SIGINT/SIGBREAK or child-process termination."""


def emit_json_error(error: LocalizerError, stream: Any = None) -> None:
    stream = stream or sys.stderr
    line = (
        json.dumps(
            error.payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    with _ERROR_WRITE_LOCK:
        stream.write(line)
        stream.flush()
