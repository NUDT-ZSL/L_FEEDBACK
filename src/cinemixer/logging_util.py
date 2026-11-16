"""Structured JSON-line logging.

Log records are emitted as one JSON object per line (``application/json-seq``
style) to stderr, or to an explicit log file. Warnings and errors always carry
an error/warning code and the related track/event identifiers when available.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, TextIO


class StructuredLogger:
    def __init__(self, stream: TextIO | None = None):
        self._stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()

    def emit(self, level: str, code: str, message: str, **fields: Any) -> None:
        record: dict[str, Any] = {"level": level, "code": code, "message": message}
        for key, value in fields.items():
            if value is not None:
                record[key] = value
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def info(self, message: str, code: str = "I_INFO", **fields: Any) -> None:
        self.emit("INFO", code, message, **fields)

    def warning(self, code: str, message: str, **fields: Any) -> None:
        self.emit("WARNING", code, message, **fields)

    def error(self, code: str, message: str, **fields: Any) -> None:
        self.emit("ERROR", code, message, **fields)


# Shared default instance writing to stderr.
logger = StructuredLogger()
