from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, TextIO


class ProgressReporter:
    """Serialized JSON Lines progress events backed by a monotonic clock."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._start_ns = time.monotonic_ns()
        self._last_ns = self._start_ns
        self._lock = threading.Lock()

    def emit(
        self,
        stage: str,
        language: str = "*",
        **fields: Any,
    ) -> None:
        with self._lock:
            now_ns = time.monotonic_ns()
            if now_ns < self._last_ns:
                now_ns = self._last_ns
            self._last_ns = now_ns
            elapsed_ms = (now_ns - self._start_ns) // 1_000_000
            payload: dict[str, Any] = {
                "event": "progress",
                "language": language,
                "stage": stage,
                "elapsed_ms": elapsed_ms,
                "monotonic_ns": now_ns,
            }
            payload.update(fields)
            self._write(payload)

    def summary(self, **fields: Any) -> None:
        with self._lock:
            now_ns = time.monotonic_ns()
            if now_ns < self._last_ns:
                now_ns = self._last_ns
            self._last_ns = now_ns
            payload: dict[str, Any] = {
                "event": "summary",
                "elapsed_ms": (now_ns - self._start_ns) // 1_000_000,
                "monotonic_ns": now_ns,
            }
            payload.update(fields)
            self._write(payload)

    def _write(self, payload: dict[str, Any]) -> None:
        self._stream.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    @property
    def elapsed_ms(self) -> int:
        return (time.monotonic_ns() - self._start_ns) // 1_000_000
