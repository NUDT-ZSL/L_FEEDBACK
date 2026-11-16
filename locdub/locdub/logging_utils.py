"""Deterministic, dependency-free structured logger.

Errors and warnings are both printed to the console and collected in memory
so they can be embedded into the per-language JSON report. A separate error
log file is written for every run.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """Thread-safe logger that mirrors everything into an error log file."""

    def __init__(self, log_path: Optional[Path] = None, quiet: bool = False) -> None:
        self._lock = threading.Lock()
        self.records: List[Dict[str, Any]] = []
        self.log_path = Path(log_path) if log_path else None
        self.quiet = quiet
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def _emit(self, level: str, message: str, **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": level,
            "message": message,
        }
        record.update(fields)
        with self._lock:
            self.records.append(record)
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not self.quiet:
                prefix = "[ERROR]" if level == "error" else "[WARN ]"
                detail = ""
                if fields:
                    detail = " " + " ".join(f"{k}={v}" for k, v in fields.items())
                print(f"{prefix} {message}{detail}", file=sys.stderr)
        return record

    def error(self, message: str, **fields: Any) -> Dict[str, Any]:
        return self._emit("error", message, **fields)

    def warning(self, message: str, **fields: Any) -> Dict[str, Any]:
        return self._emit("warning", message, **fields)

    def info(self, message: str) -> None:
        if not self.quiet:
            with self._lock:
                print(message)

    def records_for(self, language: Optional[str] = None,
                    segment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for record in self.records:
            if language is not None and record.get("language") != language:
                continue
            if segment_id is not None and record.get("segment_id") != segment_id:
                continue
            out.append(record)
        return out
