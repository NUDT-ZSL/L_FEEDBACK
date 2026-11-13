"""Persistent, atomic job journal."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from . import codes
from .model import JobFailure
from .util import SCHEMA_VERSION, atomic_write_json, utc_now


class Journal:
    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root
        self.entries: dict[str, dict[str, Any]] = {}
        self.duplicate_ids_repaired = 0
        self._lock = threading.RLock()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = {}
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JobFailure(
                codes.JOURNAL_CORRUPT,
                f"cannot read journal {self.path}: {exc}",
                fatal=True,
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
            raise JobFailure(
                codes.JOURNAL_CORRUPT,
                "journal must be a JSON object with a jobs array",
                fatal=True,
            )
        entries: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(data["jobs"]):
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
                raise JobFailure(
                    codes.JOURNAL_CORRUPT,
                    f"journal entry {index} has no valid id",
                    fatal=True,
                )
            job_id = entry["id"]
            if job_id in entries:
                self.duplicate_ids_repaired += 1
            entries[job_id] = entry
        with self._lock:
            self.entries = entries

    def peek(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self.entries.get(job_id)

    def upsert(self, entry: dict[str, Any]) -> None:
        job_id = entry.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("journal entry id must be a non-empty string")
        with self._lock:
            self.entries[job_id] = entry

    def save(self) -> None:
        with self._lock:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_now(),
                "jobs": [self.entries[key] for key in sorted(self.entries)],
            }
        try:
            atomic_write_json(self.path, payload)
        except OSError as exc:
            raise JobFailure(
                codes.JOURNAL_WRITE_FAILED,
                f"cannot persist journal {self.path}: {exc}",
                fatal=True,
            ) from exc

    def cache_hit(self, job_id: str, fingerprint: str) -> Optional[dict[str, Any]]:
        with self._lock:
            entry = self.entries.get(job_id)
        if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
            return None
        output_rel = entry.get("output")
        output_size = entry.get("output_size")
        if not isinstance(output_rel, str) or not output_rel:
            return None
        if not isinstance(output_size, int) or output_size < 44:
            return None
        relative = Path(output_rel)
        if relative.is_absolute():
            return None
        try:
            output_path = (self.root / relative).resolve()
            output_path.relative_to(self.root.resolve())
            stat = output_path.stat()
        except (OSError, ValueError):
            return None
        if not output_path.is_file() or stat.st_size != output_size:
            return None
        return entry

    @staticmethod
    def output_relative(root: Path, output: Path) -> str:
        return output.resolve().relative_to(root.resolve()).as_posix()
