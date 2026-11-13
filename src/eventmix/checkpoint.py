"""Persistent per-task checkpoints with input and artifact verification."""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .locking import exclusive_file_lock
from .util import atomic_write_bytes
from .version import CHECKPOINT_SCHEMA, PLAYBACK_SCHEMA, TOOL_VERSION


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _artifacts_match(record: Mapping[str, Any]) -> bool:
    from .util import sha256_file

    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return False
    for key in ("wav", "timeline"):
        item = outputs.get(key)
        if not isinstance(item, dict):
            return False
        path = item.get("path")
        if not isinstance(path, str) or not path:
            return False
        try:
            stat = Path(path).stat()
        except OSError:
            return False
        if int(item.get("size", -1)) != stat.st_size:
            return False
        digest = item.get("sha256")
        if not isinstance(digest, str) or sha256_file(path) != digest:
            return False
    return True


class CheckpointStore:
    def __init__(self, path: Path | str, *, resume: bool = True):
        self.path = Path(path)
        self.resume = resume
        self.reset_reason: str | None = None
        self._thread_lock = threading.RLock()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "tool_version": TOOL_VERSION,
            "playback_schema": PLAYBACK_SCHEMA,
            "jobs": {},
            "updated_at": None,
        }

    def load(self) -> dict[str, Any]:
        with self._thread_lock:
            if not self.resume:
                self.reset_reason = "resume disabled; checkpoint ignored"
                return self._empty()
            if not self.path.exists():
                return self._empty()
            with exclusive_file_lock(str(self.path) + ".lock"):
                value = _read_json(self.path)
            if value is None:
                self.reset_reason = "checkpoint is unreadable; starting fresh"
                return self._empty()
            if value.get("schema_version") != CHECKPOINT_SCHEMA:
                self.reset_reason = "checkpoint schema mismatch; starting fresh"
                return self._empty()
            if value.get("tool_version") != TOOL_VERSION:
                self.reset_reason = "checkpoint tool version mismatch; starting fresh"
                return self._empty()
            if value.get("playback_schema") != PLAYBACK_SCHEMA:
                self.reset_reason = "checkpoint artifact schema mismatch; starting fresh"
                return self._empty()
            if not isinstance(value.get("jobs"), dict):
                self.reset_reason = "checkpoint jobs table is invalid; starting fresh"
                return self._empty()
            return value

    def is_reusable(self, record: Mapping[str, Any] | None, fingerprint: str) -> bool:
        if not isinstance(record, Mapping):
            return False
        artifact_version = record.get("artifact_version")
        if not isinstance(artifact_version, Mapping):
            return False
        return (
            record.get("status") == "complete"
            and record.get("fingerprint") == fingerprint
            and artifact_version.get("tool_version") == TOOL_VERSION
            and artifact_version.get("playback_schema") == PLAYBACK_SCHEMA
            and _artifacts_match(record)
        )

    def update(self, job_id: str, record: Mapping[str, Any]) -> None:
        with self._thread_lock:
            with exclusive_file_lock(str(self.path) + ".lock"):
                current = _read_json(self.path) if self.path.exists() else None
                if (
                    not isinstance(current, dict)
                    or current.get("schema_version") != CHECKPOINT_SCHEMA
                    or current.get("tool_version") != TOOL_VERSION
                ):
                    current = self._empty()
                jobs = current.setdefault("jobs", {})
                if not isinstance(jobs, dict):
                    jobs = {}
                    current["jobs"] = jobs
                jobs[job_id] = dict(record)
                current["updated_at"] = datetime.now(timezone.utc).isoformat()
                data = (
                    json.dumps(
                        current,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                atomic_write_bytes(self.path, data)

    def completed_record(
        self,
        *,
        fingerprint: str,
        outputs: Mapping[str, Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "complete",
            "fingerprint": fingerprint,
            "artifact_version": {
                "tool_version": TOOL_VERSION,
                "playback_schema": PLAYBACK_SCHEMA,
                "wav_format": "WAVE_FORMAT_EXTENSIBLE_PCM",
                "checkpoint_schema": CHECKPOINT_SCHEMA,
            },
            "outputs": copy.deepcopy(dict(outputs)),
            "summary": copy.deepcopy(dict(summary)),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def failed_record(self, *, fingerprint: str, error: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "failed",
            "fingerprint": fingerprint,
            "artifact_version": {
                "tool_version": TOOL_VERSION,
                "playback_schema": PLAYBACK_SCHEMA,
            },
            "error": copy.deepcopy(dict(error)),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

