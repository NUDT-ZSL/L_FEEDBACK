"""Job-level persistent state with atomic replacement and version guard."""

from __future__ import annotations

from pathlib import Path

from .util import atomic_write_json

STATE_FILENAME = "job_state.json"
STATE_VERSION = 1

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


class JobState:
    """File-backed per-animation state store.

    A stale state file (missing/older ``version``) is discarded on load so a
    new tool version never reuses incompatible contents. Writes are atomic
    (temp file + os.replace), so a hard kill can never leave a partial file.
    """

    def __init__(self, job_dir: str | Path):
        self.job_dir = Path(job_dir)
        self.path = self.job_dir / STATE_FILENAME
        self.version = STATE_VERSION
        self.animations: dict[str, dict] = {}

    @classmethod
    def load(cls, job_dir: str | Path) -> "JobState":
        state = cls(job_dir)
        if state.path.is_file():
            import json

            try:
                payload = json.loads(state.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Corrupt/partial state: start clean rather than trust it.
                payload = {}
            if payload.get("version") == STATE_VERSION and isinstance(payload.get("animations"), dict):
                state.animations = payload["animations"]
            # Version mismatch or unreadable content => empty store, and the
            # first save atomically replaces the obsolete file.
        return state

    def get(self, animation_id: str) -> dict | None:
        return self.animations.get(animation_id)

    def is_success(self, animation_id: str) -> bool:
        entry = self.animations.get(animation_id)
        return bool(entry and entry.get("status") == STATUS_SUCCESS)

    def mark(self, animation_id: str, status: str, **fields) -> None:
        entry = self.animations.get(animation_id, {})
        entry.update(fields)
        entry["status"] = status
        entry["attempts"] = int(entry.get("attempts", 0)) + (
            1 if status == STATUS_RUNNING and entry.get("status") != STATUS_RUNNING else 0
        )
        self.animations[animation_id] = entry
        self.flush()

    def record_success(self, animation_id: str, report_path: str, output_path: str) -> None:
        self.animations[animation_id] = {
            "status": STATUS_SUCCESS,
            "report_path": report_path,
            "output_path": output_path,
            "attempts": int(self.animations.get(animation_id, {}).get("attempts", 0)) + 1,
        }
        self.flush()

    def record_failure(self, animation_id: str, error: dict) -> None:
        prev = self.animations.get(animation_id, {})
        self.animations[animation_id] = {
            "status": STATUS_FAILED,
            "attempts": int(prev.get("attempts", 0)) + 1,
            "error": error,
        }
        self.flush()

    def flush(self) -> None:
        payload = {"version": self.version, "animations": self.animations}
        atomic_write_json(self.path, payload)
