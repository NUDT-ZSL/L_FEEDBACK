from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import ProcessingError
from .models import Manifest


class ProgressStore:
    def __init__(self, manifest: Manifest, temp_root: str | Path | None = None) -> None:
        root = (
            Path(temp_root).expanduser().resolve()
            if temp_root is not None
            else Path(tempfile.gettempdir()) / "broadcast_master"
        )
        key_payload = {
            "manifest": manifest.manifest_path.as_posix(),
            "raw": manifest.raw,
            "languages": [
                {"id": language.language_id, "path": language.path.as_posix()}
                for language in manifest.languages
            ],
            "video": manifest.video_path.as_posix(),
            "schema": 1,
        }
        canonical = json.dumps(
            key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.run_key = hashlib.sha256(canonical).hexdigest()
        self.run_dir = root / self.run_key[:24]
        self.staging_dir = self.run_dir / "staging"
        self.marker_path = self.run_dir / "progress.json"
        self._data: dict[str, Any] = {}
        self._resumed = False
        self._skipped_languages: list[str] = []

    @staticmethod
    def source_signature(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    def begin(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.marker_path.is_file():
            try:
                existing = json.loads(self.marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if (
                isinstance(existing, dict)
                and existing.get("schema") == 1
                and existing.get("run_key") == self.run_key
            ):
                self._data = existing
                self._resumed = True
        if not self._data:
            self._data = {
                "schema": 1,
                "run_key": self.run_key,
                "stage": "started",
                "completed_languages": {},
                "last_error": None,
            }
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.set_stage("probing_media")

    def set_stage(self, stage: str) -> None:
        self._data["stage"] = stage
        self._write()

    def completed_analysis(self, language_id: str, path: Path) -> dict | None:
        completed = self._data.get("completed_languages", {}).get(language_id)
        if not isinstance(completed, dict):
            return None
        try:
            signature = self.source_signature(path)
        except OSError:
            return None
        if completed.get("source_signature") != signature:
            return None
        analysis = completed.get("analysis")
        return analysis if isinstance(analysis, dict) else None

    def mark_language(self, language_id: str, path: Path, analysis: dict) -> None:
        signature = self.source_signature(path)
        completed = self._data.setdefault("completed_languages", {})
        completed[language_id] = {
            "source_signature": signature,
            "analysis": analysis,
        }
        self._skipped_languages = [
            item for item in self._skipped_languages if item != language_id
        ]
        self._write()

    def note_skipped(self, language_id: str) -> None:
        if language_id not in self._skipped_languages:
            self._skipped_languages.append(language_id)

    def fail(self, error: ProcessingError) -> None:
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)
        self._data["stage"] = "failed"
        self._data["last_error"] = {"code": error.code, "message": error.message}
        try:
            self._write()
        except OSError:
            pass

    def abort_business(self) -> None:
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)

    def succeed(self) -> None:
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)

    def report(self) -> dict:
        return {
            "resumed": self._resumed,
            "skipped_language_analyses": list(self._skipped_languages),
            "completed_language_count": len(
                self._data.get("completed_languages", {})
            ),
            "progress_marker": str(self.marker_path),
            "run_key": self.run_key,
            "staging_cleaned_on_failure": True,
        }

    def _write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.marker_path.with_suffix(".json.tmp")
        encoded = json.dumps(
            self._data, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.marker_path)

