"""Checksummed, atomically written resumable pipeline state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import StateError, StateIntegrityError
from .fftools import atomic_write_json

STATE_FORMAT = "dialogue-sync-state"
STATE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class CheckpointStore:
    """Manage state.json plus hash-validated intermediate WAV fragments."""

    def __init__(self, work_dir: Path, manifest_hash: str) -> None:
        self.work_dir = work_dir
        self.fragments_dir = work_dir / "fragments"
        self.raw_dir = work_dir / "raw"
        self.state_path = work_dir / "state.json"
        self.snapshot_path = work_dir / "report_snapshot.json"
        self.manifest_hash = manifest_hash
        self.state: dict[str, Any] = self._new_state()
        self.load_warnings: list[str] = []

    def _new_state(self) -> dict[str, Any]:
        now = utc_now()
        return {
            "format": STATE_FORMAT,
            "version": STATE_VERSION,
            "manifest_hash": self.manifest_hash,
            "phase": "processing",
            "created_at": now,
            "updated_at": now,
            "entry_reports": {},
            "fragments": {},
            "mix": None,
            "normalized": None,
            "loudness": None,
            "packaging": None,
            "warnings": [],
        }

    def reset(self) -> None:
        for path in (self.state_path, self.snapshot_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise StateError(f"cannot remove old state {path}: {exc}") from exc
        for directory in (self.fragments_dir, self.raw_dir):
            if directory.exists():
                shutil.rmtree(directory)
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._new_state()
        self.load_warnings = []

    def load(self, *, force: bool = False) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if force:
            self.reset()
            self.save()
            return
        if not self.state_path.exists():
            self.save()
            return
        try:
            raw = self.state_path.read_bytes()
        except OSError as exc:
            raise StateIntegrityError(f"cannot read state {self.state_path}: {exc}") from exc
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateIntegrityError(
                f"state file {self.state_path} is truncated or invalid JSON; "
                "run with --force to rebuild"
            ) from exc
        if not isinstance(document, dict):
            raise StateIntegrityError("state document must be an object")
        if document.get("format") != STATE_FORMAT:
            raise StateIntegrityError("unrecognized state format")
        checksum = document.get("sha256")
        payload = document.get("payload")
        if not isinstance(checksum, str) or not isinstance(payload, dict):
            raise StateIntegrityError("state file lacks a checksummed payload")
        actual = sha256_bytes(_canonical_payload(payload))
        if actual != checksum:
            raise StateIntegrityError(
                "state checksum mismatch; the checkpoint may have been truncated or "
                "modified (use --force to rebuild)"
            )
        if payload.get("version") != STATE_VERSION:
            raise StateError(
                f"unsupported state version {payload.get('version')}; use --force"
            )
        if payload.get("manifest_hash") != self.manifest_hash:
            raise StateError(
                "state belongs to different input/settings content; "
                "use --force to rebuild the checkpoint"
            )
        self.state = payload
        self._validate_fragments()

    def _validate_fragments(self) -> None:
        fragments = self.state.get("fragments", {})
        if not isinstance(fragments, dict):
            raise StateIntegrityError("state fragments entry is invalid")
        invalid: list[str] = []
        for key, record in list(fragments.items()):
            if not isinstance(record, dict):
                invalid.append(str(key))
                continue
            relative = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                invalid.append(str(key))
                continue
            fragment = self.work_dir / relative
            if not fragment.is_file():
                invalid.append(str(key))
                continue
            try:
                actual = sha256_file(fragment)
            except OSError:
                invalid.append(str(key))
                continue
            if actual != expected:
                invalid.append(str(key))
        for key in invalid:
            fragments.pop(key, None)
            self.state.get("entry_reports", {}).pop(key, None)
            self.load_warnings.append(
                f"fragment for entry {key} was missing or failed checksum validation; "
                "that entry will be reprocessed"
            )
        if invalid:
            self.save()

    def fragment_path(self, ordinal: int, digest_input: str) -> Path:
        suffix = sha256_bytes(digest_input.encode("utf-8"))[:10]
        return self.fragments_dir / f"entry_{ordinal:04d}_{suffix}.wav"

    def relative_fragment_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.work_dir).as_posix()
        except ValueError as exc:
            raise StateError(f"fragment path is outside work directory: {path}") from exc

    def fragment_absolute_path(self, relative: str) -> Path:
        candidate = (self.work_dir / relative).resolve()
        try:
            candidate.relative_to(self.work_dir.resolve())
        except ValueError as exc:
            raise StateIntegrityError(f"state fragment escapes work directory: {relative}") from exc
        return candidate

    def record_fragment(
        self,
        ordinal: int,
        path: Path,
        *,
        samples: int,
        duration_ms: float,
        weight: float,
    ) -> None:
        self.state.setdefault("fragments", {})[str(ordinal)] = {
            "path": self.relative_fragment_path(path),
            "sha256": sha256_file(path),
            "samples": samples,
            "duration_ms": round(duration_ms, 3),
            "weight": weight,
        }

    def get_fragment(self, ordinal: int) -> dict[str, Any] | None:
        value = self.state.get("fragments", {}).get(str(ordinal))
        return value if isinstance(value, dict) else None

    def record_report(self, ordinal: int, report: dict[str, Any]) -> None:
        self.state.setdefault("entry_reports", {})[str(ordinal)] = report

    def has_report(self, ordinal: int) -> bool:
        return str(ordinal) in self.state.get("entry_reports", {})

    def get_report(self, ordinal: int) -> dict[str, Any] | None:
        value = self.state.get("entry_reports", {}).get(str(ordinal))
        return value if isinstance(value, dict) else None

    def snapshot(self) -> dict[str, Any]:
        reports = self.state.get("entry_reports", {})
        return {
            "status": "in_progress",
            "phase": self.state.get("phase"),
            "manifest_hash": self.manifest_hash,
            "updated_at": self.state.get("updated_at"),
            "entries": [
                reports[key]
                for key in sorted(reports, key=lambda value: int(value))
                if isinstance(reports[key], dict)
            ],
            "warnings": self.state.get("warnings", []) + self.load_warnings,
        }

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        payload = self.state
        envelope = {
            "format": STATE_FORMAT,
            "version": STATE_VERSION,
            "sha256": sha256_bytes(_canonical_payload(payload)),
            "payload": payload,
        }
        try:
            atomic_write_json(self.state_path, envelope)
            atomic_write_json(self.snapshot_path, self.snapshot())
            _fsync_directory(self.work_dir)
        except OSError as exc:
            raise StateError(f"cannot persist checkpoint state: {exc}") from exc


def inspect_state(work_dir: Path) -> dict[str, Any]:
    state_path = work_dir / "state.json"
    if not state_path.is_file():
        raise StateError(f"no state file found at {state_path}")
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateIntegrityError(f"invalid state file: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("payload"), dict):
        raise StateIntegrityError("state envelope is invalid")
    if document.get("format") != STATE_FORMAT:
        raise StateIntegrityError("unrecognized state format")
    checksum = document.get("sha256")
    payload = document["payload"]
    if checksum != sha256_bytes(_canonical_payload(payload)):
        raise StateIntegrityError("state checksum mismatch")
    if payload.get("version") != STATE_VERSION:
        raise StateIntegrityError("unsupported state version")
    fragments = payload.get("fragments", {})
    fragment_status: list[dict[str, Any]] = []
    for key, record in sorted(fragments.items(), key=lambda item: int(item[0])):
        path = work_dir / str(record.get("path", ""))
        exists = path.is_file()
        checksum_ok = exists and sha256_file(path) == record.get("sha256")
        fragment_status.append(
            {
                "entry": int(key),
                "path": str(record.get("path")),
                "exists": exists,
                "checksum_ok": checksum_ok,
            }
        )
    fragments_valid = all(
        item["exists"] and item["checksum_ok"] for item in fragment_status
    )
    return {
        "valid": fragments_valid,
        "envelope_valid": True,
        "fragments_valid": fragments_valid,
        "phase": payload.get("phase"),
        "manifest_hash": payload.get("manifest_hash"),
        "entry_reports": len(payload.get("entry_reports", {})),
        "fragments": fragment_status,
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)

