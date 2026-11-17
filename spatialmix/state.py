"""Atomic JSON progress state for resumable rendering."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from .errors import (
    BusinessError,
    E_IO,
    E_STATE_INVALID,
    E_STATE_MISMATCH,
)

STATE_SCHEMA_VERSION = 1


def _fsync_parent(path: Path) -> None:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(str(path.parent), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Windows and some network file systems do not permit directory fsync.
        pass


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BusinessError(
            E_IO,
            f"Unable to create state directory: {destination.parent}",
            {"path": str(destination.parent), "reason": str(exc)},
        ) from exc
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".tmp-",
            dir=str(destination.parent),
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_parent(destination)
    except OSError as exc:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise BusinessError(
            E_IO,
            f"Unable to atomically write state file: {destination}",
            {"path": str(destination), "reason": str(exc)},
        ) from exc
    except BaseException:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise


def load_state(path: str | Path) -> Dict[str, Any]:
    state_path = Path(path)
    try:
        payload = state_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(
            E_STATE_INVALID,
            f"Unable to read resume state: {state_path}",
            {"path": str(state_path), "reason": str(exc)},
        ) from exc
    try:
        state = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BusinessError(
            E_STATE_INVALID,
            f"Resume state is not valid JSON: {state_path}",
            {"path": str(state_path), "reason": str(exc)},
        ) from exc
    if not isinstance(state, dict):
        raise BusinessError(
            E_STATE_INVALID,
            "Resume state root must be an object",
            {"path": str(state_path)},
        )
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise BusinessError(
            E_STATE_INVALID,
            "Resume state schema version is not supported",
            {
                "path": str(state_path),
                "schema_version": state.get("schema_version"),
                "expected": STATE_SCHEMA_VERSION,
            },
        )
    return state


def validate_state_identity(
    state: Mapping[str, Any],
    expected: Mapping[str, Any],
    state_path: str | Path,
) -> None:
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key, expected_value in expected.items():
        actual_value = state.get(key)
        if actual_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }
    if mismatches:
        raise BusinessError(
            E_STATE_MISMATCH,
            "Existing resume state does not match the current run parameters",
            {
                "state_path": str(state_path),
                "mismatches": mismatches,
            },
        )


def remove_path(path: str | Path) -> None:
    target = Path(path)
    try:
        if target.is_dir():
            for child in sorted(target.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            target.rmdir()
        elif target.exists() or target.is_symlink():
            target.unlink()
    except OSError:
        pass
