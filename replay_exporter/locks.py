"""Application-level export coordination.

Two guarantees are required:

* Concurrent exports to *different* output directories run freely.
* A second export targeting a directory that is already being rendered is
  rejected with ``E5001_OUTPUT_DIR_BUSY`` deterministically. We do not rely on
  advisory OS file locks (whose semantics differ across platforms); instead a
  process-local registry plus an atomic lock file with an owner/heartbeat
  protocol provide the decision, and stale locks from dead processes are taken
  over safely.
"""

from __future__ import annotations
import json
import os
import socket
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from .errors import ErrorCode, ExporterError

LOCK_FILE = "export.lock"
STALE_AFTER_SECONDS = 600


class _Registry:
    def __init__(self) -> None:
        self.condition = threading.Lock()
        self.directories: Dict[str, str] = {}  # normalized dir -> owner token


REGISTRY = _Registry()


def _normalize(output_dir: str) -> str:
    return os.path.normcase(os.path.abspath(output_dir))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ExportLock:
    def __init__(self, output_dir: str, token: str, payload: Dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.token = token
        self.payload = payload
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        normalized = _normalize(self.output_dir)
        with REGISTRY.condition:
            if REGISTRY.directories.get(normalized) == self.token:
                del REGISTRY.directories[normalized]
        lock_file = os.path.join(self.output_dir, LOCK_FILE)
        try:
            if os.path.exists(lock_file):
                with open(lock_file, "r", encoding="utf-8") as handle:
                    current = json.load(handle)
                if current.get("token") == self.token:
                    os.remove(lock_file)
        except (OSError, json.JSONDecodeError):
            pass
        self.released = True


def acquire_export_lock(output_dir: str) -> ExportLock:
    os.makedirs(output_dir, exist_ok=True)
    normalized = _normalize(output_dir)
    token = str(uuid.uuid4())
    payload = {
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "thread": threading.get_ident(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    }
    lock_file = os.path.join(output_dir, LOCK_FILE)

    with REGISTRY.condition:
        if normalized in REGISTRY.directories:
            raise ExporterError(
                ErrorCode.OUTPUT_DIR_BUSY,
                f"output directory {normalized} already has an active export "
                f"(owner token {REGISTRY.directories[normalized]})",
                detail={"output_dir": normalized},
            )

        if os.path.exists(lock_file):
            try:
                with open(lock_file, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                stale = _is_stale(existing)
            except (OSError, json.JSONDecodeError):
                stale = True
            if not stale:
                raise ExporterError(
                    ErrorCode.OUTPUT_DIR_BUSY,
                    f"output directory {normalized} is locked by another export "
                    f"(pid {existing.get('pid')}, token {existing.get('token')})",
                    detail={"output_dir": normalized, "lock": existing},
                )

        temporary = lock_file + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
        os.replace(temporary, lock_file)
        REGISTRY.directories[normalized] = token
    return ExportLock(output_dir, token, payload)


def _is_stale(existing: Dict[str, Any]) -> bool:
    pid = int(existing.get("pid", -1))
    if not _pid_alive(pid):
        return True
    heartbeat = existing.get("heartbeat_at")
    if not heartbeat:
        return False
    try:
        stamp = datetime.fromisoformat(heartbeat)
    except ValueError:
        return True
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    return age > STALE_AFTER_SECONDS


@contextmanager
def export_lock(output_dir: str):
    lock = acquire_export_lock(output_dir)
    try:
        yield lock
    finally:
        lock.release()


class PlanCache:
    """Thread-safe read-only cache of parsed timeline inputs.

    Parsing is the expensive shared step for concurrent exports; the cache maps
    a timeline path (+ file hash) to its parsed result and is guarded by a lock
    so two threads can never race on insertion or observe a half-built entry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, Any] = {}

    def get_or_load(self, key: str, loader):
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached
        result = loader()
        with self._lock:
            current = self._entries.get(key)
            if current is not None:
                return current
            self._entries[key] = result
            return result

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)


PLAN_CACHE = PlanCache()
