from __future__ import annotations

import os
import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable

from localizer.errors import EXIT_EXISTS, LocalizerError


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


class OutputManager:
    """Path-granular locks plus same-directory atomic temp-file commits."""

    def __init__(self, *, no_overwrite: bool = False) -> None:
        self.no_overwrite = no_overwrite
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._temps: dict[Path, str] = {}
        self._committed: list[Path] = []

    def preflight(self, paths: list[Path]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        conflicts: list[str] = []
        for path in paths:
            key = _path_key(path)
            if key in seen:
                duplicates.append(os.fspath(path))
            seen.add(key)
            if path.exists():
                conflicts.append(os.fspath(path))
        if duplicates:
            raise LocalizerError(
                "E_EXISTS",
                "multiple planned outputs resolve to the same path",
                exit_code=EXIT_EXISTS,
                details={"conflicts": sorted(set(duplicates))},
            )
        if self.no_overwrite and conflicts:
            raise LocalizerError(
                "E_EXISTS",
                "output files already exist and --no-overwrite was specified",
                exit_code=EXIT_EXISTS,
                details={"conflicts": sorted(set(conflicts))},
            )

    def _path_lock(self, path: Path) -> threading.Lock:
        key = _path_key(path)
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _new_temp(self, destination: Path, owner: str) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        safe_owner = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner).strip("._") or "global"
        prefix = f".{destination.name}.{safe_owner}."
        fd, raw_path = tempfile.mkstemp(
            prefix=prefix,
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(fd)
        path = Path(raw_path)
        with self._guard:
            self._temps[path] = owner
        return path

    def write_atomic(
        self,
        destination: Path,
        writer: Callable[[Path], None],
        *,
        owner: str,
    ) -> None:
        with self._path_lock(destination):
            if self.no_overwrite and destination.exists():
                raise LocalizerError(
                    "E_EXISTS",
                    "output file already exists",
                    exit_code=EXIT_EXISTS,
                    details={"conflicts": [os.fspath(destination)]},
                )
            temp = self._new_temp(destination, owner)
            try:
                writer(temp)
                self._commit_locked(temp, destination)
            finally:
                self._unlink_temp(temp)

    def reserve_temp(self, destination: Path, *, owner: str) -> Path:
        return self._new_temp(destination, owner)

    def commit_temp(self, temp: Path, destination: Path) -> None:
        with self._path_lock(destination):
            if self.no_overwrite and destination.exists():
                self._unlink_temp(temp)
                raise LocalizerError(
                    "E_EXISTS",
                    "output file already exists",
                    exit_code=EXIT_EXISTS,
                    details={"conflicts": [os.fspath(destination)]},
                )
            self._commit_locked(temp, destination)
        self._unlink_temp(temp)

    def _commit_locked(self, temp: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, destination)
        with self._guard:
            self._temps.pop(temp, None)
            self._committed.append(destination)

    def _unlink_temp(self, path: Path) -> None:
        with self._guard:
            self._temps.pop(path, None)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def cleanup_uncommitted(self) -> None:
        with self._guard:
            paths = list(self._temps)
        for path in paths:
            self._unlink_temp(path)

    def registered_temps(self) -> list[Path]:
        with self._guard:
            return list(self._temps)

    def committed_outputs(self) -> list[Path]:
        with self._guard:
            return list(self._committed)

    def unique_work_name(self, base: str) -> str:
        return f"{base}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
