"""Same-volume atomic output with all-or-nothing commit semantics.

All artifacts are first written into a temporary directory created on the
same physical volume as the target (``os.replace`` is only guaranteed
atomic within one volume). On success every staged file is moved into
place and stale files from a previous manifest are removed. Any failure
removes the staging directory and leaves the target untouched.
"""

from __future__ import annotations

import os
import shutil
from typing import Callable

from .errors import IOFailure


def _shutil_rmtree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


class AtomicStage:
    """Context manager staging output files before an atomic commit."""

    def __init__(
        self,
        target_dir: str,
        inject_failure: Callable[[str], None] | None = None,
    ) -> None:
        self.target_dir = os.path.abspath(target_dir)
        self.staging_dir = self.target_dir + ".staging"
        self._inject_failure = inject_failure
        self._files: list[str] = []

    def __enter__(self) -> "AtomicStage":
        try:
            os.makedirs(self.target_dir, exist_ok=True)
            if os.path.isdir(self.staging_dir):
                _shutil_rmtree(self.staging_dir)
            os.makedirs(self.staging_dir)
        except OSError as exc:
            raise IOFailure(f"cannot create staging directory: {exc}") from exc
        return self

    def path(self, relative: str) -> str:
        """Absolute staging path for a target-relative output file."""
        rel = relative.replace("\\", "/").lstrip("/")
        destination = os.path.normpath(os.path.join(self.staging_dir, rel))
        staging_root = os.path.abspath(self.staging_dir)
        if os.path.commonpath([staging_root, os.path.abspath(destination)]) != staging_root:
            raise IOFailure(f"path escapes staging directory: {relative}")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        return destination

    def register(self, relative: str) -> None:
        self._files.append(relative.replace("\\", "/"))

    def _maybe_fail(self, relative: str) -> None:
        if self._inject_failure is not None:
            self._inject_failure(relative)

    def commit(self, previous_manifest: set[str] | None = None) -> list[str]:
        """Move every staged file into place and remove stale outputs."""
        committed: list[str] = []
        try:
            for relative in self._files:
                self._maybe_fail(relative)
                source = os.path.join(self.staging_dir, relative)
                destination = os.path.join(self.target_dir, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                os.replace(source, destination)
                committed.append(relative)
            if previous_manifest:
                new_manifest = set(self._files)
                for stale in previous_manifest - new_manifest:
                    stale_path = os.path.join(self.target_dir, stale)
                    if os.path.exists(stale_path):
                        os.remove(stale_path)
                        committed.append(f"removed:{stale}")
            _shutil_rmtree(self.staging_dir)
        except OSError as exc:
            self.abort()
            raise IOFailure(f"atomic commit failed: {exc}") from exc
        return committed

    def abort(self) -> None:
        _shutil_rmtree(self.staging_dir)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()


def read_manifest(path: str) -> set[str]:
    """Load the relative file list from a previous manifest, if present."""
    if not os.path.exists(path):
        return set()
    import json

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return set(data.get("files", []))
    except (OSError, ValueError) as exc:
        raise IOFailure(f"cannot read previous manifest: {exc}") from exc
