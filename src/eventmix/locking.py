"""Small cross-platform advisory file locks."""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Iterator

from .errors import ConfigError


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def exclusive_file_lock(
    path: Path | str, *, timeout_seconds: float = 600.0
) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    if os.name == "nt":
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\x00")
            handle.flush()
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while True:
            if _try_lock(handle):
                acquired = True
                break
            if time.monotonic() >= deadline:
                raise ConfigError(f"timed out waiting for file lock {path}")
            time.sleep(0.05)
        yield
    finally:
        if acquired:
            _unlock(handle)
        handle.close()


@contextlib.contextmanager
def exclusive_output_locks(paths: list[Path]) -> Iterator[None]:
    ordered = sorted(
        {str(path.resolve()) + ".eventmix.lock" for path in paths},
        key=os.path.normcase,
    )
    with contextlib.ExitStack() as stack:
        for lock_path in ordered:
            stack.enter_context(exclusive_file_lock(lock_path))
        yield

