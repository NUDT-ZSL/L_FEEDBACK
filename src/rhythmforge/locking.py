"""Cross-process mutual exclusion for a build output directory.

On Windows an ``msvcrt``-style lock would be used; here the portable
:mod:`fcntl` implementation covers Linux/macOS while keeping the same
:class:`OutputLock` interface.  The lock is acquired non-blocking so a
second concurrent build fails immediately with exit code 7 instead of
silently corrupting artifacts.
"""

from __future__ import annotations

import os

from .errors import BuildError, ExitCode

try:  # Windows (msvcrt is available on the target platform).
    import msvcrt  # type: ignore

    _IS_WINDOWS = True
except ImportError:  # pragma: no cover - exercised on POSIX CI hosts.
    import fcntl

    _IS_WINDOWS = False


class OutputLock:
    """Context manager holding an exclusive lock on ``<output>/.build.lock``."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, ".build.lock")
        self._handle = None

    def __enter__(self) -> "OutputLock":
        self._handle = open(self.path, "a+b")
        try:
            if _IS_WINDOWS:
                # Lock one byte exclusively and non-blocking.
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise BuildError(
                ExitCode.LOCKED_OUTPUT,
                f"output directory is locked by another build: {os.path.dirname(self.path)}",
                {"lock_file": self.path},
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n".encode("ascii"))
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is None:
            return
        try:
            if _IS_WINDOWS:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
