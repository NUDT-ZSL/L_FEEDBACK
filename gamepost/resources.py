from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from .errors import CancelledError


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError()


class MemoryQuota:
    """A blocking shared memory quota.

    Jobs wait at the boundary when the quota is exhausted. No job is rejected
    merely because another job is using the configured resource pool.
    """

    def __init__(self, limit_bytes: int) -> None:
        if limit_bytes <= 0:
            raise ValueError("memory limit must be positive")
        self.limit_bytes = int(limit_bytes)
        self._used_bytes = 0
        self._condition = threading.Condition()

    @property
    def used_bytes(self) -> int:
        with self._condition:
            return self._used_bytes

    def acquire(
        self,
        required_bytes: int,
        token: CancellationToken | None = None,
    ) -> int:
        required_bytes = max(0, int(required_bytes))
        reserved_bytes = min(required_bytes, self.limit_bytes)
        with self._condition:
            while self._used_bytes + reserved_bytes > self.limit_bytes:
                if token is not None:
                    token.raise_if_cancelled()
                self._condition.wait(timeout=0.05)
            self._used_bytes += reserved_bytes
            return reserved_bytes

    def release(self, required_bytes: int) -> None:
        required_bytes = max(0, int(required_bytes))
        with self._condition:
            self._used_bytes = max(0, self._used_bytes - required_bytes)
            self._condition.notify_all()

    @contextmanager
    def lease(
        self,
        required_bytes: int,
        token: CancellationToken | None = None,
    ):
        reserved = self.acquire(required_bytes, token)
        try:
            yield reserved
        finally:
            self.release(reserved)


class CancelFileWatcher:
    def __init__(
        self,
        path: str,
        token: CancellationToken,
        interval_seconds: float = 0.05,
    ) -> None:
        self.path = str(path)
        self.token = token
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="gamepost-cancel-watcher",
            daemon=True,
        )

    def start(self) -> "CancelFileWatcher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.interval_seconds * 4))

    def _loop(self) -> None:
        import os

        while not self._stop.is_set():
            if os.path.exists(self.path):
                self.token.cancel()
                return
            time.sleep(self.interval_seconds)

    def __enter__(self) -> "CancelFileWatcher":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def estimate_memory_bytes(*file_sizes: int) -> int:
    # Int16 input expands to several float64 working buffers. The estimate is a
    # conservative admission figure, not an exact allocator measurement.
    return 16 * 1024 * 1024 + 24 * sum(max(0, int(size)) for size in file_sizes)

