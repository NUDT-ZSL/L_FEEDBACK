"""Cross-thread limits for open WAV handles and transient audio buffers."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass

from .errors import ConfigError


class ResourceGovernor:
    """A shared budgeting object used by every worker in a batch.

    File handles and memory are acquired independently.  A renderer never holds
    all input WAV files open at once; each chunk read opens, reads, and closes a
    file while accounting against the global handle budget.
    """

    def __init__(self, max_open_wav: int = 8, max_memory_bytes: int = 256 * 1024 * 1024):
        if max_open_wav < 1:
            raise ConfigError("max_open_wav must be >= 1")
        if max_memory_bytes < 4 * 1024 * 1024:
            raise ConfigError("max_memory_bytes must be >= 4 MiB")
        self.max_open_wav = int(max_open_wav)
        self.max_memory_bytes = int(max_memory_bytes)
        self._wav_slots = threading.BoundedSemaphore(self.max_open_wav)
        self._memory_lock = threading.Condition()
        self._memory_used = 0

    @contextmanager
    def wav_handle(self):
        with self._wav_slots:
            yield

    @contextmanager
    def memory(self, nbytes: int):
        nbytes = max(0, int(nbytes))
        if nbytes > self.max_memory_bytes:
            raise ConfigError(
                f"single buffer request {nbytes} exceeds max_memory_bytes={self.max_memory_bytes}"
            )
        with self._memory_lock:
            while self._memory_used + nbytes > self.max_memory_bytes:
                self._memory_lock.wait()
            self._memory_used += nbytes
        try:
            yield
        finally:
            with self._memory_lock:
                self._memory_used -= nbytes
                self._memory_lock.notify_all()

    @property
    def memory_used(self) -> int:
        with self._memory_lock:
            return self._memory_used


@dataclass(frozen=True)
class RenderLimits:
    chunk_frames: int = 8192
    max_open_wav: int = 8
    max_memory_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.chunk_frames < 1:
            raise ConfigError("chunk_frames must be >= 1")
        if self.max_open_wav < 1:
            raise ConfigError("max_open_wav must be >= 1")
        if self.max_memory_bytes < 4 * 1024 * 1024:
            raise ConfigError("max_memory_bytes must be >= 4 MiB")

