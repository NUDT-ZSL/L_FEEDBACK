"""Runtime memory budget tracking for tile-based baking.

Only arrays explicitly allocated as *active working data* are counted;
memory-mapped heightfields are backed by the on-disk cache and are not
resident working memory. The guard enforces a hard ceiling so large
terrains never materialize more than the budget in RAM at once.
"""

from __future__ import annotations

import threading

from .errors import MemoryBudgetExceeded


class MemoryBudget:
    """Tracks total bytes of live arrays against a hard byte ceiling."""

    def __init__(self, limit_bytes: int) -> None:
        if limit_bytes <= 0:
            raise ValueError("memory limit must be positive")
        self.limit_bytes = int(limit_bytes)
        self._current = 0
        self.peak = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        return self._current

    def acquire(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        with self._lock:
            if self._current + nbytes > self.limit_bytes:
                raise MemoryBudgetExceeded(
                    f"memory budget exceeded: need {nbytes} bytes, "
                    f"live {self._current}, limit {self.limit_bytes}"
                )
            self._current += nbytes
            self.peak = max(self.peak, self._current)

    def release(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        with self._lock:
            self._current -= nbytes
            if self._current < 0:
                self._current = 0

    def snapshot(self) -> dict:
        return {
            "limit_bytes": self.limit_bytes,
            "peak_live_bytes": self.peak,
            "final_live_bytes": self._current,
        }
