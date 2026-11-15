"""Concurrent batch scheduling with per-file timeouts."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_ordered(
    items: list[T],
    fn: Callable[[T], R],
    timeout_factory: Callable[[T], R],
    jobs: int = 4,
    timeout_s: float = 30.0,
) -> list[R]:
    """Run fn over items with at most `jobs` workers, preserving order.

    Items exceeding timeout_s are replaced by timeout_factory(item); the
    underlying worker is abandoned (subprocesses it spawned are killed
    by their own timeouts) and never blocks the rest of the batch.
    """
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        deadline = {f: time.monotonic() + timeout_s for f in futures}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            for f in done:
                results[futures[f]] = f.result()
            now = time.monotonic()
            for f in list(pending):
                if now >= deadline[f]:
                    results[futures[f]] = timeout_factory(items[futures[f]])
                    pending.discard(f)
        for f in pending:  # pragma: no cover - defensive
            f.cancel()
    return [r for r in results]  # type: ignore[misc]
