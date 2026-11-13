"""Scheduler and worker orchestration.

Implements per-source BFS with strict three-way limits (per-source start
interval, per-source in-flight cap, global in-flight cap), exponential backoff
retries, a source-level circuit breaker, and the URL state machine. Workers
run in a bounded thread pool; all persistence goes through the locked Database.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .blobstore import BlobStore
from .config import Config, SourceConfig
from .db import Database
from .fetcher import FetchError, fetch_url
from .urlutils import extract_links, same_origin

RETRYABLE_PREFIXES = (
    "timeout",
    "connection_reset",
    "connection_error",
    "network_error",
)


@dataclass
class SourceState:
    cfg: SourceConfig
    queue: deque = field(default_factory=deque)
    inflight: int = 0
    consecutive_failures: int = 0
    open_until: float = 0.0
    circuit_open: bool = False
    finished: bool = False
    # Live counters; the JSON report is reconciled against the database instead
    # of trusting these alone.
    retry_count: int = 0
    # Monotonic rate gate: the earliest monotonic time the next request to
    # this source may be sent. Guards both dispatch and retry backoff.
    gate_ready_at: float = 0.0
    gate_lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire_send_slot(self) -> None:
        """Block until the per-source start interval has elapsed."""
        with self.gate_lock:
            now = time.monotonic()
            wait_for = self.gate_ready_at - now
            if wait_for > 0:
                time.sleep(wait_for)
            self.gate_ready_at = time.monotonic() + self.cfg.start_interval


class Crawler:
    def __init__(self, config: Config, exit_after: int | None = None):
        self.config = config
        self.exit_after = exit_after
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.blob_store = BlobStore(self.output)
        self.db = Database(self.output)
        self._global_slots = threading.BoundedSemaphore(config.global_concurrency)
        self._source_slots = {
            s.name: threading.BoundedSemaphore(s.per_source_concurrency)
            for s in config.sources
        }
        self.states = {s.name: SourceState(cfg=s) for s in config.sources}
        self._wakeup = threading.Event()
        self._fetch_counter = 0
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def startup(self) -> int:
        """Recovery steps: drop stale temp blobs, requeue orphaned inflight."""
        removed_tmp = self.blob_store.cleanup_temp()
        reset_rows = self.db.reset_inflight()
        for state in self.states.values():
            for url, depth in self.db.pending_urls(state.cfg.name):
                state.queue.append((url, depth))
            # Roots are (re)enqueued; existing roots are ignored by the DB.
            if self.db.count_seen(state.cfg.name) < self.config.max_pages:
                if self.db.enqueue_url(
                    state.cfg.name, state.cfg.root_url, 0
                ):
                    state.queue.append((state.cfg.root_url, 0))
        return removed_tmp, reset_rows

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        removed_tmp, reset_rows = self.startup()
        print(
            f"[startup] temp files removed={removed_tmp}, "
            f"inflight rows reset={reset_rows}"
        )
        with ThreadPoolExecutor(
            max_workers=self.config.global_concurrency
        ) as pool:
            active = dict(self.states)
            while active:
                now = time.monotonic()
                made_progress = False
                for state in list(active.values()):
                    if self._maybe_recover_circuit(state, now):
                        made_progress = True
                    if self._dispatchable(state, now):
                        url, depth = state.queue.popleft()
                        acquired_global = self._global_slots.acquire(
                            blocking=False
                        )
                        if not acquired_global:
                            state.queue.appendleft((url, depth))
                            continue
                        if not self._source_slots[state.cfg.name].acquire(
                            blocking=False
                        ):
                            self._global_slots.release()
                            state.queue.appendleft((url, depth))
                            continue
                        state.inflight += 1
                        self.db.mark_inflight(state.cfg.name, url)
                        pool.submit(self._worker, state, url, depth)
                        made_progress = True
                for state in list(active.values()):
                    if (
                        not state.queue
                        and state.inflight == 0
                    ):
                        state.finished = True
                for name, state in list(active.items()):
                    if state.finished:
                        del active[name]
                if not active:
                    break
                if not made_progress:
                    self._wakeup.wait(timeout=self._idle_wait(active))
                    self._wakeup.clear()

    def _idle_wait(self, active: dict) -> float:
        # Wake promptly at the next interval deadline or circuit cooldown.
        deadlines = []
        now = time.monotonic()
        for state in active.values():
            if state.queue and not state.circuit_open:
                deadlines.append(max(0.0, state.gate_ready_at - now))
            if state.circuit_open:
                deadlines.append(max(0.0, state.open_until - now))
        return min(deadlines) if deadlines else 0.05

    def _dispatchable(self, state: SourceState, now: float) -> bool:
        if state.finished or state.circuit_open or not state.queue:
            return False
        if state.inflight >= state.cfg.per_source_concurrency:
            return False
        if now < state.gate_ready_at:
            return False
        return True

    def _maybe_recover_circuit(self, state: SourceState, now: float) -> bool:
        if state.circuit_open and now >= state.open_until:
            state.circuit_open = False
            state.consecutive_failures = 0
            self.db.log_event(
                state.cfg.name, "circuit_recovered", "cooldown elapsed"
            )
            return True
        return False

    # ---------------------------------------------------------------- worker
    def _worker(self, state: SourceState, url: str, depth: int) -> None:
        try:
            try:
                self._process_with_retry(state, url, depth)
            except Exception as exc:  # defensive: never leak a slot silently
                error_type = f"unexpected:{type(exc).__name__}"
                self.db.log_event(
                    state.cfg.name, "worker_exception", f"{url} {exc!r}"
                )
                self._terminal_failure(state, url, error_type)
        finally:
            state.inflight -= 1
            self._source_slots[state.cfg.name].release()
            self._global_slots.release()
            self._wakeup.set()

    def _process_with_retry(self, state: SourceState, url: str, depth: int) -> None:
        attempt = 0
        while True:
            attempt += 1
            # Strict per-source pacing applies to every HTTP request,
            # including retries, so the start interval is never violated.
            state.acquire_send_slot()
            self.db.record_attempt(state.cfg.name, url)
            try:
                with self._counter_lock:
                    self._fetch_counter += 1
                    fetch_seq = self._fetch_counter
                result = fetch_url(
                    url=url,
                    root_url=state.cfg.root_url,
                    blob_store=self.blob_store,
                    timeout=self.config.timeout,
                    max_bytes=self.config.max_bytes,
                    user_agent=self.config.user_agent,
                    conditional_headers=self.db.get_cache_headers(
                        state.cfg.name, url
                    ),
                )
                if self.exit_after is not None and fetch_seq >= self.exit_after:
                    self._hard_exit(fetch_seq, state, url)
            except FetchError as exc:
                error_type = str(exc)
                if self._retryable(error_type) and attempt <= self.config.max_retries:
                    state.retry_count += 1
                    self._backoff(attempt)
                    continue
                self._terminal_failure(state, url, error_type)
                return

            if 500 <= result.http_status < 600:
                error_type = f"http_{result.http_status}"
                if attempt <= self.config.max_retries:
                    state.retry_count += 1
                    self._cleanup_temp(result.temp_path)
                    self._backoff(attempt)
                    continue
                self._terminal_failure(state, url, error_type)
                return

            if result.http_status == 304:
                # Not Modified: no body, no new version, no new content row.
                self.db.log_event(
                    state.cfg.name, "not_modified", url
                )
                self._reset_circuit_on_success(state)
                self.db.mark_done(state.cfg.name, url)
                return

            if 400 <= result.http_status < 500:
                # 4xx (400/404/...): immediate failure, consumes no retries.
                self._cleanup_temp(result.temp_path)
                self._terminal_failure(
                    state, url, f"http_{result.http_status}"
                )
                return

            if result.http_status != 200:
                self._cleanup_temp(result.temp_path)
                self._terminal_failure(
                    state, url, f"http_{result.http_status}"
                )
                return

            self._store_success(state, url, depth, result)
            self._reset_circuit_on_success(state)
            self.db.mark_done(state.cfg.name, url)
            return

    def _store_success(self, state, url, depth, result) -> None:
        # 1) Atomic blob commit; only then reference it from database rows.
        self.blob_store.write_atomic(result.content_hash, result.temp_path)
        # 2) Content row keyed by hash alone (cross-source dedup).
        new_content = self.db.insert_content(
            content_hash=result.content_hash,
            is_text=result.is_text,
            encoding_used=result.encoding_used,
            encoding_failed=result.encoding_failed,
            size_bytes=result.size_bytes,
            oversized=result.oversized,
        )
        # 3) Attribution: one row per (source, url, hash).
        self.db.add_attribution(
            source=state.cfg.name,
            url=url,
            content_hash=result.content_hash,
            http_status=result.http_status,
            last_modified=result.last_modified,
            etag=result.etag,
        )
        self.db.upsert_cache_headers(
            state.cfg.name, url, result.etag, result.last_modified
        )
        # 4) Version + current pointer; unchanged hash adds no version.
        self.db.add_version_if_changed(
            state.cfg.name, url, result.content_hash
        )
        # 5) BFS expansion of same-origin links.
        if result.is_text and depth < self.config.max_depth:
            html = self._read_text(result)
            for link in extract_links(html, url):
                if not same_origin(link, state.cfg.root_url):
                    continue
                if self.db.count_seen(state.cfg.name) >= self.config.max_pages:
                    break
                if self.db.enqueue_url(
                    state.cfg.name, link, depth + 1
                ):
                    state.queue.append((link, depth + 1))

    def _read_text(self, result) -> str:
        path = self.blob_store.path_for(result.content_hash)
        raw = path.read_bytes()
        if result.encoding_failed:
            return raw.decode("utf-8", errors="replace")
        return raw.decode(result.encoding_used or "utf-8", errors="replace")

    def _terminal_failure(self, state, url, error_type) -> None:
        self.db.mark_failed(state.cfg.name, url, error_type)
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.config.circuit_threshold:
            state.circuit_open = True
            state.open_until = time.monotonic() + self.config.circuit_cooldown
            self.db.log_event(
                state.cfg.name,
                "circuit_open",
                f"threshold={self.config.circuit_threshold} "
                f"cooldown={self.config.circuit_cooldown} last_error={error_type}",
            )

    def _reset_circuit_on_success(self, state) -> None:
        state.consecutive_failures = 0

    def _retryable(self, error_type: str) -> bool:
        return error_type.startswith(RETRYABLE_PREFIXES)

    def _backoff(self, attempt: int) -> None:
        # Deterministic exponential backoff (no random jitter): reproducible.
        delay = min(
            self.config.backoff_cap,
            self.config.backoff_base * (2 ** (attempt - 1)),
        )
        time.sleep(delay)

    def _cleanup_temp(self, temp_path: str | None) -> None:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _hard_exit(self, fetch_seq, state, url) -> None:
        # The URL row is already 'inflight'; flush WAL and exit without cleanup
        # so the recovery path can be exercised on the next launch.
        print(
            f"[exit-after] request #{fetch_seq} finished "
            f"({state.cfg.name} {url}); calling os._exit(0)"
            ,
            flush=True,
        )
        self.db.conn.execute("PRAGMA wal_checkpoint(FULL)")
        os._exit(0)
