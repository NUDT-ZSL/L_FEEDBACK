"""Concurrency-bounded WAV loading with retry and injectable failures."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .config import Config
from .errors import ErrorCode
from .hashing import sha256_file


class AssetError(Exception):
    """Raised internally for asset failures; carries a business code."""

    def __init__(self, code: str, message: str, *, path: str, retryable: bool):
        super().__init__(message)
        self.code = code
        self.path = path
        self.retryable = retryable


@dataclass
class LoadedClip:
    """A validated in-memory mono clip plus its content identity."""

    path: str
    samples: np.ndarray
    sample_rate: int
    frames: int
    content_hash: str
    source_name: str


class FailureInjector:
    """Deterministic failure injection used to exercise retry/failure paths.

    Three independent sets model the required injected failure modes:

    * ``missing``  -> file treated as absent (``E_MISSING_ASSET``)
    * ``corrupt``  -> file treated as unreadable (``E_CORRUPT_ASSET``)
    * ``transient``-> temporary I/O error that succeeds once retried
      (``E_LOAD_FAILED`` while attempts remain)
    """

    def __init__(
        self,
        missing: set[str] | None = None,
        corrupt: set[str] | None = None,
        transient: set[str] | None = None,
        transient_attempts: int = 1,
    ) -> None:
        self.missing = set(missing or [])
        self.corrupt = set(corrupt or [])
        self.transient = set(transient or [])
        self.transient_attempts = transient_attempts
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = {}

    def is_missing(self, path: str) -> bool:
        return path in self.missing

    def is_corrupt(self, path: str) -> bool:
        return path in self.corrupt

    def consume_transient(self, path: str) -> bool:
        """Return True while the configured transient budget is unspent."""
        with self._lock:
            count = self._attempts.get(path, 0)
            self._attempts[path] = count + 1
            return path in self.transient and count < self.transient_attempts

    @property
    def attempts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._attempts)


class AssetStore:
    """Loads WAV clips through a bounded thread pool.

    Successful clips are cached by path so repeated events share one decode.
    Failures never touch the cache and therefore cannot corrupt a completed
    block or a snapshot written by another language.
    """

    def __init__(self, config: Config, injector: FailureInjector | None = None) -> None:
        self.config = config
        self.injector = injector or FailureInjector()
        self._cache: dict[str, LoadedClip] = {}
        self._lock = threading.Lock()

    def asset_path(self, assets_dir: str, voice_key: str, lang: str) -> str:
        return os.path.join(assets_dir, f"{voice_key}_{lang}.wav")

    def required_paths(
        self, assets_dir: str, pairs: set[tuple[str, str]]
    ) -> dict[str, tuple[str, str]]:
        return {
            self.asset_path(assets_dir, voice_key, lang): (voice_key, lang)
            for voice_key, lang in sorted(pairs)
        }

    def load_many(self, paths: list[str]) -> tuple[dict[str, LoadedClip], list[AssetError]]:
        """Load distinct paths in parallel; collect every terminal failure."""
        unique_paths = sorted(set(paths))
        results: dict[str, LoadedClip] = {}
        failures: list[AssetError] = []

        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as pool:
            future_map = {
                pool.submit(self.load_with_retry, path): path for path in unique_paths
            }
            for future in as_completed(future_map):
                path = future_map[future]
                try:
                    results[path] = future.result()
                except AssetError as error:
                    failures.append(error)

        # Deterministic ordering for reports.
        failures.sort(key=lambda error: error.path)
        return results, failures

    def load_with_retry(self, path: str) -> LoadedClip:
        """Load one path, retrying only retryable errors up to ``max_retries``."""
        last_error: AssetError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._load_once(path)
            except AssetError as error:
                last_error = error
                if not error.retryable or attempt >= self.config.max_retries:
                    raise
        # Unreachable, but keeps type checkers honest.
        assert last_error is not None
        raise last_error

    def _load_once(self, path: str) -> LoadedClip:
        injector = self.injector

        with self._lock:
            cached = self._cache.get(path)
        if cached is not None:
            return cached

        if injector.is_missing(path) or not os.path.exists(path):
            # Missing assets are not transient: retrying cannot create them.
            raise AssetError(
                ErrorCode.E_MISSING_ASSET,
                f"asset not found: {os.path.basename(path)}",
                path=path,
                retryable=False,
            )

        if injector.consume_transient(path):
            raise AssetError(
                ErrorCode.E_LOAD_FAILED,
                "injected transient read failure",
                path=path,
                retryable=True,
            )

        if injector.is_corrupt(path):
            raise AssetError(
                ErrorCode.E_CORRUPT_ASSET,
                "injected corrupt asset",
                path=path,
                retryable=False,
            )

        try:
            data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        except Exception as error:  # libsndfile raises varied exceptions
            raise AssetError(
                ErrorCode.E_CORRUPT_ASSET,
                f"unreadable WAV: {error}",
                path=path,
                retryable=False,
            ) from error

        if data.ndim != 1:
            raise AssetError(
                ErrorCode.E_CORRUPT_ASSET,
                f"expected mono clip, got {data.ndim} dimensions",
                path=path,
                retryable=False,
            )
        if sample_rate != self.config.sample_rate:
            raise AssetError(
                ErrorCode.E_CORRUPT_ASSET,
                f"sample rate {sample_rate} != {self.config.sample_rate}",
                path=path,
                retryable=False,
            )
        if data.size == 0:
            raise AssetError(
                ErrorCode.E_CORRUPT_ASSET,
                "empty WAV clip",
                path=path,
                retryable=False,
            )

        clip = LoadedClip(
            path=path,
            samples=np.ascontiguousarray(data, dtype=np.float32),
            sample_rate=sample_rate,
            frames=int(data.shape[0]),
            content_hash=sha256_file(path),
            source_name=os.path.basename(path),
        )

        with self._lock:
            # Another worker may have loaded the same path concurrently.
            existing = self._cache.get(path)
            if existing is not None:
                return existing
            self._cache[path] = clip
        return clip
