from __future__ import annotations

import hashlib
import json
import os
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import BusinessError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise BusinessError(
            code="E_AUDIO_READ",
            message=f"Unable to hash input file: {exc}",
            path=str(path),
        ) from exc
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fsync_file(path: Path) -> None:
    try:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def temp_path_for(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise BusinessError(
            code="E_OUTPUT_WRITE",
            message=f"Unable to remove stale artifact: {exc}",
            path=str(path),
        ) from exc


def remove_partial(path: Path) -> None:
    remove_path(temp_path_for(path))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = temp_path_for(path)
    try:
        with open(temp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise BusinessError(
            code="E_OUTPUT_WRITE",
            message=f"Unable to write output atomically: {exc}",
            path=str(path),
        ) from exc


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _write_bytes_atomic(target, payload)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def maybe_test_interrupt(failpoint: str) -> None:
    """Test-only deterministic interruption point.

    This simulates Ctrl+C at a precise artifact boundary. It is intentionally
    opt-in through an environment variable and has no effect in normal use.
    """
    if os.environ.get("CUTSCENE_DUB_TEST_FAILPOINT") != failpoint:
        return
    hold_ms = os.environ.get("CUTSCENE_DUB_TEST_HOLD_MS")
    if hold_ms:
        try:
            time.sleep(max(0.0, float(hold_ms)) / 1000.0)
        except (TypeError, ValueError):
            pass
    raise KeyboardInterrupt(f"test failpoint: {failpoint}")


class AtomicWavWriter:
    """Chunked 16-bit PCM WAV writer that only publishes a complete file."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int,
        channels: int,
        failpoint: str | None = None,
        chunk_frames: int = 16384,
    ) -> None:
        self.path = Path(path)
        self.temp_path = temp_path_for(self.path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.failpoint = failpoint
        self.chunk_frames = chunk_frames
        self._wave: wave.Wave_write | None = None
        self._closed = False
        self._committed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        remove_path(self.temp_path)
        try:
            self._wave = wave.open(str(self.temp_path), "wb")
            self._wave.setnchannels(channels)
            self._wave.setsampwidth(2)
            self._wave.setframerate(sample_rate)
            self._wave.setcomptype("NONE", "not compressed")
        except OSError as exc:
            raise BusinessError(
                code="E_OUTPUT_WRITE",
                message=f"Unable to create temporary WAV: {exc}",
                path=str(self.temp_path),
            ) from exc

    def write_pcm(self, pcm: np.ndarray) -> None:
        if self._wave is None or self._closed:
            raise RuntimeError("WAV writer is closed")
        array = np.asarray(pcm)
        if array.dtype != np.int16:
            raise ValueError("write_pcm expects int16 samples")
        if array.ndim == 1:
            frames = array.shape[0]
        elif array.ndim == 2 and array.shape[1] == self.channels:
            frames = array.shape[0]
        else:
            raise ValueError("PCM shape does not match output channel count")
        for offset in range(0, frames, self.chunk_frames):
            chunk = array[offset : offset + self.chunk_frames]
            self._wave.writeframesraw(
                np.ascontiguousarray(chunk, dtype="<i2").tobytes(order="C")
            )
            if self.failpoint is not None:
                maybe_test_interrupt(self.failpoint)

    def commit(self) -> Path:
        if self._committed:
            return self.path
        self._close()
        try:
            _fsync_file(self.temp_path)
            os.replace(self.temp_path, self.path)
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise BusinessError(
                code="E_OUTPUT_WRITE",
                message=f"Unable to publish WAV atomically: {exc}",
                path=str(self.path),
            ) from exc
        self._committed = True
        return self.path

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._wave is not None:
            self._wave.close()
            self._wave = None

    def abort(self, *, keep_partial: bool = True) -> None:
        self._close()
        if not keep_partial and not self._committed:
            try:
                self.temp_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "AtomicWavWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._committed:
            self.abort(keep_partial=True)


def write_pcm_wav_atomic(
    path: str | Path,
    pcm: np.ndarray,
    *,
    sample_rate: int,
    channels: int,
    failpoint: str | None = None,
) -> Path:
    writer = AtomicWavWriter(
        path,
        sample_rate=sample_rate,
        channels=channels,
        failpoint=failpoint,
    )
    try:
        writer.write_pcm(pcm)
        return writer.commit()
    finally:
        if not writer._committed:
            writer.abort(keep_partial=True)


def frames_to_seconds(frames: int, sample_rate: int) -> float:
    return frames / float(sample_rate)


def seconds_to_frames(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float | None:
    if value <= 0:
        return None
    return 20.0 * np.log10(value)


def path_list_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()

