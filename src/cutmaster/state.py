"""Persistent track cache, checkpoint and process-lock handling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import (
    prepare_stereo_track,
    sha256_file,
    write_raw_track,
)
from .errors import err
from .manifest import Manifest, TrackSpec
from .timeline import TrackLayout

CHECKPOINT_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
CACHE_PIPELINE_VERSION = "stereo-f32le-sinc64-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_safe(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows os.kill(pid, 0) may terminate the process; use
        # OpenProcess/GetExitCodeProcess instead.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class ProcessLock:
    path: Path
    acquired: bool = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": utc_now(),
            }
        ).encode("utf-8")
        for attempt in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                try:
                    os.write(descriptor, payload)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                self.acquired = True
                return
            except FileExistsError:
                stale = False
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(existing.get("pid", -1))
                    host = str(existing.get("host", ""))
                    age = time.time() - self.path.stat().st_mtime
                    stale = (
                        pid <= 0
                        or age > 86_400
                        or (host == socket.gethostname() and not _pid_alive(pid))
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    stale = True
                if stale and attempt == 0:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise err(
                    "E_STATE_LOCKED",
                    "another mastering process owns the project state",
                    exit_code=7,
                    lock_path=str(self.path),
                )
        raise err(
            "E_STATE_LOCKED",
            "could not acquire the project state lock",
            exit_code=7,
            lock_path=str(self.path),
        )

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.acquired = False

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


class ProjectState:
    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root.resolve(strict=False)
        self.manifest = manifest
        self.project_root = self.root / "projects" / manifest.project_key
        self.cache_root = self.project_root / "cache" / "tracks"
        self.work_root = self.project_root / "runs"
        self.checkpoint_path = self.project_root / "checkpoint.json"
        self.lock_path = self.project_root / ".lock"
        self.run_id = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        self.run_root = self.work_root / self.run_id

    def initialize(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        for partial in self.cache_root.glob("*.partial"):
            try:
                partial.unlink()
            except OSError:
                pass
        self.run_root.mkdir(parents=True, exist_ok=False)

    def cleanup_stale_workdirs(self) -> int:
        removed = 0
        if not self.work_root.is_dir():
            return removed
        for child in self.work_root.iterdir():
            if child == self.run_root:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def read_checkpoint(self) -> tuple[dict[str, Any] | None, str | None]:
        if not self.checkpoint_path.is_file():
            return None, None
        try:
            value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            try:
                self.checkpoint_path.replace(
                    self.checkpoint_path.with_name(
                        f"checkpoint.corrupt-{int(time.time())}.json"
                    )
                )
            except OSError:
                pass
            return None, "checkpoint_unreadable"
        if not isinstance(value, dict):
            return None, "checkpoint_unreadable"
        return value, None

    def write_checkpoint(
        self,
        *,
        status: str,
        stage: str,
        completed_track_ids: list[str],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "project": {
                "name": self.manifest.project.name,
                "version": self.manifest.project.version,
                "segment_id": self.manifest.project.segment_id,
            },
            "manifest_sha256": self.manifest.sha256,
            "run_id": self.run_id,
            "status": status,
            "stage": stage,
            "completed_track_ids": completed_track_ids,
            "updated_at": utc_now(),
        }
        if extra:
            value.update(extra)
        atomic_write_json(self.checkpoint_path, value)
        return value

    def mark_failed(self, error: dict[str, Any], completed_track_ids: list[str]) -> None:
        try:
            self.write_checkpoint(
                status="failed",
                stage="failed",
                completed_track_ids=completed_track_ids,
                extra={"error": error},
            )
        except OSError:
            pass

    def cache_paths(self, fingerprint: str) -> tuple[Path, Path]:
        return (
            self.cache_root / f"{fingerprint}.f32",
            self.cache_root / f"{fingerprint}.json",
        )

    def cache_fingerprint(
        self, track: TrackSpec, source_sha256: str, target_rate: int
    ) -> str:
        payload = "|".join(
            (
                CACHE_PIPELINE_VERSION,
                source_sha256,
                str(track.sample_rate),
                str(track.channels),
                str(target_rate),
                "stereo",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_cache(
        self,
        fingerprint: str,
        expected_frames: int,
        expected_rate: int,
    ) -> tuple[Path | None, str | None]:
        raw_path, meta_path = self.cache_paths(fingerprint)
        if not raw_path.is_file() or not meta_path.is_file():
            return None, "cache_missing"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame_count = int(meta["frames"])
            channels = int(meta["channels"])
            sample_rate = int(meta["sample_rate"])
            raw_sha = str(meta["raw_sha256"])
            expected_size = int(meta["bytes"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None, "cache_metadata_invalid"
        if (
            meta.get("schema_version") != CACHE_SCHEMA_VERSION
            or meta.get("fingerprint") != fingerprint
            or frame_count != expected_frames
            or channels != 2
            or sample_rate != expected_rate
            or expected_size != frame_count * 2 * 4
        ):
            return None, "cache_metadata_mismatch"
        try:
            actual_size = raw_path.stat().st_size
        except OSError:
            return None, "cache_missing"
        if actual_size != expected_size:
            return None, "cache_size_mismatch"
        if sha256_file(raw_path) != raw_sha:
            return None, "cache_checksum_mismatch"
        return raw_path, None

    def prepare_track_cache(
        self, layout: TrackLayout, *, crash_after: str | None = None
    ) -> dict[str, Any]:
        track = layout.spec
        info_sha = sha256_file(track.path)
        fingerprint = self.cache_fingerprint(
            track, info_sha, self.manifest.project.sample_rate
        )
        raw_path, meta_path = self.cache_paths(fingerprint)
        cached, reason = self._validate_cache(
            fingerprint,
            layout.resampled_frames,
            self.manifest.project.sample_rate,
        )
        if cached is not None:
            return {
                "track_id": track.track_id,
                "cache_hit": True,
                "fingerprint": fingerprint,
                "raw_path": raw_path,
                "reason": reason,
            }

        partial_raw = raw_path.with_name(
            f".{raw_path.name}.{self.run_id}.partial"
        )
        partial_meta = meta_path.with_name(
            f".{meta_path.name}.{self.run_id}.partial"
        )
        try:
            data = prepare_stereo_track(
                track.path,
                track.sample_rate,
                self.manifest.project.sample_rate,
            )
            actual_frames = int(data.shape[0])
            if actual_frames != layout.resampled_frames:
                raise err(
                    "E_RESAMPLE_LENGTH",
                    "resampled frame count does not match timeline planning",
                    exit_code=4,
                    track_id=track.track_id,
                    expected=layout.resampled_frames,
                    actual=actual_frames,
                )
            raw_sha = write_raw_track(partial_raw, data)
            del data
            os.replace(partial_raw, raw_path)
            atomic_write_json(
                meta_path,
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "pipeline_version": CACHE_PIPELINE_VERSION,
                    "fingerprint": fingerprint,
                    "source_sha256": info_sha,
                    "source_sample_rate": track.sample_rate,
                    "source_channels": track.channels,
                    "sample_rate": self.manifest.project.sample_rate,
                    "channels": 2,
                    "dtype": "float32le",
                    "frames": layout.resampled_frames,
                    "bytes": layout.resampled_frames * 2 * 4,
                    "raw_sha256": raw_sha,
                    "created_run_id": self.run_id,
                    "created_at": utc_now(),
                },
            )
        finally:
            for temporary in (partial_raw, partial_meta):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        if crash_after == f"track:{track.track_id}":
            os._exit(75)
        return {
            "track_id": track.track_id,
            "cache_hit": False,
            "fingerprint": fingerprint,
            "raw_path": raw_path,
            "reason": reason,
        }
