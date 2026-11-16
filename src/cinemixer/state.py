"""Checkpoint state for crash recovery.

The state file lives next to the rendered output and records:

* a checksum of the full project description and per-track source-file
  checksums, so stale or tampered checkpoints are rejected;
* the set of successfully processed tracks and the cached rendered track
  buffer (``.npy``) for each one;
* the ordinal position of the most recently completed track.

Any JSON/IO corruption causes a full re-process fallback (documented warning
``W_STATE_CORRUPT``); a state file from a different project is ignored
silently as if no checkpoint existed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .logging_util import StructuredLogger

STATE_SUFFIX = ".state.json"
CACHE_SUFFIX = ".cache.npy"
STATE_VERSION = 1


def checksum_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checksum_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_path_for(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(output.name + STATE_SUFFIX)


def _cache_path(state_file: Path, track_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in track_id)
    return state_file.with_name(state_file.name + f".{safe_id}{CACHE_SUFFIX}")


@dataclass
class TrackState:
    track_id: str
    source_checksum: str
    cache_path: str
    status: str  # "done" | "failed"
    source_length: int


class CheckpointStore:
    def __init__(
        self,
        output_path: str | Path,
        project_checksum: str,
        log: StructuredLogger,
    ):
        self.state_file = state_path_for(output_path)
        self.project_checksum = project_checksum
        self.log = log
        self.tracks: dict[str, TrackState] = {}
        self.version = STATE_VERSION
        self.valid = False

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """Attempt to load a checkpoint. Returns True when reusable."""
        if not self.state_file.is_file():
            return False
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("state root is not an object")
            if payload.get("version") != STATE_VERSION:
                raise ValueError("unsupported state version")
            if payload.get("project_checksum") != self.project_checksum:
                # Different project: ignore rather than treat as corruption.
                self.log.info("ignoring checkpoint from a different project", "I_STATE_STALE")
                return False
            raw_tracks = payload.get("tracks")
            if not isinstance(raw_tracks, dict):
                raise ValueError("tracks section missing")
            tracks: dict[str, TrackState] = {}
            for track_id, item in raw_tracks.items():
                if not isinstance(item, dict):
                    raise ValueError(f"invalid track state for {track_id}")
                cache = Path(item["cache_path"])
                if item.get("status") == "done" and not cache.is_file():
                    raise ValueError(f"missing cache for track {track_id}")
                tracks[track_id] = TrackState(
                    track_id=track_id,
                    source_checksum=str(item["source_checksum"]),
                    cache_path=str(cache),
                    status=str(item["status"]),
                    source_length=int(item.get("source_length", 0)),
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
            self.log.warning(
                "W_STATE_CORRUPT",
                "checkpoint unreadable or invalid; falling back to full reprocessing",
                detail=str(exc),
            )
            self._discard()
            return False
        self.tracks = tracks
        self.valid = True
        return True

    def _discard(self) -> None:
        self.tracks = {}
        self.valid = False
        try:
            for cache in self.state_file.parent.glob(self.state_file.name + f".*{CACHE_SUFFIX}"):
                cache.unlink(missing_ok=True)
            self.state_file.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - cleanup is best effort
            pass

    # ------------------------------------------------------------- inspection
    def is_track_done(self, track_id: str, source_checksum: str) -> bool:
        state = self.tracks.get(track_id)
        return (
            state is not None
            and state.status == "done"
            and state.source_checksum == source_checksum
        )

    def load_cached_track(self, track_id: str) -> np.ndarray | None:
        state = self.tracks.get(track_id)
        if state is None or state.status != "done":
            return None
        try:
            return np.load(state.cache_path, mmap_mode=None)
        except (OSError, ValueError) as exc:
            self.log.warning(
                "W_STATE_CORRUPT",
                "cached track buffer unreadable; reprocessing track",
                track_id=track_id,
                detail=str(exc),
            )
            return None

    # --------------------------------------------------------------- mutation
    def mark_track(
        self,
        track_id: str,
        source_checksum: str,
        status: str,
        buffer: np.ndarray | None = None,
        source_length: int = 0,
    ) -> None:
        cache_path = _cache_path(self.state_file, track_id)
        if status == "done":
            if buffer is None:
                raise ValueError("a rendered buffer is required when marking a track done")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, buffer)
        self.tracks[track_id] = TrackState(
            track_id, source_checksum, str(cache_path), status, source_length
        )
        self._flush()

    def _flush(self) -> None:
        payload: dict[str, Any] = {
            "version": self.version,
            "project_checksum": self.project_checksum,
            "tracks": {
                track_id: {
                    "source_checksum": item.source_checksum,
                    "cache_path": item.cache_path,
                    "status": item.status,
                    "source_length": item.source_length,
                }
                for track_id, item in sorted(self.tracks.items())
            },
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.state_file)

    def cleanup(self) -> None:
        """Remove checkpoint and caches after a successful full render."""
        self._discard()
