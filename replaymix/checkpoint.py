"""Versioned JSONL checkpoints with byte row offsets and audio caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import TrackError
from .track import SegmentAudio, TrackResult

CHECKPOINT_FORMAT = "replaymix-checkpoint"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CheckpointLoad:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    valid: bool = False
    reason: str = "absent"
    version: int | None = None
    row_offsets: dict[str, int] = field(default_factory=dict)


def load_checkpoint_file(
    path: Path,
    *,
    expected_version: int,
    run_signature: str,
) -> CheckpointLoad:
    if not path.is_file():
        return CheckpointLoad(valid=False, reason="absent")
    try:
        data = path.read_bytes()
    except OSError:
        return CheckpointLoad(valid=False, reason="unreadable")

    lines: list[tuple[int, bytes]] = []
    position = 0
    for line in data.splitlines(keepends=True):
        lines.append((position, line))
        position += len(line)
    if len(lines) < 2:
        return CheckpointLoad(valid=False, reason="corrupt")

    try:
        header = json.loads(lines[0][1])
        footer = json.loads(lines[-1][1])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return CheckpointLoad(valid=False, reason="corrupt")
    if (
        not isinstance(header, dict)
        or header.get("type") != "header"
        or header.get("format") != CHECKPOINT_FORMAT
        or not isinstance(footer, dict)
        or footer.get("type") != "footer"
    ):
        return CheckpointLoad(valid=False, reason="corrupt")

    version = header.get("version")
    if version != expected_version:
        return CheckpointLoad(
            valid=False, reason="version_changed", version=version
        )
    if header.get("run_signature") != run_signature:
        return CheckpointLoad(
            valid=False, reason="run_signature_changed", version=version
        )

    row_offsets_raw = footer.get("row_offsets")
    row_sha_raw = footer.get("row_sha256")
    if not isinstance(row_offsets_raw, dict) or not isinstance(row_sha_raw, dict):
        return CheckpointLoad(valid=False, reason="row_offset_mismatch", version=version)

    line_by_offset = {offset: line for offset, line in lines[:-1]}
    rows: dict[str, dict[str, Any]] = {}
    offsets: dict[str, int] = {}
    for track_key, offset_raw in row_offsets_raw.items():
        if isinstance(offset_raw, bool) or not isinstance(offset_raw, int):
            return CheckpointLoad(
                valid=False, reason="row_offset_mismatch", version=version
            )
        line = line_by_offset.get(offset_raw)
        if line is None:
            return CheckpointLoad(
                valid=False, reason="row_offset_mismatch", version=version
            )
        expected_sha = row_sha_raw.get(track_key)
        if not isinstance(expected_sha, str) or sha256_bytes(line) != expected_sha:
            return CheckpointLoad(
                valid=False, reason="row_offset_mismatch", version=version
            )
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return CheckpointLoad(
                valid=False, reason="row_offset_mismatch", version=version
            )
        if (
            not isinstance(row, dict)
            or row.get("type") != "track-row"
            or str(row.get("track_id")) != str(track_key)
        ):
            return CheckpointLoad(
                valid=False, reason="row_offset_mismatch", version=version
            )
        rows[str(track_key)] = row
        offsets[str(track_key)] = offset_raw

    return CheckpointLoad(
        rows=rows,
        valid=True,
        reason="valid",
        version=version,
        row_offsets=offsets,
    )


class CheckpointManager:
    def __init__(
        self,
        *,
        path: Path,
        cache_dir: Path,
        version: int,
        run_signature: str,
        enabled: bool = True,
    ) -> None:
        self.path = path
        self.cache_dir = cache_dir
        self.version = version
        self.run_signature = run_signature
        self.enabled = enabled
        self.rows: dict[str, dict[str, Any]] = {}
        self.load_result = CheckpointLoad(valid=False, reason="disabled")
        self.reused_track_ids: list[Any] = []
        self.recomputed_track_ids: list[Any] = []
        self.full_rebuild = False
        self.last_offsets: dict[str, int] = {}

    def _key(self, track_id: Any) -> str:
        return str(track_id)

    def load(self) -> CheckpointLoad:
        if not self.enabled:
            self.load_result = CheckpointLoad(valid=False, reason="disabled")
            return self.load_result
        result = load_checkpoint_file(
            self.path,
            expected_version=self.version,
            run_signature=self.run_signature,
        )
        self.load_result = result
        if result.valid:
            self.rows = dict(result.rows)
        else:
            self.rows = {}
            if result.reason not in ("absent",):
                self.full_rebuild = True
                self._clear_cache()
            self.flush()
        return result

    def _clear_cache(self) -> None:
        if not self.cache_dir.is_dir():
            return
        for path in self.cache_dir.glob("track_*.npz"):
            try:
                path.unlink()
            except OSError:
                pass

    def _cache_name(self, track_id: Any) -> str:
        raw = str(track_id)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "track"
        suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        return f"track_{safe}_{suffix}.npz"

    def _save_audio_cache(self, result: TrackResult) -> dict[str, Any] | None:
        if not result.active_audio:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        audio = np.concatenate([item.audio for item in result.active_audio])
        lengths = np.asarray(
            [item.audio.size for item in result.active_audio], dtype=np.int64
        )
        if int(lengths.sum()) != int(audio.size):
            raise ValueError("internal active-audio cache length mismatch")
        name = self._cache_name(result.track_id)
        target = self.cache_dir / name
        fd, temp_name = tempfile.mkstemp(
            prefix=".cache-", suffix=".npz", dir=str(self.cache_dir)
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            np.savez(temp_path, audio=audio, lengths=lengths)
            os.replace(temp_path, target)
        except BaseException:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise
        return {
            "path": name,
            "sha256": sha256_path(target),
            "lengths": [int(value) for value in lengths],
            "frames": int(audio.size),
        }

    def _load_audio_cache(
        self, row: dict[str, Any], active_segment_count: int
    ) -> list[SegmentAudio] | None:
        cache = row.get("cache")
        if not isinstance(cache, dict):
            return None
        name = cache.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            return None
        path = self.cache_dir / name
        if not path.is_file():
            return None
        try:
            if sha256_path(path) != cache.get("sha256"):
                return None
            with np.load(path, allow_pickle=False) as loaded:
                audio = np.asarray(loaded["audio"], dtype=np.float64)
                lengths = [int(value) for value in loaded["lengths"].tolist()]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if len(lengths) != active_segment_count or int(sum(lengths)) != audio.size:
            return None
        if cache.get("lengths") != lengths or cache.get("frames") != int(audio.size):
            return None

        active_records = [
            segment
            for segment in row.get("segments", [])
            if segment.get("kind") == "active"
        ]
        if len(active_records) != active_segment_count:
            return None
        segments: list[SegmentAudio] = []
        cursor = 0
        for record, length in zip(active_records, lengths):
            chunk = audio[cursor : cursor + length].copy()
            cursor += length
            if int(record.get("frames", -1)) != length:
                return None
            segments.append(
                SegmentAudio(start_frame=int(record["start_frame"]), audio=chunk)
            )
        return segments

    def restore_result(
        self,
        *,
        spec: Any,
        input_signature: str,
    ) -> TrackResult | None:
        row = self.rows.get(self._key(spec.track_id))
        if row is None or row.get("input_signature") != input_signature:
            return None
        if row.get("status") == "failed":
            return None
        active_records = [
            segment
            for segment in row.get("segments", [])
            if segment.get("kind") == "active"
        ]
        audio: list[SegmentAudio] = []
        if active_records:
            loaded = self._load_audio_cache(row, len(active_records))
            if loaded is None:
                return None
            audio = loaded
        errors = [
            TrackError(code=str(item["code"]), message=str(item["message"]))
            for item in row.get("errors", [])
            if isinstance(item, dict) and "code" in item and "message" in item
        ]
        result = TrackResult(
            track_id=spec.track_id,
            spec=spec,
            status=str(row.get("status", "skipped")),
            errors=errors,
            actual=dict(row.get("actual", {})),
            resample=dict(row.get("resample", {})),
            segments=list(row.get("segments", [])),
            active_audio=audio,
            counts=dict(row.get("counts", {})),
            input_signature=input_signature,
            from_checkpoint=True,
        )
        self.reused_track_ids.append(spec.track_id)
        return result

    def put(self, result: TrackResult) -> None:
        if not self.enabled:
            return
        cache = None
        if result.status in ("processed", "processed_with_errors"):
            cache = self._save_audio_cache(result)
        row = {
            "type": "track-row",
            "track_id": result.track_id,
            "input_signature": result.input_signature,
            "status": result.status,
            "errors": [item.as_dict() for item in result.errors],
            "actual": result.actual,
            "resample": result.resample,
            "segments": result.segments,
            "counts": result.counts,
            "cache": cache,
        }
        self.rows[self._key(result.track_id)] = row
        self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        header = {
            "type": "header",
            "format": CHECKPOINT_FORMAT,
            "version": self.version,
            "run_signature": self.run_signature,
        }
        header_line = (_canonical_json(header) + "\n").encode("utf-8")
        ordered_keys = sorted(
            self.rows,
            key=lambda key: (
                0,
                int(key),
            )
            if key.lstrip("-").isdigit()
            else (1, key),
        )
        body = bytearray()
        offsets: dict[str, int] = {}
        row_hashes: dict[str, str] = {}
        for key in ordered_keys:
            line = (_canonical_json(self.rows[key]) + "\n").encode("utf-8")
            offsets[key] = len(header_line) + len(body)
            row_hashes[key] = sha256_bytes(line)
            body.extend(line)
        footer = {
            "type": "footer",
            "row_offsets": offsets,
            "row_sha256": row_hashes,
            "track_count": len(ordered_keys),
        }
        footer_line = (_canonical_json(footer) + "\n").encode("utf-8")
        self.last_offsets = dict(offsets)
        _atomic_write(self.path, header_line + bytes(body) + footer_line)

    def summary(self, *, enabled: bool = True) -> dict[str, Any]:
        if not enabled:
            return {
                "enabled": False,
                "path": None,
                "status": "disabled",
                "reused": False,
                "reused_track_ids": [],
                "recomputed_track_ids": [],
                "rebuild_reason": None,
            }
        load = self.load_result
        if not load.valid and load.reason == "absent":
            status = "created"
        elif not load.valid and load.reason == "disabled":
            status = "disabled"
        elif self.full_rebuild:
            status = "rebuilt"
        elif self.reused_track_ids and self.recomputed_track_ids:
            status = "partial"
        elif self.reused_track_ids:
            status = "reused"
        else:
            status = "created"
        def track_sort(value: Any) -> tuple[int, Any]:
            text = str(value)
            try:
                return (0, int(text))
            except ValueError:
                return (1, text)

        return {
            "enabled": True,
            "path": str(self.path),
            "version": self.version,
            "status": status,
            "reused": bool(self.reused_track_ids),
            "reused_track_ids": sorted(
                dict.fromkeys(self.reused_track_ids), key=track_sort
            ),
            "recomputed_track_ids": sorted(
                dict.fromkeys(self.recomputed_track_ids), key=track_sort
            ),
            "rebuild_reason": (
                load.reason
                if self.full_rebuild
                else None
            ),
            "row_offsets": (
                dict(self.last_offsets)
                if self.last_offsets
                else (dict(load.row_offsets) if load.valid else {})
            ),
        }
