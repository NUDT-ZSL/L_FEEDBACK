"""Versioned, checksum-protected render snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import BusinessError, invalid_output_dir, snapshot_error


SNAPSHOT_FORMAT = "spatial-audio-render-snapshot"
SNAPSHOT_VERSION = 1
_MANIFEST_RE = re.compile(r"snapshot_(\d+)\.json$")


@dataclass
class LoadedSnapshot:
    manifest_path: Path
    manifest: Dict[str, Any]
    dry_mix: np.ndarray
    wet_lanes: np.ndarray
    render_frame: int
    sample_frame: int


@dataclass
class SnapshotScan:
    snapshot: Optional[LoadedSnapshot]
    issues: List[Dict[str, Any]]
    version_mismatch: bool


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".%s." % path.name, suffix=".tmp",
            dir=str(path.parent), delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(str(temporary_path), str(path))
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise invalid_output_dir(
            "snapshot file cannot be written",
            path=str(path),
            reason=exc.strerror or str(exc),
        ) from exc


class SnapshotStore:
    def __init__(self, directory: Path, scene_fingerprint: str,
                 frame_size: int, total_frames: int, total_samples: int,
                 max_snapshots: int = 8) -> None:
        self.directory = directory
        self.scene_fingerprint = scene_fingerprint
        self.frame_size = frame_size
        self.total_frames = total_frames
        self.total_samples = total_samples
        self.max_snapshots = max(1, int(max_snapshots))
        render_identity = {
            "scene_fingerprint": scene_fingerprint,
            "frame_size": frame_size,
            "total_frames": total_frames,
            "total_samples": total_samples,
            "snapshot_format": SNAPSHOT_FORMAT,
            "snapshot_version": SNAPSHOT_VERSION,
        }
        self.render_fingerprint = hashlib.sha256(
            _canonical_json_bytes(render_identity)
        ).hexdigest()

    def write(self, render_frame: int, sample_frame: int,
              dry_mix: np.ndarray, wet_lanes: np.ndarray,
              emitter_states: Sequence[Mapping[str, Any]],
              reverb_state: Mapping[str, Any]) -> Path:
        manifest_name = "snapshot_%012d.json" % render_frame
        state_name = "snapshot_%012d.state.npz" % render_frame
        state_path = self.directory / state_name
        manifest_path = self.directory / manifest_name

        state_buffer = io.BytesIO()
        np.savez_compressed(
            state_buffer,
            dry_mix=np.asarray(dry_mix[:, :sample_frame], dtype=np.float64),
            wet_lanes=np.asarray(wet_lanes[:, :sample_frame], dtype=np.float64),
            snapshot_version=np.asarray([SNAPSHOT_VERSION], dtype=np.int64),
            sample_frame=np.asarray([sample_frame], dtype=np.int64),
            render_frame=np.asarray([render_frame], dtype=np.int64),
        )
        state_bytes = state_buffer.getvalue()
        state_sha256 = hashlib.sha256(state_bytes).hexdigest()
        _atomic_write(state_path, state_bytes)

        manifest: Dict[str, Any] = {
            "format": SNAPSHOT_FORMAT,
            "version": SNAPSHOT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scene_fingerprint": self.scene_fingerprint,
            "render_fingerprint": self.render_fingerprint,
            "frame_size": self.frame_size,
            "total_frames": self.total_frames,
            "total_samples": self.total_samples,
            "render_frame": int(render_frame),
            "sample_frame": int(sample_frame),
            "progress": {
                "processed_frame_range": [0, int(render_frame)],
                "next_frame": int(render_frame),
                "processed_frames": int(render_frame),
                "processed_samples": int(sample_frame),
                "remaining_frames": max(0, self.total_frames - int(render_frame)),
                "remaining_samples": max(0, self.total_samples - int(sample_frame)),
            },
            "emitters": list(emitter_states),
            "reverb": dict(reverb_state),
            "state_file": state_name,
            "state_size": len(state_bytes),
            "state_sha256": state_sha256,
        }
        manifest["integrity"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
        _atomic_write(manifest_path, _canonical_json_bytes(manifest) + b"\n")
        self._prune()
        return manifest_path

    def _manifest_paths(self) -> List[Tuple[int, Path]]:
        paths: List[Tuple[int, Path]] = []
        try:
            entries = list(self.directory.iterdir())
        except OSError:
            return paths
        for entry in entries:
            match = _MANIFEST_RE.match(entry.name)
            if entry.is_file() and match:
                paths.append((int(match.group(1)), entry))
        paths.sort(key=lambda item: item[0], reverse=True)
        return paths

    def _prune(self) -> None:
        for _, manifest_path in self._manifest_paths()[self.max_snapshots:]:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                state_name = str(manifest.get("state_file", ""))
                if state_name and Path(state_name).name == state_name:
                    state_path = self.directory / state_name
                    if state_path.exists():
                        state_path.unlink()
            except (OSError, ValueError, TypeError):
                pass
            try:
                manifest_path.unlink()
            except OSError:
                pass

    def _load_state(self, manifest: Mapping[str, Any], manifest_path: Path,
                    expected_lanes: int) -> Tuple[np.ndarray, np.ndarray]:
        state_name = manifest.get("state_file")
        if not isinstance(state_name, str) or not state_name:
            raise snapshot_error("SNAP_CORRUPTED", "snapshot state file is missing",
                                 path=str(manifest_path))
        if Path(state_name).name != state_name:
            raise snapshot_error("SNAP_CORRUPTED", "snapshot state path is invalid",
                                 path=str(manifest_path), state_file=state_name)
        state_path = self.directory / state_name
        try:
            state_bytes = state_path.read_bytes()
        except OSError as exc:
            raise snapshot_error("SNAP_CORRUPTED", "snapshot state file cannot be read",
                                 path=str(manifest_path), state_file=state_name,
                                 reason=str(exc)) from exc
        expected_size = manifest.get("state_size")
        if expected_size is not None and expected_size != len(state_bytes):
            raise snapshot_error("SNAP_CORRUPTED", "snapshot state size mismatch",
                                 path=str(manifest_path), state_file=state_name)
        expected_hash = manifest.get("state_sha256")
        actual_hash = hashlib.sha256(state_bytes).hexdigest()
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise snapshot_error("SNAP_CORRUPTED", "snapshot state checksum mismatch",
                                 path=str(manifest_path), state_file=state_name)

        expected_sample_frame = int(manifest["sample_frame"])
        try:
            with np.load(io.BytesIO(state_bytes), allow_pickle=False) as archive:
                version_values = np.asarray(archive["snapshot_version"]).reshape(-1)
                if version_values.size != 1 or int(version_values[0]) != SNAPSHOT_VERSION:
                    raise snapshot_error("SNAP_VERSION_MISMATCH",
                                         "snapshot payload version does not match",
                                         path=str(manifest_path))
                payload_render_frame = int(np.asarray(archive["render_frame"]).reshape(-1)[0])
                payload_sample_frame = int(np.asarray(archive["sample_frame"]).reshape(-1)[0])
                dry_mix = np.asarray(archive["dry_mix"], dtype=np.float64)
                wet_lanes = np.asarray(archive["wet_lanes"], dtype=np.float64)
        except BusinessError:
            raise
        except (KeyError, ValueError, OSError, EOFError, TypeError) as exc:
            raise snapshot_error("SNAP_CORRUPTED", "snapshot payload cannot be decoded",
                                 path=str(manifest_path), reason=str(exc)) from exc

        if payload_render_frame != int(manifest.get("render_frame", -1)):
            raise snapshot_error("SNAP_CORRUPTED", "snapshot render frame mismatch",
                                 path=str(manifest_path))
        if payload_sample_frame != expected_sample_frame:
            raise snapshot_error("SNAP_CORRUPTED", "snapshot sample frame mismatch",
                                 path=str(manifest_path))
        if dry_mix.ndim != 2 or dry_mix.shape[0] != 2 or dry_mix.shape[1] != expected_sample_frame:
            raise snapshot_error("SNAP_CORRUPTED", "dry mix state shape is invalid",
                                 path=str(manifest_path), shape=list(dry_mix.shape))
        if (wet_lanes.ndim != 2 or wet_lanes.shape[0] != expected_lanes
                or wet_lanes.shape[1] != expected_sample_frame):
            raise snapshot_error("SNAP_CORRUPTED", "reverb lane state shape is invalid",
                                 path=str(manifest_path), shape=list(wet_lanes.shape),
                                 expected_lanes=expected_lanes)
        if not np.all(np.isfinite(dry_mix)) or not np.all(np.isfinite(wet_lanes)):
            raise snapshot_error("SNAP_CORRUPTED", "snapshot state contains non-finite samples",
                                 path=str(manifest_path))
        return dry_mix, wet_lanes

    def scan(self, expected_lanes: int, force_resume: bool) -> SnapshotScan:
        issues: List[Dict[str, Any]] = []
        version_mismatch = False
        saw_any_manifest = False

        for _, manifest_path in self._manifest_paths():
            saw_any_manifest = True
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "manifest cannot be decoded: %s" % exc})
                continue
            if not isinstance(manifest_data, dict):
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "manifest root is not an object"})
                continue
            if manifest_data.get("format") != SNAPSHOT_FORMAT:
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "unknown snapshot format"})
                continue
            if manifest_data.get("version") != SNAPSHOT_VERSION:
                version_mismatch = True
                issues.append({"code": "SNAP_VERSION_MISMATCH", "path": str(manifest_path),
                               "reason": "snapshot version is not supported",
                               "found_version": manifest_data.get("version"),
                               "supported_version": SNAPSHOT_VERSION})
                if force_resume:
                    raise snapshot_error(
                        "SNAP_VERSION_MISMATCH",
                        "the latest snapshot uses an unsupported version",
                        path=str(manifest_path),
                        found_version=manifest_data.get("version"),
                        supported_version=SNAPSHOT_VERSION,
                    )
                continue

            integrity = manifest_data.pop("integrity", None)
            actual_integrity = hashlib.sha256(_canonical_json_bytes(manifest_data)).hexdigest()
            if not isinstance(integrity, str) or integrity != actual_integrity:
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "manifest checksum mismatch"})
                continue
            manifest_data["integrity"] = integrity

            if manifest_data.get("scene_fingerprint") != self.scene_fingerprint:
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "snapshot scene fingerprint does not match"})
                continue
            if manifest_data.get("render_fingerprint") != self.render_fingerprint:
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "snapshot render settings do not match"})
                continue
            try:
                render_frame = int(manifest_data["render_frame"])
                sample_frame = int(manifest_data["sample_frame"])
            except (KeyError, TypeError, ValueError):
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "snapshot progress fields are invalid"})
                continue
            if (render_frame < 0 or sample_frame < 0 or render_frame > self.total_frames
                    or sample_frame > self.total_samples
                    or sample_frame != min(render_frame * self.frame_size, self.total_samples)):
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "snapshot progress is inconsistent"})
                continue
            if not manifest_data.get("reverb", {}).get("precomputed", False):
                issues.append({"code": "SNAP_CORRUPTED", "path": str(manifest_path),
                               "reason": "snapshot lacks reverb precompute state"})
                continue

            try:
                dry_mix, wet_lanes = self._load_state(manifest_data, manifest_path, expected_lanes)
            except BusinessError as exc:
                if exc.code == "SNAP_VERSION_MISMATCH":
                    version_mismatch = True
                issues.append({"code": exc.code, "path": str(manifest_path),
                               "reason": exc.message})
                continue
            return SnapshotScan(
                LoadedSnapshot(
                    manifest_path=manifest_path,
                    manifest=manifest_data,
                    dry_mix=dry_mix,
                    wet_lanes=wet_lanes,
                    render_frame=render_frame,
                    sample_frame=sample_frame,
                ),
                issues,
                version_mismatch,
            )

        if force_resume or saw_any_manifest:
            if version_mismatch:
                raise snapshot_error(
                    "SNAP_VERSION_MISMATCH",
                    "no usable snapshot matches the supported version",
                    supported_version=SNAPSHOT_VERSION,
                    snapshot_directory=str(self.directory),
                    issues=issues[-5:],
                )
            if issues or saw_any_manifest:
                raise snapshot_error(
                    "SNAP_CORRUPTED",
                    "no intact snapshot could be restored",
                    snapshot_directory=str(self.directory),
                    issues=issues[-5:],
                )
            if not force_resume:
                raise snapshot_error(
                    "SNAP_CORRUPTED",
                    "snapshot files exist but none could be restored",
                    snapshot_directory=str(self.directory),
                    issues=issues[-5:],
                )
            raise snapshot_error(
                "SNAP_NOT_FOUND",
                "no snapshot exists to resume",
                snapshot_directory=str(self.directory),
            )
        return SnapshotScan(None, issues, version_mismatch)
