"""Persistent intermediate state, atomic writes and incremental detection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .hashing import canonical_json_hash, sha256_file
from .project import Project

STATE_VERSION = 1


def state_file_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".state.json")


def intermediate_video_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(output.stem + ".intermediate.mp4")


def audio_cache_dir(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(output.stem + ".audiocache")


def cleanup_stale_temps(output_path: str | Path) -> None:
    """Remove leftovers from a force-killed previous run.

    A state file only becomes visible via an atomic rename, so any ``*.tmp``
    state/intermediate file left behind is by definition incomplete.
    """
    output = Path(output_path)
    directory = output.parent
    if not directory.exists():
        return
    state_tmp = state_file_path(output).with_suffix(".json.tmp")
    candidates = [state_tmp]
    candidates.extend(directory.glob("*.state.json.tmp"))
    candidates.extend(directory.glob("*.intermediate.mp4.tmp"))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


@dataclass
class IncrementalDecision:
    reuse_video: bool
    reason: str
    changed_tracks: list[int]


def build_state(
    project: Project,
    video_info: dict,
    video_digest: str,
    intermediate_path: Path,
    intermediate_digest: str,
    track_digests: list[str],
) -> dict:
    return {
        "version": STATE_VERSION,
        "video": {
            "path": str(project.video_path),
            "sha256": video_digest,
            "width": video_info["width"],
            "height": video_info["height"],
            "frame_rate": str(video_info["frame_rate"]),
            "frame_count": video_info["frame_count"],
            "markers_hash": canonical_json_hash(
                [
                    {"frame": marker.frame, "drift_ms": marker.drift_ms}
                    for marker in project.markers
                ]
            ),
        },
        "intermediate_video": {
            "path": str(intermediate_path),
            "sha256": intermediate_digest,
        },
        "audio_tracks": [
            {
                "index": track.index,
                "path": str(track.path),
                "sha256": digest,
                "offset_ms": track.offset_ms,
                "envelope_hash": canonical_json_hash(list(track.envelope)),
            }
            for track, digest in zip(project.tracks, track_digests)
        ],
    }


def evaluate(project: Project, video_info: dict, video_digest: str) -> IncrementalDecision:
    """Decide whether the persisted intermediate video can be reused."""
    state_path = state_file_path(project.output_path)
    if not state_path.exists():
        return IncrementalDecision(False, "no state file present", [])
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return IncrementalDecision(False, "state file unreadable", [])
    if state.get("version") != STATE_VERSION:
        return IncrementalDecision(False, "state version mismatch", [])

    saved_video = state.get("video", {})
    if saved_video.get("sha256") != video_digest:
        return IncrementalDecision(False, "video file content changed", [])
    if saved_video.get("frame_rate") != str(video_info["frame_rate"]):
        return IncrementalDecision(False, "frame rate mismatch", [])
    if saved_video.get("frame_count") != video_info["frame_count"]:
        return IncrementalDecision(False, "frame count mismatch", [])
    if (
        saved_video.get("width") != video_info["width"]
        or saved_video.get("height") != video_info["height"]
    ):
        return IncrementalDecision(False, "video dimensions mismatch", [])

    expected_markers = canonical_json_hash(
        [
            {"frame": marker.frame, "drift_ms": marker.drift_ms}
            for marker in project.markers
        ]
    )
    if saved_video.get("markers_hash") != expected_markers:
        return IncrementalDecision(False, "frame markers changed", [])

    saved_tracks = state.get("audio_tracks", [])
    if len(saved_tracks) != len(project.tracks):
        return IncrementalDecision(False, "audio track count changed", [])

    changed: list[int] = []
    for track, saved in zip(project.tracks, saved_tracks):
        if saved.get("index") != track.index or saved.get("path") != str(track.path):
            return IncrementalDecision(False, "audio track identity changed", [])
        if saved.get("sha256") != sha256_file(track.path):
            return IncrementalDecision(False, "audio file content changed", [])
        config_changed = (
            saved.get("offset_ms") != track.offset_ms
            or saved.get("envelope_hash")
            != canonical_json_hash(list(track.envelope))
        )
        if config_changed:
            changed.append(track.index)

    intermediate = state.get("intermediate_video", {})
    intermediate_path = Path(intermediate.get("path", ""))
    if not intermediate_path.is_file():
        return IncrementalDecision(False, "intermediate video missing", [])
    if sha256_file(intermediate_path) != intermediate.get("sha256"):
        return IncrementalDecision(False, "intermediate video digest mismatch", [])

    if changed:
        reason = f"audio-only changes on tracks {changed}"
    else:
        reason = "all inputs identical; reusing persisted intermediate video"
    return IncrementalDecision(True, reason, changed)
