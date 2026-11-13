"""Content-addressed per-language snapshots for resumable runs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import Config
from .hashing import (
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
)
from .normalize import NormalizationResult
from .scheduler import LanguageSchedule

SNAPSHOT_FILE = "snapshot.json"
TRACK_FILE = "track.f32.npy"


def block_asset_hash(
    schedule: LanguageSchedule, clip_hash_by_source: dict[str, str]
) -> str:
    """Hash the actual source clips referenced by this language block."""
    payload = [
        {
            "source_name": placement.source_name,
            "content_hash": clip_hash_by_source.get(
                placement.source_name, placement.content_hash
            ),
        }
        for placement in schedule.placements
    ]
    payload.extend(
        {"dropped": item.event_id, "reason": item.reason}
        for item in schedule.dropped
    )
    return canonical_json_hash(payload)


def schedule_fingerprint(schedule: LanguageSchedule) -> str:
    """Stable hash of scheduling decisions, independent of byte offsets."""
    payload = {
        "lang": schedule.lang,
        "placements": [item.to_index_dict() for item in schedule.placements],
        "dropped": [item.to_dict() for item in schedule.dropped],
    }
    return canonical_json_hash(payload)


@dataclass
class Snapshot:
    lang: str
    snapshot_version: int
    config_hash: str
    manifest_hash: str
    asset_hash: str
    schedule_fingerprint: str
    track_hash: str
    total_frames: int
    wav_info: dict
    index_entries: list[dict]
    normalization: dict
    dropped: list[dict]
    track_file: str
    snapshot_file: str

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        return cls(**data)


def save_snapshot(
    block_dir: str,
    config: Config,
    manifest_hash: str,
    asset_hash: str,
    schedule: LanguageSchedule,
    track: np.ndarray,
    normalization: NormalizationResult,
    wav_info: dict,
    index_entries: list[dict],
) -> Snapshot:
    """Persist track + snapshot atomically; no half-written snapshot survives."""
    os.makedirs(block_dir, exist_ok=True)
    track_path = os.path.join(block_dir, TRACK_FILE)
    snapshot_path = os.path.join(block_dir, SNAPSHOT_FILE)

    # Write the (large) track first and only publish the snapshot once it is
    # durable, so a crash never leaves a snapshot pointing at missing audio.
    tmp_track = os.path.join(block_dir, f"{TRACK_FILE}.tmp.{os.getpid()}")
    try:
        with open(tmp_track, "wb") as handle:
            np.save(handle, track.astype(np.float32, copy=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_track, track_path)
    finally:
        if os.path.exists(tmp_track):
            os.remove(tmp_track)

    track_hash = sha256_file(track_path)
    snapshot = Snapshot(
        lang=schedule.lang,
        snapshot_version=config.snapshot_version,
        config_hash=config.content_hash(),
        manifest_hash=manifest_hash,
        asset_hash=asset_hash,
        schedule_fingerprint=schedule_fingerprint(schedule),
        track_hash=track_hash,
        total_frames=int(track.shape[0]),
        wav_info=wav_info,
        index_entries=index_entries,
        normalization=normalization.to_dict(),
        dropped=[item.to_dict() for item in schedule.dropped],
        track_file=TRACK_FILE,
        snapshot_file=SNAPSHOT_FILE,
    )
    atomic_write_json(snapshot_path, snapshot.to_dict())
    return snapshot


def load_snapshot(block_dir: str) -> Snapshot | None:
    path = os.path.join(block_dir, SNAPSHOT_FILE)
    if not os.path.exists(path):
        return None
    import json

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return Snapshot.from_dict(data)
    except Exception:
        # Corrupt/incomplete snapshots are treated as absent for safety.
        return None


@dataclass
class ReuseDecision:
    reusable: bool
    reason: str
    snapshot: Snapshot | None = None


def evaluate_reuse(
    block_dir: str,
    config: Config,
    manifest_hash: str,
    asset_hash: str,
    current_schedule: LanguageSchedule,
) -> ReuseDecision:
    """Decide whether an on-disk block can be reused on resume."""
    snapshot = load_snapshot(block_dir)
    if snapshot is None:
        return ReuseDecision(False, "no_snapshot", None)
    if snapshot.snapshot_version != config.snapshot_version:
        return ReuseDecision(False, "version_mismatch", snapshot)
    if snapshot.config_hash != config.content_hash():
        return ReuseDecision(False, "config_changed", snapshot)
    if snapshot.manifest_hash != manifest_hash:
        return ReuseDecision(False, "manifest_changed", snapshot)
    if snapshot.asset_hash != asset_hash:
        return ReuseDecision(False, "asset_changed", snapshot)
    if snapshot.schedule_fingerprint != schedule_fingerprint(current_schedule):
        return ReuseDecision(False, "schedule_changed", snapshot)

    track_path = os.path.join(block_dir, snapshot.track_file)
    if not os.path.exists(track_path):
        return ReuseDecision(False, "track_missing", snapshot)
    if sha256_file(track_path) != snapshot.track_hash:
        return ReuseDecision(False, "track_hash_mismatch", snapshot)
    return ReuseDecision(True, "reused", snapshot)


def load_track(block_dir: str, snapshot: Snapshot) -> np.ndarray:
    track_path = os.path.join(block_dir, snapshot.track_file)
    with open(track_path, "rb") as handle:
        track = np.load(handle)
    if track.shape[0] != snapshot.total_frames:
        raise ValueError("reused track frame count disagrees with snapshot")
    return track.astype(np.float32, copy=False)


def revalidate_offsets(
    snapshot: Snapshot,
    current_schedule: LanguageSchedule,
    new_total_frames: int,
) -> list[dict]:
    """Re-check reused offsets against the new timeline instead of trusting
    stale stored file offsets.  Returns a list of violation dicts (empty if
    every retained clip is consistent and inside the new total duration).
    """
    violations: list[dict] = []
    by_id = {item.event_id: item for item in current_schedule.placements}
    stored = {entry["event_id"]: entry for entry in snapshot.index_entries}

    if set(by_id) != set(stored):
        violations.append(
            {
                "code": "E_OFFSET_REVALIDATION",
                "message": "reused block event set differs from current schedule",
                "detail": {
                    "current": sorted(by_id),
                    "snapshot": sorted(stored),
                },
            }
        )
        return violations

    for event_id, placement in by_id.items():
        entry = stored[event_id]
        checks = {
            "sample_offset": placement.start,
            "end_sample": placement.end,
            "frames": placement.frames,
        }
        for key, expected in checks.items():
            if entry.get(key) != expected:
                violations.append(
                    {
                        "code": "E_OFFSET_REVALIDATION",
                        "message": f"{event_id}: stored {key} stale",
                        "detail": {
                            "field": key,
                            "stored": entry.get(key),
                            "expected": expected,
                        },
                    }
                )
        if placement.end > new_total_frames:
            violations.append(
                {
                    "code": "E_OFFSET_REVALIDATION",
                    "message": f"{event_id}: end exceeds new total frames",
                    "detail": {
                        "end": placement.end,
                        "new_total_frames": new_total_frames,
                    },
                }
            )
    return violations
