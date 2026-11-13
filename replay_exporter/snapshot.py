"""Versioned, deterministic intermediate plan snapshots.

A snapshot captures the normalized plan and an *input hash* for every render
block. Input hashes are SHA-256 over the exact source bytes and plan parameters
that influence the block, so they are available before rendering and stable
across machines. Rendered PCM output hashes live in the checkpoint instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .dsp import frames_to_ms
from .errors import ErrorCode, ExporterError
from .model import BlockDescriptor, Snapshot, TimelinePlan

SNAPSHOT_VERSION = "1.0.0"
SNAPSHOT_FILE = "plan_snapshot.json"


def canonical_json(payload: Any) -> str:
    """Stable serialization: sorted keys, fixed separators, no whitespace drift."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_hashes(plan: TimelinePlan) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for segment in plan.segments:
        for path in (segment.voice_wav, segment.ambience_wav):
            hashes.setdefault(path, sha256_file(path))
    return hashes


def describe_blocks(plan: TimelinePlan) -> List[BlockDescriptor]:
    block_count = plan.block_count()
    descriptors: List[BlockDescriptor] = []
    for index in range(block_count):
        start = index * plan.block_frames
        frame_count = min(plan.block_frames, plan.total_frames - start)
        descriptors.append(
            BlockDescriptor(
                index=index,
                start_frame=start,
                frame_count=frame_count,
                start_ms=round(frames_to_ms(start, plan.sample_rate), 6),
                end_ms=round(
                    frames_to_ms(start + frame_count, plan.sample_rate), 6
                ),
                input_hash="",
            )
        )
    return descriptors


def build_snapshot(plan: TimelinePlan, timeline_path: str) -> Snapshot:
    source_hashes = _source_hashes(plan)
    plan_dict = plan.to_dict()
    plan_dict["source_file_hashes"] = source_hashes

    blocks = describe_blocks(plan)
    for block in blocks:
        material = {
            "version": SNAPSHOT_VERSION,
            "block_index": block.index,
            "start_frame": block.start_frame,
            "frame_count": block.frame_count,
            "plan": plan_dict,
        }
        block.input_hash = sha256_bytes(canonical_json(material).encode("utf-8"))

    return Snapshot(
        version=SNAPSHOT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_timeline_path=os.path.abspath(timeline_path),
        plan=plan_dict,
        blocks=[block.__dict__ for block in blocks],
    )


def write_snapshot_atomic(output_dir: str, snapshot: Snapshot) -> str:
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, SNAPSHOT_FILE)
    temporary = target + ".tmp"
    payload = json.dumps(snapshot.to_dict(), sort_keys=True, indent=2)
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_snapshot(output_dir: str) -> Optional[Snapshot]:
    path = os.path.join(output_dir, SNAPSHOT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return Snapshot(
            version=raw["version"],
            created_at=raw["created_at"],
            source_timeline_path=raw["source_timeline_path"],
            plan=raw["plan"],
            blocks=raw.get("blocks", []),
            deprecated_blocks=raw.get("deprecated_blocks", []),
        )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ExporterError(
            ErrorCode.CHECKPOINT_CORRUPT,
            f"snapshot in {output_dir} is unreadable: {exc}",
            detail={"path": path},
        ) from exc


def reconcile_snapshots(
    old: Optional[Snapshot],
    new: Snapshot,
) -> Tuple[Snapshot, List[dict]]:
    """Decide which previously rendered blocks may be reused.

    * A version mismatch invalidates every old block: old byte offsets must not
      be reused, the whole snapshot is replaced and the deprecated block list is
      recorded inside the new snapshot.
    * Matching versions allow reuse only when the block input hash is unchanged.
    """
    deprecated: List[dict] = []
    if old is None:
        return new, deprecated

    if old.version != new.version:
        deprecated = [
            {
                "reason": "version_mismatch",
                "old_version": old.version,
                "new_version": new.version,
                "block_index": block.get("index"),
                "old_input_hash": block.get("input_hash"),
            }
            for block in old.blocks
        ]
        new.deprecated_blocks = deprecated
        return new, deprecated

    old_by_index = {block["index"]: block for block in old.blocks}
    new_by_index = {block["index"]: block for block in new.blocks}
    for index, new_block in new_by_index.items():
        previous = old_by_index.get(index)
        if previous is not None and previous.get("input_hash") == new_block["input_hash"]:
            new_block["reusable"] = True
            new_block["output_hash"] = previous.get("output_hash")
        else:
            deprecated.append(
                {
                    "reason": "input_hash_mismatch",
                    "block_index": index,
                    "old_input_hash": previous.get("input_hash") if previous else None,
                    "new_input_hash": new_block["input_hash"],
                }
            )
    for index, previous in old_by_index.items():
        if index not in new_by_index:
            deprecated.append(
                {
                    "reason": "block_removed",
                    "block_index": index,
                    "old_input_hash": previous.get("input_hash"),
                }
            )
    new.deprecated_blocks = deprecated
    return new, deprecated
