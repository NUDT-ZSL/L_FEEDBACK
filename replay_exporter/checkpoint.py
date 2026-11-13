"""Persistent render checkpoints with atomic replace and half-write detection."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .errors import ErrorCode, ExporterError

CHECKPOINT_FILE = "render_checkpoint.json"


def checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, CHECKPOINT_FILE)


def write_checkpoint_atomic(output_dir: str, payload: Dict[str, Any]) -> str:
    """Temp file + fsync + os.replace: readers never observe a half-written file."""
    os.makedirs(output_dir, exist_ok=True)
    target = checkpoint_path(output_dir)
    temporary = target + ".tmp"
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    with open(temporary, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    path = checkpoint_path(output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExporterError(
            ErrorCode.CHECKPOINT_CORRUPT,
            f"checkpoint is unreadable and cannot be trusted: {exc}",
            detail={"path": path},
        ) from exc
    required = {"version", "completed_blocks", "master_data_bytes", "block_output_hashes"}
    missing = required - set(payload)
    if missing:
        raise ExporterError(
            ErrorCode.CHECKPOINT_CORRUPT,
            f"checkpoint missing fields: {sorted(missing)}",
            detail={"path": path, "missing": sorted(missing)},
        )
    return payload


def build_checkpoint(
    version: str,
    completed_blocks: List[int],
    master_data_bytes: int,
    block_output_hashes: Dict[str, str],
    master_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": version,
        "completed_blocks": sorted(completed_blocks),
        "master_data_bytes": master_data_bytes,
        "block_output_hashes": dict(sorted(block_output_hashes.items())),
        "master_path": os.path.abspath(master_path),
    }
    if extra:
        payload.update(extra)
    return payload


def verify_persisted_block(
    master_path: str,
    block_index: int,
    block_frames: int,
    frame_count: int,
    expected_hash: str,
    read_block,
    sha256_bytes,
) -> bool:
    """Re-read a block from disk and compare its PCM hash.

    A crash during ``write_block`` leaves the block partially overwritten or
    still zero-filled; its hash then fails and the block must be re-rendered.
    """
    if not os.path.exists(master_path):
        return False
    actual_size = os.path.getsize(master_path)
    required_size = 44 + (block_index * block_frames + frame_count) * 4
    if actual_size < required_size:
        return False
    try:
        pcm = read_block(block_index, block_frames, frame_count)
    except OSError:
        return False
    return sha256_bytes(pcm) == expected_hash
