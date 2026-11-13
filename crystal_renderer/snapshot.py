"""Versioned, checksummed snapshots of the float accumulator."""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

from . import SNAPSHOT_VERSION
from . import errors


def _checksum(frame: int, buffer: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(SNAPSHOT_VERSION).encode("ascii"))
    digest.update(str(frame).encode("ascii"))
    digest.update(np.ascontiguousarray(buffer).tobytes())
    return digest.hexdigest()


def save_snapshot(directory: str, frame: int, acc: np.ndarray) -> str:
    os.makedirs(directory, exist_ok=True)
    buffer_path = os.path.join(directory, f"snap_{frame:05d}.npy")
    meta_path = os.path.join(directory, f"snap_{frame:05d}.meta.json")
    np.save(buffer_path, np.ascontiguousarray(acc))
    meta = {
        "version": SNAPSHOT_VERSION,
        "frame": frame,
        "shape": list(acc.shape),
        "checksum": _checksum(frame, acc),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    return meta_path


def _load_one(directory: str, frame: int, log: errors.ErrorLog):
    buffer_path = os.path.join(directory, f"snap_{frame:05d}.npy")
    meta_path = os.path.join(directory, f"snap_{frame:05d}.meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        acc = np.load(buffer_path)
    except (OSError, json.JSONDecodeError):
        log.emit(errors.E_SNAPSHOT_CHECKSUM, None, f"unreadable snapshot at frame {frame}")
        return None
    if meta.get("version") != SNAPSHOT_VERSION:
        log.emit(
            errors.E_SNAPSHOT_VERSION,
            None,
            f"found={meta.get('version')!r} want={SNAPSHOT_VERSION}",
        )
        return None
    if meta.get("checksum") != _checksum(int(meta.get("frame", -1)), acc):
        log.emit(errors.E_SNAPSHOT_CHECKSUM, None, f"frame={meta.get('frame')}")
        return None
    return int(meta["frame"]), acc


def load_latest_snapshot(directory: str, log: errors.ErrorLog):
    if not os.path.isdir(directory):
        return None
    frames = []
    for name in os.listdir(directory):
        match = re.fullmatch(r"snap_(\d+)\.meta\.json", name)
        if match:
            frames.append(int(match.group(1)))
    if not frames:
        return None
    latest = max(frames)
    return _load_one(directory, latest, log)
