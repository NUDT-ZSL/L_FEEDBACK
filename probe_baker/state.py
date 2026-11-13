"""Persistent bake state: per-block files + manifest, all written atomically
(temp file + os.replace). A bake interrupted at any point (including
taskkill /F / kill -9) can resume from the last fully persisted block."""
import json
import os
import struct

import numpy as np

from .errors import StateCorrupt

BLOCK_MAGIC = b"PBBLK001"
BLOCK_HDR_LEN = struct.Struct("<I")
MANIFEST_NAME = "manifest.json"


def _atomic_write(path: str, payload: bytes) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _block_path(state_dir: str, bx: int, by: int) -> str:
    return os.path.join(state_dir, f"block_{bx:05d}_{by:05d}.pblk")


def write_block(state_dir, bx, by, meta: dict, data: np.ndarray) -> None:
    """meta carries version hash, block coords, progress flag and shape."""
    arr = np.ascontiguousarray(data, dtype="<f4")
    header = dict(meta)
    header.update(
        bx=bx, by=by, done=True, dtype="float32le", shape=list(arr.shape)
    )
    blob = json.dumps(header, sort_keys=True).encode("utf-8")
    payload = (
        BLOCK_MAGIC + BLOCK_HDR_LEN.pack(len(blob)) + blob + arr.tobytes()
    )
    _atomic_write(_block_path(state_dir, bx, by), payload)


def read_block(state_dir, bx, by, expected_version: str):
    """Returns float32 ndarray or None if missing/invalid/stale."""
    path = _block_path(state_dir, bx, by)
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    if len(blob) < len(BLOCK_MAGIC) + 4 or not blob.startswith(BLOCK_MAGIC):
        return None
    (hlen,) = BLOCK_HDR_LEN.unpack(
        blob[len(BLOCK_MAGIC):len(BLOCK_MAGIC) + 4]
    )
    try:
        header = json.loads(
            blob[len(BLOCK_MAGIC) + 4:len(BLOCK_MAGIC) + 4 + hlen].decode("utf-8")
        )
    except (ValueError, UnicodeDecodeError):
        return None
    if (
        header.get("version_hash") != expected_version
        or header.get("bx") != bx
        or header.get("by") != by
        or not header.get("done")
    ):
        return None
    shape = header.get("shape")
    data = blob[len(BLOCK_MAGIC) + 4 + hlen:]
    count = int(np.prod(shape)) if shape else -1
    if count <= 0 or len(data) != count * 4:
        return None
    return np.frombuffer(data, dtype="<f4").reshape(shape).copy()


def write_manifest(state_dir: str, manifest: dict) -> None:
    _atomic_write(
        os.path.join(state_dir, MANIFEST_NAME),
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def read_manifest(state_dir: str):
    path = os.path.join(state_dir, MANIFEST_NAME)
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def wipe_state(state_dir: str) -> None:
    """Drop every persisted block and the manifest. Used on version or
    parameter change: old block offsets/counts are never reused."""
    if not os.path.isdir(state_dir):
        return
    for name in os.listdir(state_dir):
        if name == MANIFEST_NAME or name.endswith(".pblk"):
            os.unlink(os.path.join(state_dir, name))


def require_manifest(state_dir: str) -> dict:
    manifest = read_manifest(state_dir)
    if manifest is None:
        raise StateCorrupt("no readable manifest in state dir", dir=state_dir)
    return manifest
