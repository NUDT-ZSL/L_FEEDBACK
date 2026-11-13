"""Content-addressed blob store with atomic temp-file writes.

Blobs live under ``<output>/blobs/ab/cd/<sha256>``. New blobs are written to a
unique temporary file and renamed into place atomically, so a blob is never
referenced before its bytes are durable. Stale ``*.tmp`` files left by a
crashed run are removed at startup.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path


class BlobStore:
    def __init__(self, output_dir: str | Path):
        self.root = Path(output_dir) / "blobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, content_hash: str) -> Path:
        return self.root / content_hash[:2] / content_hash[2:4] / content_hash

    def exists(self, content_hash: str) -> bool:
        return self.path_for(content_hash).is_file()

    def write_atomic(self, content_hash: str, source_path: str | Path) -> Path:
        """Move a fully-written temp file to its final content-addressed path.

        The rename is atomic on both POSIX and Windows (os.replace). A blob
        already present means identical content, so the temp file is dropped.
        """
        target = self.path_for(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            try:
                os.remove(source_path)
            except OSError:
                pass
            return target
        os.replace(source_path, target)
        return target

    def new_temp_file(self):
        """Create a unique temporary file inside the blobs directory."""
        fd, name = tempfile.mkstemp(
            prefix=f".{uuid.uuid4().hex}.", suffix=".tmp", dir=str(self.root)
        )
        return fd, name

    def cleanup_temp(self) -> int:
        """Remove orphaned temp files from an interrupted run. Returns count."""
        removed = 0
        for entry in self.root.rglob("*.tmp"):
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
        # Also clean dot-prefixed temp names created by new_temp_file.
        for entry in self.root.glob(".*.tmp"):
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def all_hashes(self) -> set[str]:
        result = set()
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                result.add(path.name)
        return result

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
