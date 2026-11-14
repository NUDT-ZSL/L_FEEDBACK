"""SHA-256 helpers used by the loadable JSON manifest."""

from __future__ import annotations

import hashlib


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def overall_sha256(per_file: dict[str, str]) -> str:
    """Stable overall checksum over sorted (path, digest) pairs."""
    digest = hashlib.sha256()
    for relative in sorted(per_file):
        digest.update(relative.encode("utf-8"))
        digest.update(b":")
        digest.update(per_file[relative].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
