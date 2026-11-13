"""Small deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_root_hash(root: Path, extensions: Iterable[str] | None = None) -> str:
    """Hash relative path plus bytes for every source file below *root*."""

    extensions = tuple(
        extensions
        or (".c", ".h", ".cc", ".cpp", ".c++", ".hpp", ".hh", ".hxx")
    )
    records: list[tuple[str, str]] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(current) / filename
            if extensions and path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(root).as_posix()
            records.append((relative, sha256_file(path)))
    digest = hashlib.sha256()
    for relative, file_hash in sorted(records):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def deterministic_token(seed: int, role: str, length: int = 8) -> str:
    return hashlib.sha256(f"{seed}:{role}".encode("ascii")).hexdigest()[:length]

