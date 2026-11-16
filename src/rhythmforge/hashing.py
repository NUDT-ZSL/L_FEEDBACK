"""Stable content hashing helpers used by the incremental build manifest."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(payload: dict | list) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_json(payload: dict | list) -> str:
    return sha256_bytes(stable_json_bytes(payload))


def sha256_array(array: np.ndarray) -> str:
    """Hash raw array bytes together with shape/dtype metadata."""
    digest = hashlib.sha256()
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def hash_chart_tree(chart: dict) -> str:
    """Hash the semantic chart content independent of file formatting."""
    return sha256_json(chart)


def hash_source_files(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(paths):
        if not os.path.isfile(path):
            from .errors import BuildError, ExitCode

            raise BuildError(ExitCode.MISSING_SOURCE,
                             f"source wav not found: {path}",
                             {"path": path})
        result[os.path.basename(path)] = sha256_file(path)
    return result
