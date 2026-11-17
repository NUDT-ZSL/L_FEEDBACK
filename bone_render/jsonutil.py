from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is a hard runtime dependency
        np = None

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def dumps_fixed(value: Any, *, indent: int = 2, decimals: int = 6) -> str:
    """Serialize JSON with all floating point numbers at a fixed precision."""

    value = _plain(value)

    def encode(node: Any, level: int) -> str:
        prefix = " " * (indent * (level + 1))
        suffix = " " * (indent * level)
        if node is None:
            return "null"
        if node is True:
            return "true"
        if node is False:
            return "false"
        if isinstance(node, str):
            return json.dumps(node, ensure_ascii=False)
        if isinstance(node, int):
            return str(node)
        if isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError("non-finite float is not valid statistics JSON")
            text = f"{node:.{decimals}f}"
            if text == f"-0.{('0' * decimals)}":
                text = f"0.{('0' * decimals)}"
            return text
        if isinstance(node, list):
            if not node:
                return "[]"
            body = ",\n".join(prefix + encode(item, level + 1) for item in node)
            return "[\n" + body + "\n" + suffix + "]"
        if isinstance(node, dict):
            if not node:
                return "{}"
            items = []
            for key in sorted(node):
                encoded_key = json.dumps(str(key), ensure_ascii=False)
                items.append(prefix + encoded_key + ": " + encode(node[key], level + 1))
            return "{\n" + ",\n".join(items) + "\n" + suffix + "}"
        raise TypeError(f"unsupported JSON value: {type(node)!r}")

    return encode(value, 0)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json_fixed(path: Path, value: Any) -> bytes:
    data = (dumps_fixed(value) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)
    return data
