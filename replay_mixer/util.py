from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, BinaryIO

from .errors import invalid_config, missing_input


_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(value: float, floor: float = 1.0e-12) -> float:
    if value <= 0.0:
        return -240.0
    return 20.0 * math.log10(max(value, floor))


def rms_to_db(value: float) -> float:
    return lin_to_db(value)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as handle:
        while True:
            size = 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                size = min(size, remaining)
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def read_hash_prefix(path: Path, byte_count: int) -> str:
    if byte_count == 0:
        return sha256_bytes(b"")
    if not path.exists():
        raise FileNotFoundError(path)
    return sha256_file(path, limit=byte_count)


def resolve_relative_path(base_dir: Path, raw_path: Any, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise invalid_config(f"{field} must be a non-empty relative path", field=field)
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//") or _DRIVE_RE.match(normalized):
        raise invalid_config(f"{field} must be relative, not absolute", field=field, value=raw_path)
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise invalid_config(
            f"{field} must not be empty or escape the configuration directory",
            field=field,
            value=raw_path,
        )
    candidate = (base_dir / Path(*parts)).resolve()
    base_resolved = base_dir.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise invalid_config(
            f"{field} escapes the configuration directory",
            field=field,
            value=raw_path,
        ) from None
    return candidate


def ensure_existing_file(path: Path, role: str, display_path: str) -> None:
    if not path.is_file():
        raise missing_input(display_path, role)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


def atomic_replace(source: Path, destination: Path) -> None:
    last_error: PermissionError | None = None
    for delay in (0.0, 0.01, 0.03, 0.08, 0.2):
        if delay:
            time.sleep(delay)
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            # Some Windows network/UNC shares reject ReplaceFile semantics when
            # the target already exists. Remove it and retry after a short
            # delete-pending interval.
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
    assert last_error is not None
    raise last_error


def fsync_handle(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())
