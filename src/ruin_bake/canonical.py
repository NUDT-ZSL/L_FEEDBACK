"""Canonical serialization and hashing helpers.

All persisted JSON uses sorted keys, compact separators, and Python's
round-trippable float representation.  This removes dictionary insertion
order from hashes and makes the generated scene byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class SerializationError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"value is not canonical-JSON serializable: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def require_finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SerializationError(f"{label} must be a finite number")
    if not math.isfinite(float(value)):
        raise SerializationError(f"{label} must be finite")


def vec3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SerializationError(f"{label} must contain exactly three numbers")
    values = tuple(float(item) for item in value)
    for index, item in enumerate(values):
        require_finite(item, f"{label}[{index}]")
    return values


def rgb(value: Any, label: str) -> tuple[float, float, float]:
    values = vec3(value, label)
    for index, item in enumerate(values):
        if item < 0.0:
            raise SerializationError(f"{label}[{index}] must be non-negative")
    return values

