"""Deterministic serialization and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


STATS_DECIMALS = 6


def quantize(value: float, decimals: int = STATS_DECIMALS) -> float:
    """Quantize a report value using decimal place, not platform formatting."""

    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite report value: {value!r}")
    value = float(value)
    if abs(value) < 0.5 * 10.0 ** (-decimals):
        value = 0.0
    return round(value, decimals)


def quantize_vec(values: Any, decimals: int = STATS_DECIMALS) -> list[float]:
    return [quantize(float(v), decimals) for v in values]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def strip_volatile(value: Any) -> Any:
    """Remove fields explicitly allowed to vary between equivalent runs."""

    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in {"generated_at", "timestamp"}
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value

