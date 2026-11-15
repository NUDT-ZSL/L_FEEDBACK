"""Deterministic serialization, hashing, and time conversion helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .constants import SECONDS_DECIMAL_PLACES


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    quantum = Decimal(1).scaleb(-SECONDS_DECIMAL_PLACES)
    value = (Decimal(int(samples)) / Decimal(sample_rate)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    return float(value)


def seconds_text(samples: int, sample_rate: int) -> str:
    return f"{samples_to_seconds(samples, sample_rate):.{SECONDS_DECIMAL_PLACES}f}"


def seconds_to_samples(seconds: float | str, sample_rate: int) -> int:
    value = Decimal(str(seconds)) * Decimal(sample_rate)
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def alignment_error_samples(
    samples: int, seconds: float | str, sample_rate: int
) -> int:
    return abs(seconds_to_samples(seconds, sample_rate) - int(samples))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def read_json_file(path: Path) -> Any:
    with path.open("rb") as handle:
        return loads_json_strict(handle.read())


def loads_json_strict(data: bytes | str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    text = data.decode("utf-8") if isinstance(data, bytes) else data
    return json.loads(text, parse_constant=reject_constant)

