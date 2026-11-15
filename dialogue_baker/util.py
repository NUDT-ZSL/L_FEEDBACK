from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Union

from .errors import BakerError


PathLike = Union[str, os.PathLike[str]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_bytes(path: PathLike, logical_name: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except FileNotFoundError as exc:
        raise BakerError(
            "E_MISSING_INPUT",
            "Required input file is missing.",
            {"path": str(path), "logical_name": logical_name},
        ) from exc
    except OSError as exc:
        raise BakerError(
            "E_INPUT_READ",
            "Required input file could not be read.",
            {"path": str(path), "logical_name": logical_name, "reason": str(exc)},
        ) from exc


def load_json_bytes(data: bytes, logical_name: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BakerError(
            "E_INVALID_JSON",
            "Input JSON is invalid.",
            {"logical_name": logical_name, "reason": str(exc)},
        ) from exc


def atomic_write(path: PathLike, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".tmp-" + destination.name + "-",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BakerError(
            "E_INVALID_CONFIG",
            "Expected a finite number.",
            {"field": field, "value": value},
        )
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise BakerError(
            "E_INVALID_CONFIG",
            "Expected a finite number.",
            {"field": field, "value": value},
        )
    return number


def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BakerError(
            "E_INVALID_SCHEMA",
            "Expected a JSON object.",
            {"field": field, "actual_type": type(value).__name__},
        )
    return value

