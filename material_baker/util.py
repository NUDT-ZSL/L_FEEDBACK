"""Small deterministic helpers shared by the pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import MaterialError


REPORT_SCHEMA = "material-bake-validation-report/1"
ASSET_SCHEMA = "material-bake-assets/1"
RUN_SCHEMA = "material-bake-run/1"


def stable_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
        separators=(",", ": "),
    ) + "\n"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, stable_json_text(value).encode("utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_vector(value: Iterable[float], field: str) -> np.ndarray:
    vector = np.asarray(list(value), dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise MaterialError("E004", f"{field} must be a finite 3-component vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise MaterialError("E004", f"{field} must be non-zero")
    return vector / length


def ensure_finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MaterialError("E004", f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise MaterialError("E004", f"{field} must be finite")
    return number


def canonical_lights(lights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return lights in a content-derived order.

    The shading code walks this order. Reordering entries in the scene JSON
    therefore cannot change floating-point accumulation order.
    """

    def key(light: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(light.get("id", "")),
            tuple(float(v) for v in light["direction"]),
            tuple(float(v) for v in light["color"]),
            float(light["intensity"]),
        )

    return sorted(lights, key=key)


def safe_relative_path(path: Path, base: Path) -> str:
    try:
        relative = path.resolve().relative_to(base.resolve())
        return relative.as_posix()
    except ValueError:
        return path.name


def ensure_output_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise MaterialError("E009", "output filename must not be empty")
    if any(character in name for character in '/\\:*?"<>|'):
        raise MaterialError(
            "E009",
            "output filename contains a path separator or Windows-invalid character",
        )
    if any(ord(character) < 32 for character in name):
        raise MaterialError(
            "E009", "output filename must not contain control characters"
        )
    if name.startswith("."):
        raise MaterialError(
            "E009", "output filename must not start with a hidden-file dot"
        )
    return name
