"""Atomic artifact writers and output-directory management."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import ValidationError


CSV_FIELDS = (
    "instance_index",
    "material_name",
    "color_file",
    "normal_file",
    "depth_file",
    "object_id_file",
    "render_time_ms",
    "state_hash",
)

MANAGED_LOG_FILES = (
    "render_log.csv",
    "summary.json",
    "channel_map.json",
    "state_snapshots.jsonl",
)
_MANAGED_NPY = re.compile(
    r"^instance_\d{3,}_[A-Za-z0-9_.-]+_(color|normal|depth|object_id)\.npy$"
)


def io_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_IO_001", 1, message, details)


def prepare_output_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise io_error(f"cannot create output directory '{path}': {exc}", path=str(path)) from exc
    if not path.is_dir():
        raise io_error(f"output path is not a directory: {path}", path=str(path))
    try:
        for name in MANAGED_LOG_FILES:
            candidate = path / name
            if candidate.is_file():
                candidate.unlink()
        for candidate in path.glob("instance_*.npy"):
            if candidate.is_file() and _MANAGED_NPY.match(candidate.name):
                candidate.unlink()
    except OSError as exc:
        raise io_error(
            f"cannot clean managed artifacts in '{path}': {exc}", path=str(path)
        ) from exc


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise io_error(f"cannot write '{path}': {exc}", path=str(path)) from exc
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False
        ) as handle:
            temporary = handle.name
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise io_error(f"cannot write NumPy channel '{path}': {exc}", path=str(path)) from exc
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise io_error(f"cannot write CSV '{path}': {exc}", path=str(path)) from exc
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def write_json_lines(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for record in records
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_write_bytes(path, payload)


def channel_filename(
    instance_index: int,
    material_slug: str,
    channel: str,
) -> str:
    return f"instance_{instance_index:03d}_{material_slug}_{channel}.npy"

