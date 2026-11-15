from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union


PathLike = Union[str, os.PathLike]


def read_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(str(src), str(dst))


def write_json(path: PathLike, payload: Any) -> None:
    """Write deterministic UTF-8 JSON without a BOM, atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=False,
    )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
        _atomic_replace(Path(temporary), destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_text_atomic(path: PathLike, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        _atomic_replace(Path(temporary), destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
