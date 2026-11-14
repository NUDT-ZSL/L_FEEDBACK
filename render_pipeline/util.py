from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=False,
        indent=2,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def rounded(value: np.ndarray | float, digits: int = 6) -> Any:
    array = np.asarray(value, dtype=np.float64)
    result = np.round(array, decimals=digits)
    if result.ndim == 0:
        scalar = float(result)
        return 0.0 if abs(scalar) < 5e-7 else scalar
    result[np.abs(result) < 5e-7] = 0.0
    return result.tolist()


def require_dict(
    value: Any,
    *,
    code: str,
    path: str,
    message: str,
) -> dict[str, Any]:
    from .errors import PipelineError

    if not isinstance(value, dict):
        raise PipelineError(code=code, message=message, scope="scene", path=path)
    return value


def require_list(
    value: Any,
    *,
    code: str,
    path: str,
    message: str,
) -> list[Any]:
    from .errors import PipelineError

    if not isinstance(value, list):
        raise PipelineError(code=code, message=message, scope="scene", path=path)
    return value


def as_finite_float(value: Any, *, path: str, code: str) -> float:
    from .errors import PipelineError

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(
            code=code,
            message=f"{path} must be a finite number",
            scope="scene",
            path=path,
        )
    number = float(value)
    if not math.isfinite(number):
        raise PipelineError(
            code=code,
            message=f"{path} must be finite",
            scope="scene",
            path=path,
        )
    return number


def rgb_triplet(value: Any, *, path: str, code: str, nonnegative: bool = True) -> tuple[float, float, float]:
    from .errors import PipelineError, ErrorCode

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise PipelineError(
            code=code,
            message=f"{path} must be a three-element RGB array",
            scope="scene",
            path=path,
        )
    result = tuple(as_finite_float(item, path=f"{path}[{index}]", code=code) for index, item in enumerate(value))
    if nonnegative and min(result) < 0.0:
        raise PipelineError(
            code=ErrorCode.SCENE_FIELD_INVALID,
            message=f"{path} must be non-negative",
            scope="scene",
            path=path,
        )
    return result  # type: ignore[return-value]


def safe_id(value: Any, *, path: str) -> str:
    from .errors import ErrorCode, PipelineError

    if not isinstance(value, str) or not value:
        raise PipelineError(
            code=ErrorCode.SCENE_FIELD_INVALID,
            message=f"{path} must be a non-empty string",
            scope="scene",
            path=path,
        )
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not set(value) <= allowed or value in {".", ".."}:
        raise PipelineError(
            code=ErrorCode.SCENE_FIELD_INVALID,
            message=f"{path} may only use letters, digits, '.', '_' and '-'",
            scope="scene",
            path=path,
            context={"value": value},
        )
    return value


def clamp01(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0)
