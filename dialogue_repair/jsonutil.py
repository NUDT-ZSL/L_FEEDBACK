"""Stable JSON encoding, including fixed-width floating point rendering."""

from __future__ import annotations

import json
import math
from typing import Any


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("JSON output does not permit non-finite floats")
    if abs(value) < 0.0000000000005:
        value = 0.0
    return format(value, ".12f")


def _encode(value: Any, indent: int | None, level: int) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float_text(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if indent is None:
            return "[" + ",".join(_encode(item, None, level + 1) for item in value) + "]"
        pad = " " * (indent * (level + 1))
        close_pad = " " * (indent * level)
        body = (",\n").join(pad + _encode(item, indent, level + 1) for item in value)
        return "[\n" + body + "\n" + close_pad + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        if indent is None:
            pieces = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                pieces.append(
                    json.dumps(key, ensure_ascii=False)
                    + ":"
                    + _encode(item, None, level + 1)
                )
            return "{" + ",".join(pieces) + "}"
        pad = " " * (indent * (level + 1))
        close_pad = " " * (indent * level)
        pieces = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            pieces.append(
                pad
                + json.dumps(key, ensure_ascii=False)
                + ": "
                + _encode(item, indent, level + 1)
            )
        return "{\n" + ",\n".join(pieces) + "\n" + close_pad + "}"
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def fixed_json(value: Any, *, indent: int | None = 2) -> str:
    return _encode(value, indent, 0)

