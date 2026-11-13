"""Canonical JSON serialization used for deterministic reports."""

from __future__ import annotations

import json
import math
from typing import Any


def _escape(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _float(value: float) -> str:
    if not math.isfinite(value):
        return "null"
    if abs(value) < 0.000000000001:
        value = 0.0
    return f"{value:.6f}"


def dumps_fixed(value: Any, indent: int = 2, level: int = 0) -> str:
    """Serialize JSON with stable key order and six decimal float fields."""
    pad = " " * (indent * level)
    child_pad = " " * (indent * (level + 1))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, str):
        return _escape(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        items = [child_pad + dumps_fixed(item, indent, level + 1) for item in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for key, item in value.items():
            items.append(child_pad + _escape(str(key)) + ": " + dumps_fixed(item, indent, level + 1))
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    raise TypeError(f"unsupported report value: {type(value)!r}")


def write_report(path: str, report: dict[str, Any]) -> None:
    text = dumps_fixed(report) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def print_json(value: Any) -> None:
    print(dumps_fixed(value))
