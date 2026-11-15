"""Event manifest parsing, validation and global timeline construction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .config import SAMPLE_RATE
from .decoder import SUPPORTED_SUFFIXES
from .errors import ERROR_CATALOG, PipelineError

VALID_CATEGORIES = ("dialog", "sfx", "music")
VALID_OVERLAP = ("replace", "mix")


@dataclass
class Event:
    index: int
    event_id: str
    path: str
    start_ms: float
    offset_ms: float = 0.0
    duration_ms: float | None = None
    loop: bool = False
    priority: int = 0
    category: str = "sfx"
    overlap: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    base_dir: str = ""
    actual_start_ms: float | None = None
    actual_end_ms: float | None = None

    @property
    def absolute_path(self) -> str:
        return self.path if os.path.isabs(self.path) else os.path.join(self.base_dir, self.path)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_manifest(path: str) -> tuple[list[Event], float, str]:
    """Parse the manifest file.

    Returns ``(events, total_duration_ms, base_dir)``.  Fatal structural
    problems raise :class:`PipelineError`; per-event problems are recorded on
    the event so processing of the remaining events can continue.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise PipelineError("E_MANIFEST_PARSE", f"清单文件不存在: {path}")
    except json.JSONDecodeError as exc:
        raise PipelineError("E_MANIFEST_PARSE", f"JSON 解析失败: {exc}")
    if not isinstance(raw, dict):
        raise PipelineError("E_MANIFEST_PARSE", "清单顶层必须是 JSON 对象")

    if "total_duration_ms" not in raw:
        raise PipelineError("E_MANIFEST_FIELD", "缺少必填字段 total_duration_ms")
    total = raw["total_duration_ms"]
    if not _is_number(total) or total <= 0:
        raise PipelineError("E_DURATION_INVALID", f"total_duration_ms 非法: {total!r}")

    items = raw.get("events")
    if not isinstance(items, list):
        raise PipelineError("E_MANIFEST_FIELD", "缺少必填字段 events 或其不是数组")

    base_dir = os.path.dirname(os.path.abspath(path))
    events: list[Event] = []
    for index, item in enumerate(items):
        events.append(_parse_event(index, item, base_dir))
    return events, float(total), base_dir


def _parse_event(index: int, item, base_dir: str) -> Event:
    if not isinstance(item, dict):
        return Event(
            index=index,
            event_id=f"#{index}",
            path="",
            start_ms=0.0,
            errors=["E_FIELD_MISSING"],
            base_dir=base_dir,
        )

    errors: list[str] = []
    event_id = str(item.get("id", f"#{index}"))

    missing = [name for name in ("id", "path", "start_ms") if name not in item]
    if missing:
        errors.append("E_FIELD_MISSING")

    path = item.get("path", "")
    start_ms = item.get("start_ms", 0.0)
    if not _is_number(start_ms) or start_ms < 0:
        errors.append("E_TIME_INVALID")
        start_ms = 0.0

    offset_ms = item.get("offset_ms", 0.0)
    if not _is_number(offset_ms) or offset_ms < 0:
        errors.append("E_TIME_INVALID")
        offset_ms = 0.0

    duration_ms = item.get("duration_ms")
    if duration_ms is not None and (
        not _is_number(duration_ms) or duration_ms <= 0
    ):
        errors.append("E_TIME_INVALID")
        duration_ms = None

    loop = bool(item.get("loop", False))
    priority = item.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        errors.append("E_TIME_INVALID")
        priority = 0

    category = str(item.get("category", "sfx"))
    if category not in VALID_CATEGORIES:
        category = "sfx"

    overlap = item.get("overlap")
    if overlap is not None and overlap not in VALID_OVERLAP:
        overlap = None

    event = Event(
        index=index,
        event_id=event_id,
        path=str(path),
        start_ms=float(start_ms),
        offset_ms=float(offset_ms),
        duration_ms=(float(duration_ms) if duration_ms is not None else None),
        loop=loop,
        priority=int(priority),
        category=category,
        overlap=overlap,
        errors=errors,
        base_dir=base_dir,
    )
    _validate_file(event)
    return event


def _validate_file(event: Event) -> None:
    if "E_FIELD_MISSING" in event.errors or not event.path:
        return
    suffix = os.path.splitext(event.path)[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        event.errors.append("E_FORMAT_UNSUPPORTED")
        return
    if not os.path.isfile(event.absolute_path):
        event.errors.append("E_FILE_NOT_FOUND")


def ms_to_samples(ms: float) -> int:
    return int(round(float(ms) * SAMPLE_RATE / 1000.0))


def samples_to_ms(samples: int) -> float:
    return round(float(samples) * 1000.0 / SAMPLE_RATE, 6)


def validate_offset_against_source(event: Event, source_frames: int) -> None:
    """Mark ``E_TIME_INVALID`` when the in-asset offset passes the clip end."""
    offset_frames = ms_to_samples(event.offset_ms)
    if offset_frames >= source_frames:
        event.errors.append("E_TIME_INVALID")


__all__ = [
    "Event",
    "ERROR_CATALOG",
    "VALID_CATEGORIES",
    "load_manifest",
    "ms_to_samples",
    "samples_to_ms",
    "validate_offset_against_source",
]
