from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from .config import ExportConfig
from .errors import BUSINESS_CODES, EventValidationError


@dataclass(frozen=True)
class EventSpec:
    index: int
    start_ms: Decimal
    duration_ms: Decimal
    category: str
    rate: Decimal
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_ms": _decimal_text(self.start_ms),
            "duration_ms": _decimal_text(self.duration_ms),
            "end_ms": _decimal_text(self.start_ms + self.duration_ms),
            "category": self.category,
            "rate": _decimal_text(self.rate),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


def _as_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InvalidOperation("not a number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise InvalidOperation("not a number") from exc
    if not result.is_finite():
        raise InvalidOperation("not finite")
    return result


def _round_frames(milliseconds: Decimal, sample_rate: int) -> int:
    return int(milliseconds * sample_rate / Decimal(1000) + Decimal("0.5"))


def _error(code: str, message: str, details: dict) -> dict:
    return {
        "code": code,
        "code_number": BUSINESS_CODES[code]["number"],
        "message": message,
        "details": details,
    }


def parse_events(
    path: str | Path,
    source_frames: int,
    config: ExportConfig,
) -> list[EventSpec]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EventValidationError(
            [
                _error(
                    "EVENT_JSON_INVALID",
                    f"Cannot read event table: {path}",
                    {"path": str(path), "reason": str(exc)},
                )
            ]
        ) from exc
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EventValidationError(
            [
                _error(
                    "EVENT_JSON_INVALID",
                    f"Event table is not valid JSON: {path}",
                    {"path": str(path), "reason": str(exc)},
                )
            ]
        ) from exc

    default_rate: Any = 1.0
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and "events" in payload:
        items = payload["events"]
        if "default_rate" in payload:
            default_rate = payload["default_rate"]
    else:
        raise EventValidationError(
            [
                _error(
                    "EVENT_JSON_INVALID",
                    "Event table must be a JSON array or an object with an 'events' array.",
                    {"path": str(path), "actual_type": type(payload).__name__},
                )
            ]
        )
    if not isinstance(items, list):
        raise EventValidationError(
            [
                _error(
                    "EVENT_JSON_INVALID",
                    "The 'events' member must be an array.",
                    {"path": str(path), "actual_type": type(items).__name__},
                )
            ]
        )

    errors: list[dict] = []
    events: list[EventSpec] = []
    previous_start: Decimal | None = None
    previous_end_frame: int | None = None
    for index, item in enumerate(items):
        where = {"event_index": index}
        if not isinstance(item, dict):
            errors.append(
                _error("EVENT_ITEM_INVALID", "Event must be a JSON object.", where)
            )
            continue

        start_value = _first_present(
            item, ("start_ms", "timestamp_ms", "time_ms", "t_ms")
        )
        duration_value = _first_present(item, ("duration_ms", "duration"))
        category_value = _first_present(item, ("category", "type", "kind"))
        rate_value = _first_present(item, ("rate", "speed"))
        if rate_value is None:
            rate_value = default_rate

        if duration_value is None:
            errors.append(
                _error(
                    "EVENT_ITEM_INVALID",
                    "Event is missing duration_ms.",
                    {**where, "available_keys": sorted(item.keys())},
                )
            )
            continue
        if category_value is None:
            errors.append(
                _error(
                    "EVENT_ITEM_INVALID",
                    "Event is missing category.",
                    {**where, "available_keys": sorted(item.keys())},
                )
            )
            continue
        if start_value is None:
            errors.append(
                _error(
                    "EVENT_ITEM_INVALID",
                    "Event is missing start_ms.",
                    {**where, "available_keys": sorted(item.keys())},
                )
            )
            continue
        if not isinstance(category_value, str) or not category_value.strip():
            errors.append(
                _error(
                    "EVENT_ITEM_INVALID",
                    "Event category must be a non-empty string.",
                    where,
                )
            )
            continue
        try:
            start_ms = _as_decimal(start_value)
            duration_ms = _as_decimal(duration_value)
            rate = _as_decimal(rate_value)
        except InvalidOperation:
            errors.append(
                _error(
                    "EVENT_ITEM_INVALID",
                    "Event timestamp, duration, and rate must be finite numbers.",
                    where,
                )
            )
            continue

        if start_ms < 0:
            errors.append(
                _error(
                    "EVENT_TIMESTAMP_INVALID",
                    "Event start timestamp must be non-negative.",
                    {**where, "start_ms": _decimal_text(start_ms)},
                )
            )
            continue
        if duration_ms <= 0:
            errors.append(
                _error(
                    "EVENT_DURATION_INVALID",
                    "Event duration must be positive.",
                    {**where, "duration_ms": _decimal_text(duration_ms)},
                )
            )
            continue
        if rate <= 0:
            errors.append(
                _error(
                    "EVENT_RATE_INVALID",
                    "Event playback rate must be positive.",
                    {**where, "rate": _decimal_text(rate)},
                )
            )
            continue
        if rate < Decimal(str(config.varispeed_min_rate)) or rate > Decimal(
            str(config.varispeed_max_rate)
        ):
            errors.append(
                _error(
                    "EVENT_RATE_INVALID",
                    "Event playback rate is outside the configured supported range.",
                    {
                        **where,
                        "rate": _decimal_text(rate),
                        "min_rate": config.varispeed_min_rate,
                        "max_rate": config.varispeed_max_rate,
                    },
                )
            )
            continue

        category = category_value.strip()
        if category not in config.event_categories:
            errors.append(
                _error(
                    "EVENT_UNKNOWN_CATEGORY",
                    f"Unknown event category: {category}",
                    {
                        **where,
                        "category": category,
                        "allowed_categories": list(config.event_categories),
                    },
                )
            )
            continue

        if previous_start is not None and start_ms < previous_start:
            errors.append(
                _error(
                    "EVENT_ORDER",
                    "Events must be sorted by ascending start timestamp.",
                    {
                        **where,
                        "start_ms": _decimal_text(start_ms),
                        "previous_start_ms": _decimal_text(previous_start),
                    },
                )
            )
        previous_start = start_ms

        start_frame = _round_frames(start_ms, config.sample_rate)
        end_ms = start_ms + duration_ms
        end_frame = _round_frames(end_ms, config.sample_rate)
        if end_frame <= start_frame:
            errors.append(
                _error(
                    "EVENT_DURATION_INVALID",
                    "Event duration rounds to zero samples.",
                    {
                        **where,
                        "duration_ms": _decimal_text(duration_ms),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    },
                )
            )
            continue
        if start_frame > source_frames or end_frame > source_frames:
            errors.append(
                _error(
                    "EVENT_OUT_OF_RANGE",
                    "Event window extends beyond the narration track.",
                    {
                        **where,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "source_frames": source_frames,
                        "start_ms": _decimal_text(start_ms),
                        "end_ms": _decimal_text(end_ms),
                    },
                )
            )
        if previous_end_frame is not None and start_frame < previous_end_frame:
            errors.append(
                _error(
                    "EVENT_OVERLAP",
                    "Event window overlaps the previous event.",
                    {
                        **where,
                        "start_frame": start_frame,
                        "previous_end_frame": previous_end_frame,
                    },
                )
            )
        previous_end_frame = end_frame

        events.append(
            EventSpec(
                index=index,
                start_ms=start_ms,
                duration_ms=duration_ms,
                category=category,
                rate=rate,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    if errors:
        raise EventValidationError(errors)
    return events


def _first_present(item: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def rate_fraction(rate: Decimal) -> Fraction:
    return Fraction(rate)
