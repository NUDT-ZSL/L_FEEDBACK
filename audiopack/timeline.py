"""timeline.json loading and strict field validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from numbers import Real

from .errors import E_MISSING_CLIP, E_RANGE, E_TIMELINE_FORMAT, PipelineError


def _is_int(value: object) -> bool:
    """True for genuine ints (``bool`` is intentionally rejected)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """True for int/float (``bool`` rejected), excluding NaN/inf."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    text = str(value)
    return "nan" not in text.lower() and "inf" not in text.lower()


@dataclass(frozen=True)
class Event:
    """One validated timeline event with its original event index."""

    index: int
    start_ms: float
    end_ms: float
    clip: str
    fade_in_ms: float
    fade_out_ms: float
    gain_db: float
    priority: int
    loop: bool


def load_timeline(input_dir: str) -> tuple[list[Event], int]:
    """Read and validate ``timeline.json``; return events and target rate.

    Raises :class:`PipelineError` with ``E_TIMELINE_FORMAT`` for structural
    problems and ``E_RANGE`` for out-of-range field values.
    """
    timeline_path = os.path.join(input_dir, "timeline.json")
    if not os.path.isfile(timeline_path):
        raise PipelineError(
            E_TIMELINE_FORMAT,
            {"file": timeline_path, "reason": "timeline.json not found"},
        )

    try:
        with open(timeline_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(
            E_TIMELINE_FORMAT,
            {"file": timeline_path, "reason": f"invalid JSON: {exc}"},
        ) from exc

    if not isinstance(raw, dict):
        raise PipelineError(
            E_TIMELINE_FORMAT, {"reason": "top-level JSON value must be an object"}
        )

    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise PipelineError(
            E_TIMELINE_FORMAT, {"reason": "'events' must be a non-empty array"}
        )

    target_sr = raw.get("sample_rate", 48000)
    if not _is_int(target_sr) or not (1 <= target_sr <= 1_000_000):
        raise PipelineError(
            E_RANGE,
            {"field": "sample_rate", "value": target_sr, "reason": "must be a positive integer"},
        )

    events: list[Event] = []
    for index, item in enumerate(raw_events):
        events.append(_validate_event(item, index, input_dir))

    return events, int(target_sr)


def _validate_event(item: object, index: int, input_dir: str) -> Event:
    prefix = {"event_index": index}
    if not isinstance(item, dict):
        raise PipelineError(
            E_TIMELINE_FORMAT, {**prefix, "reason": "event must be an object"}
        )

    # Required fields and their expected numeric kind.
    numeric_fields = ("start_ms", "end_ms", "fade_in_ms", "fade_out_ms", "gain_db")
    for field in numeric_fields:
        if field not in item:
            raise PipelineError(
                E_TIMELINE_FORMAT, {**prefix, "field": field, "reason": "missing field"}
            )
        if not _is_number(item[field]):
            raise PipelineError(
                E_TIMELINE_FORMAT,
                {**prefix, "field": field, "value": item[field], "reason": "must be a finite number"},
            )

    clip = item.get("clip")
    if not isinstance(clip, str) or not clip.strip():
        raise PipelineError(
            E_TIMELINE_FORMAT,
            {**prefix, "field": "clip", "value": clip, "reason": "must be a non-empty string"},
        )

    priority = item.get("priority", 0)
    if not _is_int(priority):
        raise PipelineError(
            E_TIMELINE_FORMAT,
            {**prefix, "field": "priority", "value": priority, "reason": "must be an integer"},
        )

    loop = item.get("loop", False)
    if not isinstance(loop, bool):
        raise PipelineError(
            E_TIMELINE_FORMAT,
            {**prefix, "field": "loop", "value": loop, "reason": "must be a boolean"},
        )

    start_ms = float(item["start_ms"])
    end_ms = float(item["end_ms"])
    fade_in_ms = float(item["fade_in_ms"])
    fade_out_ms = float(item["fade_out_ms"])
    gain_db = float(item["gain_db"])

    if start_ms < 0 or end_ms < 0:
        raise PipelineError(
            E_RANGE, {**prefix, "reason": "start_ms/end_ms must be >= 0"}
        )
    if end_ms <= start_ms:
        raise PipelineError(
            E_RANGE,
            {**prefix, "start_ms": start_ms, "end_ms": end_ms, "reason": "end_ms must be greater than start_ms"},
        )
    if fade_in_ms < 0 or fade_out_ms < 0:
        raise PipelineError(
            E_RANGE, {**prefix, "reason": "fade durations must be >= 0"}
        )
    if not (-120.0 <= gain_db <= 120.0):
        raise PipelineError(
            E_RANGE,
            {**prefix, "field": "gain_db", "value": gain_db, "reason": "out of allowed range [-120, 120]"},
        )

    duration_ms = end_ms - start_ms
    if fade_in_ms + fade_out_ms > duration_ms:
        raise PipelineError(
            E_RANGE,
            {
                **prefix,
                "reason": "fade_in_ms + fade_out_ms exceeds event duration",
                "duration_ms": duration_ms,
            },
        )

    # Clip existence on disk is reported under E_MISSING_CLIP.
    clip_path = os.path.join(input_dir, clip)
    if not os.path.isfile(clip_path):
        raise PipelineError(
            E_MISSING_CLIP, {**prefix, "clip": clip, "reason": "clip file not found"}
        )

    return Event(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        clip=clip,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
        gain_db=gain_db,
        priority=int(priority),
        loop=bool(loop),
    )
