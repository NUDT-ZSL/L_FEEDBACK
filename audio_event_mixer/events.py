"""Deterministic event-description parsing and validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .errors import BusinessError, ErrorCode


@dataclass(frozen=True)
class LoopSpec:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class EventSpec:
    index: int
    event_id: str
    source_path: Path
    trigger_seconds: float
    pan_frequency_hz: float
    pan_depth: float
    pan_phase_degrees: float = 0.0
    loop: Optional[LoopSpec] = None


_FILE_KEYS = (
    "file",
    "path",
    "file_path",
    "source",
    "source_path",
    "audio_file",
    "wav",
)
_TIME_KEYS = (
    "time",
    "trigger_time",
    "trigger_seconds",
    "trigger_time_seconds",
    "time_seconds",
    "start_time",
    "trigger",
)
_FREQ_KEYS = (
    "pan_frequency_hz",
    "pan_frequency",
    "pan_freq_hz",
    "pan_rate",
    "lfo_frequency_hz",
    "lfo_freq_hz",
    "panning_frequency_hz",
    "panning_frequency",
    "pan_lfo_frequency_hz",
    "frequency_hz",
    "freq_hz",
    "frequency",
)
_DEPTH_KEYS = ("pan_depth", "depth", "panning_depth")
_PHASE_KEYS = ("pan_phase_degrees", "phase_degrees")
_LOOP_KEYS = ("loop", "loop_region")
_LOOP_START_KEYS = (
    "start",
    "start_seconds",
    "start_time",
    "loop_start",
    "from",
)
_LOOP_END_KEYS = (
    "end",
    "end_seconds",
    "end_time",
    "loop_end",
    "to",
)


def _invalid(
    message: str,
    event_index: Optional[int] = None,
    details: Optional[dict] = None,
) -> BusinessError:
    return BusinessError(
        ErrorCode.INVALID_EVENT,
        message,
        event_index=event_index,
        details=details,
    )


def _first_present(mapping: dict, keys: tuple[str, ...]) -> tuple[Optional[str], Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _number(
    mapping: dict,
    keys: tuple[str, ...],
    label: str,
    event_index: int,
) -> float:
    key, value = _first_present(mapping, keys)
    if key is None:
        raise _invalid(f"missing required field '{label}'", event_index)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"field '{key}' must be a number", event_index)
    number = float(value)
    if not math.isfinite(number):
        raise _invalid(f"field '{key}' must be finite", event_index)
    return number


def _optional_number(
    mapping: dict,
    keys: tuple[str, ...],
    label: str,
    event_index: int,
    default: float,
) -> float:
    key, value = _first_present(mapping, keys)
    if key is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"field '{key}' must be a number", event_index)
    number = float(value)
    if not math.isfinite(number):
        raise _invalid(f"field '{key}' must be finite", event_index)
    return number


def _parse_loop(mapping: dict, event_index: int) -> Optional[LoopSpec]:
    loop_key, loop_value = _first_present(mapping, _LOOP_KEYS)
    if loop_key is None:
        top_start_key, top_start = _first_present(
            mapping, ("loop_start", "loop_start_seconds", "loop_start_time")
        )
        top_end_key, top_end = _first_present(
            mapping, ("loop_end", "loop_end_seconds", "loop_end_time")
        )
        if top_start_key is None and top_end_key is None:
            return None
        if top_start_key is None or top_end_key is None:
            raise _invalid(
                "loop_start and loop_end must be provided together",
                event_index,
            )
        start, end = top_start, top_end
    elif loop_value is None:
        return None
    elif isinstance(loop_value, (list, tuple)):
        if len(loop_value) != 2:
            raise _invalid("loop region must contain exactly [start, end]", event_index)
        start, end = loop_value
    elif isinstance(loop_value, dict):
        start_key, start = _first_present(loop_value, _LOOP_START_KEYS)
        end_key, end = _first_present(loop_value, _LOOP_END_KEYS)
        if start_key is None or end_key is None:
            raise _invalid(
                "loop region requires start and end fields",
                event_index,
            )
    else:
        raise _invalid("loop region must be an object, pair, or null", event_index)

    for label, value in (("loop start", start), ("loop end", end)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid(f"{label} must be a number", event_index)
        if not math.isfinite(float(value)):
            raise _invalid(f"{label} must be finite", event_index)
    start_number = float(start)
    end_number = float(end)
    if start_number < 0.0:
        raise _invalid("loop start must be >= 0 seconds", event_index)
    if end_number <= start_number:
        raise _invalid("loop end must be greater than loop start", event_index)
    return LoopSpec(start_seconds=start_number, end_seconds=end_number)


def _parse_event(
    raw: Any,
    index: int,
    base_dir: Path,
) -> EventSpec:
    if not isinstance(raw, dict):
        raise _invalid("event must be a JSON object", index)

    file_key, file_value = _first_present(raw, _FILE_KEYS)
    if file_key is None:
        raise _invalid("missing required field 'file'", index)
    if not isinstance(file_value, str) or not file_value.strip():
        raise _invalid(f"field '{file_key}' must be a non-empty string", index)
    source_path = Path(file_value).expanduser()
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    source_path = source_path.resolve(strict=False)

    trigger_seconds = _number(raw, _TIME_KEYS, "time", index)
    if trigger_seconds < 0.0:
        raise _invalid("trigger time must be >= 0 seconds", index)

    panning = raw.get("panning")
    pan_source = {**raw, **panning} if isinstance(panning, dict) else raw
    pan_frequency_hz = _number(
        pan_source, _FREQ_KEYS, "pan_frequency_hz", index
    )
    if pan_frequency_hz < 0.0:
        raise _invalid("panning frequency must be >= 0 Hz", index)
    pan_depth = _number(pan_source, _DEPTH_KEYS, "pan_depth", index)
    if not 0.0 <= pan_depth <= 1.0:
        raise _invalid("panning depth must be within [0.0, 1.0]", index)
    pan_phase_degrees = _optional_number(
        pan_source, _PHASE_KEYS, "pan_phase_degrees", index, 0.0
    )

    event_id_value = raw.get("id")
    event_id = (
        str(event_id_value)
        if event_id_value is not None
        else f"event_{index:04d}"
    )
    loop = _parse_loop(raw, index)

    return EventSpec(
        index=index,
        event_id=event_id,
        source_path=source_path,
        trigger_seconds=trigger_seconds,
        pan_frequency_hz=pan_frequency_hz,
        pan_depth=pan_depth,
        pan_phase_degrees=pan_phase_degrees,
        loop=loop,
    )


def load_events(
    path: Union[str, os.PathLike[str]],
    base_dir: Optional[Union[str, os.PathLike[str]]] = None,
) -> list[EventSpec]:
    """Load and validate a list of events.

    The top level may be an array, or an object containing an ``events`` or
    ``tracks`` array. Relative source paths are resolved against the JSON's
    directory unless an explicit base directory is provided.
    """

    events_path = Path(path).expanduser()
    if not events_path.is_file():
        raise BusinessError(
            ErrorCode.FILE_NOT_FOUND,
            "event description file does not exist",
            details={"path": str(events_path)},
        )
    try:
        with events_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise _invalid(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            details={"path": str(events_path)},
        ) from exc
    except OSError as exc:
        raise BusinessError(
            ErrorCode.FILE_NOT_FOUND,
            f"cannot read event description: {exc}",
            details={"path": str(events_path)},
        ) from exc

    if isinstance(payload, list):
        raw_events = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            raw_events = payload["events"]
        elif isinstance(payload.get("tracks"), list):
            raw_events = payload["tracks"]
        else:
            raise _invalid(
                "event JSON object must contain an 'events' or 'tracks' array"
            )
    else:
        raise _invalid("event JSON must be an array or object")

    if not raw_events:
        raise _invalid("event list must not be empty")

    resolve_base = (
        Path(base_dir).expanduser().resolve(strict=False)
        if base_dir is not None
        else events_path.resolve().parent
    )
    return [
        _parse_event(raw, index, resolve_base)
        for index, raw in enumerate(raw_events)
    ]
