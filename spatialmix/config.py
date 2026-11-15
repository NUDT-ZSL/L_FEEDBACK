"""JSON project parsing and input validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .errors import ToolError
from .spatial import DEFAULT_MIN_GAIN

CONFLICT_POLICIES = ("mix", "drop", "delay")
FIT_STRATEGIES = ("stretch", "crop")


@dataclass(frozen=True)
class EventSpec:
    index: int
    filename: str
    source_path: Path
    source_x: float
    source_y: float
    trigger_time: float
    duration: float
    priority: float
    conflict_policy: str
    fit_strategy: str


@dataclass(frozen=True)
class ProjectConfig:
    listener_x: float
    listener_y: float
    min_gain: float
    events: List[EventSpec]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    value: Any,
    field: str,
    minimum: Optional[float] = None,
    strictly_greater: bool = False,
    maximum: Optional[float] = None,
) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ToolError("{}: must be a finite number".format(field))
    number = float(value)
    if minimum is not None:
        invalid = number <= minimum if strictly_greater else number < minimum
        if invalid:
            comparator = "greater than" if strictly_greater else "at least"
            raise ToolError(
                "{}: must be {} {}".format(field, comparator, minimum)
            )
    if maximum is not None and number > maximum:
        raise ToolError("{}: must be at most {}".format(field, maximum))
    return number


def _first_present(
    mapping: dict,
    names: Sequence[str],
    field: str,
) -> Tuple[str, Any]:
    for name in names:
        if name in mapping:
            return name, mapping[name]
    raise ToolError("{}: missing required field".format(field))


def _optional_first_present(
    mapping: dict,
    names: Sequence[str],
) -> Tuple[Optional[str], Any]:
    for name in names:
        if name in mapping:
            return name, mapping[name]
    return None, None


def _parse_position(value: Any, field: str) -> Tuple[float, float]:
    if isinstance(value, dict):
        if "x" not in value:
            raise ToolError("{}: missing required field 'x'".format(field))
        if "y" not in value:
            raise ToolError("{}: missing required field 'y'".format(field))
        x = _number(value["x"], field + ".x")
        y = _number(value["y"], field + ".y")
        return x, y
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ToolError("{}: must contain exactly two numbers [x, y]".format(field))
        x = _number(value[0], field + "[0]")
        y = _number(value[1], field + "[1]")
        return x, y
    raise ToolError("{}: must be an object with x/y or a two-element array".format(field))


def _policy(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolError("{}: must be a string".format(field))
    policy = value.strip().lower()
    if policy not in CONFLICT_POLICIES:
        raise ToolError(
            "{}: unknown conflict policy '{}' (expected one of: {})".format(
                field, value, ", ".join(CONFLICT_POLICIES)
            )
        )
    return policy


def _fit_strategy(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolError("{}: must be a string".format(field))
    strategy = value.strip().lower()
    if strategy not in FIT_STRATEGIES:
        raise ToolError(
            "{}: unknown fit strategy '{}' (expected one of: {})".format(
                field, value, ", ".join(FIT_STRATEGIES)
            )
        )
    return strategy


def _parse_filename(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("{}: must be a non-empty string".format(field))
    filename = value.strip()
    normalized = filename.replace("\\", "/")
    if Path(normalized).is_absolute():
        raise ToolError("{}: must be a relative path inside the source directory".format(field))
    if Path(normalized).suffix.lower() != ".wav":
        raise ToolError("{}: file '{}' is not a WAV file".format(field, filename))
    return normalized


def _resolve_source_path(
    source_root: Path,
    filename: str,
    field: str,
) -> Path:
    candidate = (source_root / Path(filename)).resolve()
    try:
        within_root = os.path.commonpath((str(source_root), str(candidate))) == str(source_root)
    except ValueError:
        within_root = False
    if not within_root:
        raise ToolError(
            "{}: file '{}' escapes the source directory".format(field, filename)
        )
    if not candidate.is_file():
        raise ToolError(
            "{}: file '{}' not found in source directory '{}'".format(
                field, filename, source_root
            )
        )
    return candidate


def load_project(
    source_directory: Path,
    events_path: Path,
    global_policy: Optional[str] = None,
) -> ProjectConfig:
    """Load and validate a project JSON file."""

    source_root = Path(source_directory).expanduser().resolve()
    if not source_root.exists():
        raise ToolError(
            "sources_dir: directory does not exist: '{}'".format(source_directory)
        )
    if not source_root.is_dir():
        raise ToolError(
            "sources_dir: path is not a directory: '{}'".format(source_directory)
        )

    events_path = Path(events_path)
    try:
        with events_path.open("r", encoding="utf-8-sig") as handle:
            document = json.load(handle)
    except FileNotFoundError as exc:
        raise ToolError(
            "events_file: file not found: '{}'".format(events_path)
        ) from exc
    except OSError as exc:
        raise ToolError(
            "events_file: cannot read '{}': {}".format(events_path, exc)
        ) from exc
    except json.JSONDecodeError as exc:
        raise ToolError(
            "events_file: invalid JSON in '{}' at line {} column {}: {}".format(
                events_path, exc.lineno, exc.colno, exc.msg
            )
        ) from exc

    if not isinstance(document, dict):
        raise ToolError("events_file: JSON root must be an object")

    _, listener_value = _first_present(
        document,
        ("listener", "listener_position"),
        "listener",
    )
    listener_x, listener_y = _parse_position(listener_value, "listener")

    _, min_gain_value = _optional_first_present(
        document,
        ("min_gain", "min_distance_gain"),
    )
    if min_gain_value is None:
        min_gain = DEFAULT_MIN_GAIN
    else:
        min_gain = _number(
            min_gain_value,
            "min_gain",
            minimum=0.0,
            maximum=1.0,
        )

    _, default_policy_value = _optional_first_present(
        document,
        ("conflict_policy", "conflict_strategy"),
    )
    default_policy = None
    if default_policy_value is not None:
        default_policy = _policy(default_policy_value, "conflict_policy")
    elif global_policy is not None:
        default_policy = _policy(global_policy, "conflict_policy")

    if "events" not in document:
        raise ToolError("events: missing required field")
    raw_events = document["events"]
    if not isinstance(raw_events, list):
        raise ToolError("events: must be an array")
    if not raw_events:
        raise ToolError("events: must contain at least one event")

    events: List[EventSpec] = []
    for index, raw_event in enumerate(raw_events):
        prefix = "events[{}]".format(index)
        if not isinstance(raw_event, dict):
            raise ToolError("{}: must be an object".format(prefix))

        _, filename_value = _first_present(
            raw_event,
            ("filename", "file", "audio_file"),
            prefix + ".filename",
        )
        filename = _parse_filename(filename_value, prefix + ".filename")
        source_path = _resolve_source_path(
            source_root,
            filename,
            prefix + ".filename",
        )

        _, position_value = _first_present(
            raw_event,
            ("position", "source", "source_position"),
            prefix + ".position",
        )
        source_x, source_y = _parse_position(position_value, prefix + ".position")

        _, trigger_value = _first_present(
            raw_event,
            ("trigger_time", "start_time"),
            prefix + ".trigger_time",
        )
        trigger_time = _number(
            trigger_value,
            prefix + ".trigger_time",
            minimum=0.0,
        )

        _, duration_value = _first_present(
            raw_event,
            ("duration", "event_duration"),
            prefix + ".duration",
        )
        duration = _number(
            duration_value,
            prefix + ".duration",
            minimum=0.0,
            strictly_greater=True,
        )

        _, priority_value = _first_present(
            raw_event,
            ("priority",),
            prefix + ".priority",
        )
        priority = _number(
            priority_value,
            prefix + ".priority",
            minimum=0.0,
        )

        policy_key, policy_value = _optional_first_present(
            raw_event,
            ("conflict_policy", "conflict_strategy"),
        )
        if policy_key is None:
            if default_policy is None:
                raise ToolError(
                    "{}.conflict_policy: missing required field".format(prefix)
                )
            conflict_policy = default_policy
        else:
            conflict_policy = _policy(
                policy_value,
                prefix + "." + policy_key,
            )

        _, fit_value = _optional_first_present(
            raw_event,
            ("fit_strategy", "audio_fit"),
        )
        if fit_value is None:
            fit_strategy = "stretch"
        else:
            fit_strategy = _fit_strategy(
                fit_value,
                prefix + ".fit_strategy",
            )

        events.append(
            EventSpec(
                index=index,
                filename=filename,
                source_path=source_path,
                source_x=source_x,
                source_y=source_y,
                trigger_time=trigger_time,
                duration=duration,
                priority=priority,
                conflict_policy=conflict_policy,
                fit_strategy=fit_strategy,
            )
        )

    return ProjectConfig(
        listener_x=listener_x,
        listener_y=listener_y,
        min_gain=min_gain,
        events=events,
    )
