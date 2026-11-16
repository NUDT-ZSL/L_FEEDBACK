from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .errors import BusinessError


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_EPS = 1e-9


@dataclass(frozen=True)
class CameraEvent:
    id: str
    type: str
    at_s: float
    end_s: float | None


@dataclass(frozen=True)
class Subtitle:
    id: str
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class Dialogue:
    id: str
    shot_id: str
    start_s: float
    end_s: float
    subtitle_id: str
    text: str
    target_language: str
    subtitle_start_s: float
    subtitle_end_s: float
    source_duration_s: float | None = None

    @property
    def window_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Shot:
    id: str
    start_s: float
    end_s: float
    events: tuple[CameraEvent, ...]
    subtitles: tuple[Subtitle, ...]
    dialogues: tuple[Dialogue, ...]


@dataclass(frozen=True)
class Timeline:
    schema_version: str
    project_id: str
    target_language: str
    duration_s: float
    shots: tuple[Shot, ...]
    source_path: Path
    source_sha256: str

    @property
    def dialogues(self) -> tuple[Dialogue, ...]:
        return tuple(d for shot in self.shots for d in shot.dialogues)

    def shot_for(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError(shot_id)


def _fail(
    code: str,
    message: str,
    *,
    entry_id: str | None = None,
    path: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raise BusinessError(
        code=code,
        message=message,
        entry_id=entry_id,
        path=path,
        details=details or {},
    )


def _object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("E_TIMELINE_SCHEMA", "Expected a JSON object.", path=path)
    return value


def _array(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("E_TIMELINE_SCHEMA", "Expected a JSON array.", path=path)
    return value


def _check_keys(
    obj: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    path: str,
    entry_id: str | None = None,
) -> None:
    missing = sorted(required - obj.keys())
    if missing:
        _fail(
            "E_TIMELINE_SCHEMA",
            f"Missing required field(s): {', '.join(missing)}.",
            entry_id=entry_id,
            path=path,
        )
    unknown = sorted(obj.keys() - required - optional)
    if unknown:
        _fail(
            "E_TIMELINE_SCHEMA",
            f"Unknown field(s): {', '.join(unknown)}.",
            entry_id=entry_id,
            path=f"{path}.{unknown[0]}",
        )


def _text(value: Any, *, path: str, entry_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(
            "E_TIMELINE_SCHEMA",
            "Expected a non-empty string.",
            entry_id=entry_id,
            path=path,
        )
    return value


def _identifier(
    value: Any,
    *,
    path: str,
    kind: str,
    entry_id: str | None = None,
) -> str:
    result = _text(value, path=path, entry_id=entry_id)
    if not _ID_RE.fullmatch(result):
        _fail(
            "E_TIMELINE_SCHEMA",
            f"{kind} must match [A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}.",
            entry_id=entry_id,
            path=path,
        )
    if result.endswith(".") or result.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        _fail(
            "E_TIMELINE_SCHEMA",
            f"{kind} is not a safe Windows filename component.",
            entry_id=entry_id,
            path=path,
        )
    return result


def _language(
    value: Any,
    *,
    path: str,
    entry_id: str | None = None,
) -> str:
    result = _text(value, path=path, entry_id=entry_id)
    if not _LANGUAGE_RE.fullmatch(result):
        _fail(
            "E_TIMELINE_SCHEMA",
            "target_language must match [A-Za-z0-9][A-Za-z0-9_-]{0,31}.",
            entry_id=entry_id,
            path=path,
        )
    if result.upper() in _WINDOWS_RESERVED:
        _fail(
            "E_TIMELINE_SCHEMA",
            "target_language is not a safe Windows directory name.",
            entry_id=entry_id,
            path=path,
        )
    return result


def _number(
    value: Any,
    *,
    path: str,
    entry_id: str | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(
            "E_TIMELINE_SCHEMA",
            "Expected a finite number.",
            entry_id=entry_id,
            path=path,
        )
    result = float(value)
    if not math.isfinite(result):
        _fail(
            "E_TIMELINE_SCHEMA",
            "Expected a finite number.",
            entry_id=entry_id,
            path=path,
        )
    return result


def _ordered_ranges(
    items: list[tuple[str, float, float]],
    *,
    code: str,
    kind: str,
    path_prefix: str,
) -> None:
    previous: tuple[str, float, float] | None = None
    for index, item in enumerate(items):
        entry_id, start, end = item
        if previous is not None:
            prev_id, prev_start, prev_end = previous
            if start + _EPS < prev_start:
                _fail(
                    code,
                    f"{kind} entries must be sorted by increasing start time.",
                    entry_id=entry_id,
                    path=f"{path_prefix}[{index}].start_s",
                    details={"previous_id": prev_id},
                )
            if start + _EPS < prev_end:
                _fail(
                    code,
                    f"{kind} entries must not overlap.",
                    entry_id=entry_id,
                    path=f"{path_prefix}[{index}].start_s",
                    details={"previous_id": prev_id, "previous_end_s": prev_end},
                )
        previous = item


def _parse_event(raw: Any, *, path: str, shot_start: float, shot_end: float) -> CameraEvent:
    obj = _object(raw, path=path)
    entry_hint = obj.get("id") if isinstance(obj.get("id"), str) else None
    _check_keys(
        obj,
        required={"id", "type", "at_s"},
        optional={"end_s"},
        path=path,
        entry_id=entry_hint,
    )
    event_id = _identifier(obj["id"], path=f"{path}.id", kind="Event id", entry_id=entry_hint)
    event_type = _text(obj["type"], path=f"{path}.type", entry_id=event_id)
    at_s = _number(obj["at_s"], path=f"{path}.at_s", entry_id=event_id)
    if at_s < shot_start - _EPS or at_s > shot_end + _EPS:
        _fail(
            "E_EVENT_INVALID",
            "Camera event time must fall inside its shot.",
            entry_id=event_id,
            path=f"{path}.at_s",
            details={"shot_start_s": shot_start, "shot_end_s": shot_end},
        )
    end_s: float | None = None
    if "end_s" in obj:
        end_s = _number(obj["end_s"], path=f"{path}.end_s", entry_id=event_id)
        if end_s < at_s - _EPS or end_s > shot_end + _EPS:
            _fail(
                "E_EVENT_INVALID",
                "Camera event end must be at or after its start and inside its shot.",
                entry_id=event_id,
                path=f"{path}.end_s",
            )
    return CameraEvent(id=event_id, type=event_type, at_s=at_s, end_s=end_s)


def _parse_subtitle(
    raw: Any,
    *,
    path: str,
    shot_id: str,
    shot_start: float,
    shot_end: float,
) -> Subtitle:
    obj = _object(raw, path=path)
    entry_hint = obj.get("id") if isinstance(obj.get("id"), str) else None
    _check_keys(
        obj,
        required={"id", "start_s", "end_s", "text"},
        optional=set(),
        path=path,
        entry_id=entry_hint,
    )
    subtitle_id = _identifier(
        obj["id"], path=f"{path}.id", kind="Subtitle id", entry_id=entry_hint
    )
    start_s = _number(obj["start_s"], path=f"{path}.start_s", entry_id=subtitle_id)
    end_s = _number(obj["end_s"], path=f"{path}.end_s", entry_id=subtitle_id)
    text = _text(obj["text"], path=f"{path}.text", entry_id=subtitle_id)
    if end_s <= start_s + _EPS:
        _fail(
            "E_SUBTITLE_INVALID",
            "Subtitle end_s must be greater than start_s.",
            entry_id=subtitle_id,
            path=f"{path}.end_s",
        )
    if start_s < shot_start - _EPS or end_s > shot_end + _EPS:
        _fail(
            "E_SUBTITLE_INVALID",
            "Subtitle interval must be fully contained in its shot.",
            entry_id=subtitle_id,
            path=path,
            details={
                "shot_id": shot_id,
                "shot_start_s": shot_start,
                "shot_end_s": shot_end,
            },
        )
    return Subtitle(id=subtitle_id, start_s=start_s, end_s=end_s, text=text)


def _parse_dialogue(
    raw: Any,
    *,
    path: str,
    shot_id: str,
    shot_start: float,
    shot_end: float,
    default_language: str,
) -> Dialogue:
    obj = _object(raw, path=path)
    entry_hint = obj.get("id") if isinstance(obj.get("id"), str) else None
    _check_keys(
        obj,
        required={"id", "start_s", "end_s", "subtitle_id", "text"},
        optional={"target_language", "source_duration_s"},
        path=path,
        entry_id=entry_hint,
    )
    dialogue_id = _identifier(
        obj["id"], path=f"{path}.id", kind="Dialogue id", entry_id=entry_hint
    )
    start_s = _number(obj["start_s"], path=f"{path}.start_s", entry_id=dialogue_id)
    end_s = _number(obj["end_s"], path=f"{path}.end_s", entry_id=dialogue_id)
    subtitle_id = _identifier(
        obj["subtitle_id"],
        path=f"{path}.subtitle_id",
        kind="Subtitle reference",
        entry_id=dialogue_id,
    )
    text = _text(obj["text"], path=f"{path}.text", entry_id=dialogue_id)
    language = default_language
    if "target_language" in obj:
        language = _language(
            obj["target_language"], path=f"{path}.target_language", entry_id=dialogue_id
        )
        if language != default_language:
            _fail(
                "E_DIALOGUE_INVALID",
                "Dialogue target_language must match the timeline target_language.",
                entry_id=dialogue_id,
                path=f"{path}.target_language",
                details={"expected": default_language, "actual": language},
            )
    if end_s <= start_s + _EPS:
        _fail(
            "E_DIALOGUE_INVALID",
            "Dialogue end_s must be greater than start_s.",
            entry_id=dialogue_id,
            path=f"{path}.end_s",
        )
    if start_s < shot_start - _EPS or end_s > shot_end + _EPS:
        _fail(
            "E_DIALOGUE_INVALID",
            "Dialogue interval must be fully contained in its shot.",
            entry_id=dialogue_id,
            path=path,
            details={
                "shot_id": shot_id,
                "shot_start_s": shot_start,
                "shot_end_s": shot_end,
            },
        )
    source_duration_s: float | None = None
    if "source_duration_s" in obj:
        source_duration_s = _number(
            obj["source_duration_s"],
            path=f"{path}.source_duration_s",
            entry_id=dialogue_id,
        )
        if source_duration_s <= 0:
            _fail(
                "E_DIALOGUE_INVALID",
                "source_duration_s must be greater than zero.",
                entry_id=dialogue_id,
                path=f"{path}.source_duration_s",
            )
    return Dialogue(
        id=dialogue_id,
        shot_id=shot_id,
        start_s=start_s,
        end_s=end_s,
        subtitle_id=subtitle_id,
        text=text,
        target_language=language,
        subtitle_start_s=0.0,
        subtitle_end_s=0.0,
        source_duration_s=source_duration_s,
    )


def _validate_timeline(obj: dict[str, Any], *, path: Path, sha256: str) -> Timeline:
    _check_keys(
        obj,
        required={"schema_version", "project_id", "target_language", "duration_s", "shots"},
        optional=set(),
        path="$",
    )
    schema_version = _text(obj["schema_version"], path="$.schema_version")
    if schema_version != SCHEMA_VERSION:
        _fail(
            "E_TIMELINE_SCHEMA",
            f"Unsupported schema_version; expected {SCHEMA_VERSION}.",
            path="$.schema_version",
        )
    project_id = _identifier(
        obj["project_id"], path="$.project_id", kind="project_id"
    )
    target_language = _language(obj["target_language"], path="$.target_language")
    duration_s = _number(obj["duration_s"], path="$.duration_s")
    if duration_s <= 0:
        _fail("E_TIMELINE_RANGE", "duration_s must be greater than zero.", path="$.duration_s")

    raw_shots = _array(obj["shots"], path="$.shots")
    if not raw_shots:
        _fail("E_TIMELINE_SCHEMA", "At least one shot is required.", path="$.shots")

    shots: list[Shot] = []
    shot_ids: set[str] = set()
    subtitle_ids: set[str] = set()
    dialogue_ids: set[str] = set()
    event_ids: set[str] = set()
    shot_ranges: list[tuple[str, float, float]] = []

    for shot_index, raw_shot in enumerate(raw_shots):
        shot_path = f"$.shots[{shot_index}]"
        shot_obj = _object(raw_shot, path=shot_path)
        entry_hint = shot_obj.get("id") if isinstance(shot_obj.get("id"), str) else None
        _check_keys(
            shot_obj,
            required={"id", "start_s", "end_s", "events", "subtitles", "dialogues"},
            optional=set(),
            path=shot_path,
            entry_id=entry_hint,
        )
        shot_id = _identifier(
            shot_obj["id"], path=f"{shot_path}.id", kind="Shot id", entry_id=entry_hint
        )
        if shot_id in shot_ids:
            _fail(
                "E_SHOT_INVALID",
                "Shot ids must be globally unique.",
                entry_id=shot_id,
                path=f"{shot_path}.id",
            )
        shot_ids.add(shot_id)
        shot_start = _number(
            shot_obj["start_s"], path=f"{shot_path}.start_s", entry_id=shot_id
        )
        shot_end = _number(shot_obj["end_s"], path=f"{shot_path}.end_s", entry_id=shot_id)
        if shot_end <= shot_start + _EPS:
            _fail(
                "E_SHOT_INVALID",
                "Shot end_s must be greater than start_s.",
                entry_id=shot_id,
                path=f"{shot_path}.end_s",
            )
        if shot_start < -_EPS or shot_end > duration_s + _EPS:
            _fail(
                "E_SHOT_INVALID",
                "Shot interval must be inside the timeline duration.",
                entry_id=shot_id,
                path=shot_path,
                details={"timeline_duration_s": duration_s},
            )
        shot_ranges.append((shot_id, shot_start, shot_end))

        raw_events = _array(shot_obj["events"], path=f"{shot_path}.events")
        events: list[CameraEvent] = []
        for event_index, raw_event in enumerate(raw_events):
            event_path = f"{shot_path}.events[{event_index}]"
            event = _parse_event(
                raw_event, path=event_path, shot_start=shot_start, shot_end=shot_end
            )
            if event.id in event_ids:
                _fail(
                    "E_EVENT_INVALID",
                    "Event ids must be globally unique.",
                    entry_id=event.id,
                    path=f"{event_path}.id",
                )
            event_ids.add(event.id)
            events.append(event)

        raw_subtitles = _array(shot_obj["subtitles"], path=f"{shot_path}.subtitles")
        subtitles: list[Subtitle] = []
        for subtitle_index, raw_subtitle in enumerate(raw_subtitles):
            subtitle_path = f"{shot_path}.subtitles[{subtitle_index}]"
            subtitle = _parse_subtitle(
                raw_subtitle,
                path=subtitle_path,
                shot_id=shot_id,
                shot_start=shot_start,
                shot_end=shot_end,
            )
            if subtitle.id in subtitle_ids:
                _fail(
                    "E_SUBTITLE_INVALID",
                    "Subtitle ids must be globally unique.",
                    entry_id=subtitle.id,
                    path=f"{subtitle_path}.id",
                )
            subtitle_ids.add(subtitle.id)
            subtitles.append(subtitle)
        _ordered_ranges(
            [(s.id, s.start_s, s.end_s) for s in subtitles],
            code="E_SUBTITLE_INVALID",
            kind="Subtitle",
            path_prefix=f"{shot_path}.subtitles",
        )
        subtitles_by_id = {s.id: s for s in subtitles}

        raw_dialogues = _array(shot_obj["dialogues"], path=f"{shot_path}.dialogues")
        dialogues: list[Dialogue] = []
        for dialogue_index, raw_dialogue in enumerate(raw_dialogues):
            dialogue_path = f"{shot_path}.dialogues[{dialogue_index}]"
            dialogue = _parse_dialogue(
                raw_dialogue,
                path=dialogue_path,
                shot_id=shot_id,
                shot_start=shot_start,
                shot_end=shot_end,
                default_language=target_language,
            )
            if dialogue.id in dialogue_ids:
                _fail(
                    "E_DIALOGUE_INVALID",
                    "Dialogue ids must be globally unique.",
                    entry_id=dialogue.id,
                    path=f"{dialogue_path}.id",
                )
            dialogue_ids.add(dialogue.id)
            subtitle = subtitles_by_id.get(dialogue.subtitle_id)
            if subtitle is None:
                _fail(
                    "E_DIALOGUE_INVALID",
                    "Dialogue subtitle_id does not refer to a subtitle in the same shot.",
                    entry_id=dialogue.id,
                    path=f"{dialogue_path}.subtitle_id",
                    details={"subtitle_id": dialogue.subtitle_id, "shot_id": shot_id},
                )
            if (
                dialogue.start_s < subtitle.start_s - _EPS
                or dialogue.end_s > subtitle.end_s + _EPS
            ):
                _fail(
                    "E_DIALOGUE_INVALID",
                    "Dialogue interval must be contained in its subtitle interval.",
                    entry_id=dialogue.id,
                    path=dialogue_path,
                    details={
                        "subtitle_id": subtitle.id,
                        "subtitle_start_s": subtitle.start_s,
                        "subtitle_end_s": subtitle.end_s,
                    },
                )
            dialogue = replace(
                dialogue,
                subtitle_start_s=subtitle.start_s,
                subtitle_end_s=subtitle.end_s,
            )
            dialogues.append(dialogue)
        _ordered_ranges(
            [(d.id, d.start_s, d.end_s) for d in dialogues],
            code="E_DIALOGUE_INVALID",
            kind="Dialogue",
            path_prefix=f"{shot_path}.dialogues",
        )
        shots.append(
            Shot(
                id=shot_id,
                start_s=shot_start,
                end_s=shot_end,
                events=tuple(events),
                subtitles=tuple(subtitles),
                dialogues=tuple(dialogues),
            )
        )

    _ordered_ranges(shot_ranges, code="E_SHOT_INVALID", kind="Shot", path_prefix="$.shots")
    all_dialogues = [d for shot in shots for d in shot.dialogues]
    _ordered_ranges(
        [(d.id, d.start_s, d.end_s) for d in all_dialogues],
        code="E_DIALOGUE_INVALID",
        kind="Dialogue",
        path_prefix="$.shots[*].dialogues",
    )
    return Timeline(
        schema_version=schema_version,
        project_id=project_id,
        target_language=target_language,
        duration_s=duration_s,
        shots=tuple(shots),
        source_path=path,
        source_sha256=sha256,
    )


def load_timeline(path: str | Path) -> Timeline:
    timeline_path = Path(path)
    try:
        raw = timeline_path.read_bytes()
    except OSError as exc:
        raise BusinessError(
            code="E_TIMELINE_READ",
            message=f"Unable to read timeline file: {exc}",
            path=str(timeline_path),
            details={"reason": type(exc).__name__},
        ) from exc
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
        obj = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BusinessError(
            code="E_TIMELINE_JSON",
            message=f"Timeline is not valid UTF-8 JSON: {exc}",
            path=str(timeline_path),
        ) from exc
    obj = _object(obj, path="$")
    return _validate_timeline(obj, path=timeline_path, sha256=sha256)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

