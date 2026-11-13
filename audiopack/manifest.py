"""Manifest data model, validation and hashing.

All validation problems are collected into a single :class:`IssueCollector`.
The pipeline refuses to proceed when structural errors exist, but the caller
still receives every problem instead of just the first one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config
from .errors import IssueCollector, ErrorCode
from .hashing import canonical_json_hash


def _is_int(value: Any) -> bool:
    # bool is a subclass of int but is not a valid timestamp/priority.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    time_ms: int
    priority: int
    voice_key: str | None
    langs: tuple[str, ...]
    max_delay_ms: int
    tone: str | None


@dataclass
class Manifest:
    events: list[Event]
    raw: dict[str, Any]
    manifest_hash: str
    issues: IssueCollector

    def start_sample(self, event: Event, config: Config) -> int:
        """Integer sample position derived directly from milliseconds.

        No cumulative floating-point timing is ever used: each start is the
        exact product ``time_ms * samples_per_ms``.
        """
        return event.time_ms * config.ms_frame

    @property
    def valid(self) -> bool:
        return not self.issues


def load_manifest(path: str, config: Config) -> Manifest:
    """Read and validate a manifest file, collecting every error."""
    import json

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_manifest(raw, config)


def parse_manifest(raw: Any, config: Config) -> Manifest:
    collector = IssueCollector()
    events: list[Event] = []

    if not isinstance(raw, dict):
        collector.add(
            ErrorCode.E_BAD_FIELD,
            "manifest root must be a JSON object",
            detail={"got_type": type(raw).__name__},
        )
        return Manifest(events=[], raw={"_invalid": True}, manifest_hash="",
                        issues=collector)

    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        collector.add(
            ErrorCode.E_BAD_FIELD,
            "'events' must be a list",
            detail={"got_type": type(raw_events).__name__},
        )
        return Manifest(events=[], raw=raw, manifest_hash="", issues=collector)

    seen_ids: set[str] = set()
    previous_time: int | None = None
    previous_id: str | None = None
    allowed = set(config.allowed_langs)

    for index, row in enumerate(raw_events):
        row_issues_before = len(collector.issues)
        if not isinstance(row, dict):
            collector.add(
                ErrorCode.E_BAD_FIELD,
                f"event at index {index} must be an object",
                detail={"index": index},
            )
            continue

        event_id = row.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            collector.add(
                ErrorCode.E_BAD_FIELD,
                f"event at index {index} has missing/empty 'id'",
                detail={"index": index},
            )
            event_id = None
        elif event_id in seen_ids:
            collector.add(
                ErrorCode.E_DUP_ID,
                f"duplicate event id '{event_id}'",
                event_id=event_id,
                detail={"index": index},
            )

        event_type = row.get("type")
        if not isinstance(event_type, str) or not event_type.strip():
            collector.add(
                ErrorCode.E_BAD_FIELD,
                "missing/empty 'type'",
                event_id=event_id,
                detail={"index": index},
            )

        time_ms = row.get("time_ms")
        if not _is_int(time_ms):
            collector.add(
                ErrorCode.E_BAD_FIELD,
                "'time_ms' must be an integer",
                event_id=event_id,
                detail={"index": index, "got": time_ms},
            )
        elif not (config.min_time_ms <= time_ms <= config.max_time_ms):
            collector.add(
                ErrorCode.E_BAD_TIME,
                "'time_ms' outside configured range",
                event_id=event_id,
                detail={
                    "index": index,
                    "time_ms": time_ms,
                    "min_time_ms": config.min_time_ms,
                    "max_time_ms": config.max_time_ms,
                },
            )
        elif previous_time is not None and time_ms < previous_time:
            collector.add(
                ErrorCode.E_BAD_TIME,
                "timestamps must be monotonically non-decreasing",
                event_id=event_id,
                detail={
                    "index": index,
                    "time_ms": time_ms,
                    "previous_time_ms": previous_time,
                    "previous_id": previous_id,
                },
            )

        priority = row.get("priority")
        if not _is_int(priority):
            collector.add(
                ErrorCode.E_BAD_FIELD,
                "'priority' must be an integer",
                event_id=event_id,
                detail={"index": index, "got": priority},
            )

        voice_key = row.get("voice_key")
        if voice_key is not None and (
            not isinstance(voice_key, str) or not voice_key.strip()
        ):
            collector.add(
                ErrorCode.E_BAD_FIELD,
                "'voice_key' must be a non-empty string when present",
                event_id=event_id,
                detail={"index": index},
            )
            voice_key = None

        langs = row.get("langs")
        if langs is None:
            langs_tuple: tuple[str, ...] = tuple(config.allowed_langs)
        elif not isinstance(langs, list) or not langs:
            collector.add(
                ErrorCode.E_BAD_LANG,
                "'langs' must be a non-empty list when present",
                event_id=event_id,
                detail={"index": index, "got": langs},
            )
            langs_tuple = ()
        else:
            bad = [lang for lang in langs if not isinstance(lang, str) or not lang]
            unknown = [lang for lang in langs if isinstance(lang, str) and lang not in allowed]
            if bad:
                collector.add(
                    ErrorCode.E_BAD_LANG,
                    "'langs' contains invalid entries",
                    event_id=event_id,
                    detail={"index": index, "bad": bad},
                )
            if unknown:
                collector.add(
                    ErrorCode.E_BAD_LANG,
                    "'langs' contains languages outside allowed list",
                    event_id=event_id,
                    detail={
                        "index": index,
                        "unknown": unknown,
                        "allowed": sorted(allowed),
                    },
                )
            langs_tuple = tuple(
                lang for lang in langs if isinstance(lang, str) and lang
            )

        max_delay = row.get("max_delay_ms", config.default_max_delay_ms)
        if not _is_int(max_delay) or max_delay < 0:
            collector.add(
                ErrorCode.E_BAD_FIELD,
                "'max_delay_ms' must be a non-negative integer",
                event_id=event_id,
                detail={"index": index, "got": max_delay},
            )
            max_delay = config.default_max_delay_ms

        tone = row.get("tone")
        if tone is not None and not isinstance(tone, str):
            collector.add(
                ErrorCode.E_BAD_FIELD,
                "'tone' must be a string when present",
                event_id=event_id,
                detail={"index": index},
            )
            tone = None

        # Advance monotonic tracking even if the row had issues, and only
        # materialize events whose required scalar fields are usable.
        if _is_int(time_ms):
            previous_time = time_ms
            previous_id = event_id

        row_valid = len(collector.issues) == row_issues_before
        if event_id is not None:
            seen_ids.add(event_id)
        if (
            row_valid
            and event_id is not None
            and isinstance(event_type, str)
            and _is_int(time_ms)
            and _is_int(priority)
            and langs_tuple
        ):
            events.append(
                Event(
                    event_id=event_id,
                    event_type=event_type,
                    time_ms=time_ms,
                    priority=priority,
                    voice_key=voice_key,
                    langs=langs_tuple,
                    max_delay_ms=max_delay,
                    tone=tone,
                )
            )

    # The hash covers the raw manifest as delivered, so any edit invalidates
    # snapshots even if the parsed event set happens to be equivalent.
    manifest_hash = canonical_json_hash(raw)
    return Manifest(
        events=events,
        raw=raw,
        manifest_hash=manifest_hash,
        issues=collector,
    )
