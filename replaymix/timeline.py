"""Event-table parsing and the shared 48 kHz timeline conversion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (
    E_TRACK_TIMELINE_MISMATCH,
    EventTableError,
    TrackError,
    error,
)

INTERNAL_SAMPLE_RATE = 48_000
FRAMES_PER_MS = INTERNAL_SAMPLE_RATE / 1000.0
MIN_SUPPORTED_RATE = 8_000
MAX_SUPPORTED_RATE = 48_000


def ms_to_frames(milliseconds: float) -> int:
    """Convert a session/segment millisecond boundary using round(ms * 48)."""

    return int(round(float(milliseconds) * FRAMES_PER_MS))


def frames_to_ms(frames: int) -> float:
    return float(frames) / FRAMES_PER_MS


def supported_sample_rate(rate: int | None) -> bool:
    return (
        isinstance(rate, int)
        and MIN_SUPPORTED_RATE <= rate <= MAX_SUPPORTED_RATE
    )


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise EventTableError(f"{field_name} must be numeric") from exc
        else:
            raise EventTableError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EventTableError(f"{field_name} must be finite")
    return number


def _integer(value: Any, field_name: str) -> int:
    number = _number(value, field_name)
    if number != int(number):
        raise EventTableError(f"{field_name} must be an integer")
    return int(number)


def _first(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _parse_intervals(value: Any, field_name: str) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise EventTableError(f"{field_name} must be a list of intervals")
    intervals: list[tuple[float, float]] = []
    for index, item in enumerate(value):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start_raw, end_raw = item
        elif isinstance(item, dict):
            start_raw = _first(item, "start_ms", "start", "from_ms", "begin_ms")
            end_raw = _first(item, "end_ms", "end", "to_ms", "finish_ms")
        else:
            raise EventTableError(
                f"{field_name}[{index}] must be [start_ms, end_ms] or an object"
            )
        start = _number(start_raw, f"{field_name}[{index}].start_ms")
        end = _number(end_raw, f"{field_name}[{index}].end_ms")
        if end < start:
            raise EventTableError(
                f"{field_name}[{index}] has end_ms before start_ms"
            )
        intervals.append((start, end))
    return intervals


def _track_sort_key(track_id: Any) -> tuple[int, Any]:
    if isinstance(track_id, int) and not isinstance(track_id, bool):
        return (0, track_id)
    text = str(track_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


@dataclass
class TrackSpec:
    track_id: Any
    index: int
    player: str | None
    file_path: Path | None
    file_display: str
    sample_rate_hz: int | None
    channels: int | None
    declared_frames: int | None
    write_start_ms: float
    write_end_ms: float
    silence: list[tuple[float, float]]
    gaps: list[tuple[float, float]]
    markers: list[dict[str, Any]]
    raw_events: dict[str, Any]
    spec_errors: list[TrackError] = field(default_factory=list)

    @property
    def sort_key(self) -> tuple[int, Any]:
        return _track_sort_key(self.track_id)

    def canonical(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "player": self.player,
            "file": self.file_display,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "frames": self.declared_frames,
            "write_start_ms": self.write_start_ms,
            "write_end_ms": self.write_end_ms,
            "silence": [list(item) for item in self.silence],
            "gaps": [list(item) for item in self.gaps],
            "markers": self.markers,
        }


@dataclass
class EventTable:
    path: Path
    session_id: str
    session_ms: float
    tracks: list[TrackSpec]
    events: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def total_frames(self) -> int:
        return ms_to_frames(self.session_ms)

    def canonical(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_ms": self.session_ms,
            "tracks": [track.canonical() for track in self.tracks],
            "events": self.events,
        }


def _parse_track(item: Any, index: int, table_dir: Path) -> TrackSpec:
    errors: list[TrackError] = []
    if not isinstance(item, dict):
        item = {}
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, "track entry is not an object"))

    track_id = _first(item, "track_id", "id", "track")
    if track_id is None:
        track_id = f"unnamed-{index}"
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, "track_id is missing"))

    file_raw = _first(item, "file", "path", "filename")
    file_display = str(file_raw) if file_raw is not None else ""
    if file_raw is None or file_display == "":
        file_path = None
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, "track file is missing"))
    else:
        file_path = (table_dir / file_display).resolve()

    rate_raw = _first(item, "sample_rate_hz", "sample_rate", "rate")
    try:
        sample_rate_hz = _integer(rate_raw, "sample_rate_hz")
    except EventTableError as exc:
        sample_rate_hz = None
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))

    channels_raw = _first(item, "channels", "channel_count", "num_channels", default=1)
    try:
        channels = _integer(channels_raw, "channels")
    except EventTableError as exc:
        channels = None
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))

    frames_raw = _first(item, "frames", "num_frames", "total_frames")
    try:
        declared_frames = (
            _integer(frames_raw, "frames") if frames_raw is not None else None
        )
    except EventTableError as exc:
        declared_frames = None
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))

    try:
        write_start_ms = _number(
            _first(item, "write_start_ms", "start_ms", "record_start_ms", default=0.0),
            "write_start_ms",
        )
    except EventTableError as exc:
        write_start_ms = 0.0
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))

    try:
        write_end_ms = _number(
            _first(
                item,
                "write_end_ms",
                "end_ms",
                "record_end_ms",
                "duration_ms",
            ),
            "write_end_ms",
        )
    except EventTableError as exc:
        write_end_ms = write_start_ms
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))
    if "duration_ms" in item and not any(
        key in item for key in ("write_end_ms", "end_ms", "record_end_ms")
    ):
        write_end_ms = write_start_ms + _number(item["duration_ms"], "duration_ms")

    try:
        silence = _parse_intervals(
            _first(item, "silence_ms", "silence", "silence_intervals"),
            "silence_ms",
        )
    except EventTableError as exc:
        silence = []
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))

    try:
        gaps = _parse_intervals(
            _first(item, "gaps_ms", "gaps", "gap_intervals", "dropouts"),
            "gaps_ms",
        )
    except EventTableError as exc:
        gaps = []
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, str(exc)))

    markers_raw = _first(item, "markers", "events", default=[])
    if not isinstance(markers_raw, list):
        markers_raw = []
        errors.append(error(E_TRACK_TIMELINE_MISMATCH, "markers must be a list"))
    markers = [dict(marker) for marker in markers_raw if isinstance(marker, dict)]

    return TrackSpec(
        track_id=track_id,
        index=index,
        player=_first(item, "player", "name"),
        file_path=file_path,
        file_display=file_display,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        declared_frames=declared_frames,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        silence=silence,
        gaps=gaps,
        markers=markers,
        raw_events=dict(item),
        spec_errors=errors,
    )


def load_event_table(path: str | Path) -> EventTable:
    table_path = Path(path).resolve()
    try:
        raw = json.loads(table_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EventTableError(f"cannot read event table: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EventTableError(f"event table is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise EventTableError("event table root must be an object")

    session_ms_raw = _first(
        raw, "session_ms", "session_duration_ms", "duration_ms", "match_ms"
    )
    if session_ms_raw is None:
        raise EventTableError("session_ms is required")
    session_ms = _number(session_ms_raw, "session_ms")
    if session_ms <= 0:
        raise EventTableError("session_ms must be positive")

    tracks_raw = raw.get("tracks")
    if not isinstance(tracks_raw, list):
        raise EventTableError("tracks must be a list")

    events_raw = raw.get("events", [])
    if not isinstance(events_raw, list):
        raise EventTableError("events must be a list")
    events = [dict(event) for event in events_raw if isinstance(event, dict)]

    table_dir = table_path.parent
    tracks = [
        _parse_track(item, index, table_dir) for index, item in enumerate(tracks_raw)
    ]
    tracks.sort(key=lambda track: track.sort_key)

    session_id = str(raw.get("session_id", table_path.stem))
    return EventTable(
        path=table_path,
        session_id=session_id,
        session_ms=session_ms,
        tracks=tracks,
        events=events,
        raw=raw,
    )
