"""Project JSON loading, schema validation and timeline consistency checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

REQUIRED_SOURCE_FORMAT = {"sample_rate": 44100, "channels": 1, "bit_depth": 16}


class ProjectError(Exception):
    """Fatal project-level error (exit code 2)."""


@dataclass
class Event:
    event_id: str
    track: str
    start: float
    duration: float
    source: str


@dataclass
class Project:
    name: str
    events: list
    tracks: dict
    chapters: list
    video: dict
    loudness: dict
    sources: dict
    timeline_duration: float
    warnings: list = field(default_factory=list)
    skipped_events: list = field(default_factory=list)
    validation: list = field(default_factory=list)

    def warn(self, kind: str, message: str, event_id=None):
        entry = {"type": kind, "message": message}
        if event_id is not None:
            entry["event_id"] = event_id
        self.warnings.append(entry)


def _require(mapping: dict, key: str, ctx: str):
    if key not in mapping:
        raise ProjectError(f"{ctx}: missing required key '{key}'")
    return mapping[key]


def load_project(path) -> Project:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"cannot read project JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProjectError("project root must be a JSON object")

    name = str(raw.get("project_name", "untitled"))
    tracks = _require(raw, "tracks", "project")
    if not isinstance(tracks, dict) or not tracks:
        raise ProjectError("'tracks' must be a non-empty object")
    for tname, tconf in tracks.items():
        if not isinstance(tconf, dict) or not isinstance(tconf.get("gain_db"), (int, float)):
            raise ProjectError(f"track '{tname}': missing numeric 'gain_db'")

    video = _require(raw, "video", "project")
    for key in ("width", "height", "fps"):
        if not isinstance(video.get(key), (int, float)) or video[key] <= 0:
            raise ProjectError(f"video: '{key}' must be a positive number")
    video.setdefault("color", "0x000000")

    loud = raw.get("loudness", {})
    loudness = {
        "target_lufs": float(loud.get("target_lufs", -16.0)),
        "true_peak_dbtp": float(loud.get("true_peak_dbtp", -1.0)),
    }

    sources = raw.get("sources", {})
    if not isinstance(sources, dict):
        raise ProjectError("'sources' must be an object")

    timeline = _require(raw, "timeline", "project")
    raw_events = _require(timeline, "events", "timeline")
    if not isinstance(raw_events, list):
        raise ProjectError("timeline.events must be a list")

    chapters = []
    for ch in raw.get("chapters", []):
        chapters.append({"title": str(ch.get("title", "chapter")),
                         "start": float(ch.get("start", 0.0))})
    chapters.sort(key=lambda c: c["start"])

    project = Project(
        name=name, events=[], tracks=tracks, chapters=chapters, video=video,
        loudness=loudness, sources=sources, timeline_duration=0.0,
    )

    # --- per-event validation: invalid events are skipped with warnings ---
    valid = []
    for idx, ev in enumerate(raw_events):
        ctx = f"event[{idx}]"
        try:
            event = Event(
                event_id=str(_require(ev, "event_id", ctx)),
                track=str(_require(ev, "track", ctx)),
                start=float(_require(ev, "start", ctx)),
                duration=float(_require(ev, "duration", ctx)),
                source=str(_require(ev, "source", ctx)),
            )
        except (ProjectError, TypeError, ValueError) as exc:
            project.warn("invalid_event", f"{ctx}: malformed event ({exc})")
            project.skipped_events.append({"index": idx, "reason": "malformed"})
            continue
        if event.duration <= 0:
            project.warn("negative_duration",
                         f"event '{event.event_id}': duration {event.duration} <= 0",
                         event.event_id)
            project.skipped_events.append({"event_id": event.event_id,
                                           "reason": "negative_duration"})
            continue
        if event.start < 0:
            project.warn("negative_start",
                         f"event '{event.event_id}': start {event.start} < 0",
                         event.event_id)
            project.skipped_events.append({"event_id": event.event_id,
                                           "reason": "negative_start"})
            continue
        if event.track not in tracks:
            project.warn("unknown_track",
                         f"event '{event.event_id}': unknown track '{event.track}'",
                         event.event_id)
            project.skipped_events.append({"event_id": event.event_id,
                                           "reason": "unknown_track"})
            continue
        src = sources.get(event.source)
        if src is None:
            project.warn("missing_source",
                         f"event '{event.event_id}': source '{event.source}' not declared",
                         event.event_id)
            project.skipped_events.append({"event_id": event.event_id,
                                           "reason": "missing_source"})
            continue
        fmt = {k: src.get(k) for k in REQUIRED_SOURCE_FORMAT}
        if fmt != REQUIRED_SOURCE_FORMAT:
            project.warn("format_mismatch",
                         f"event '{event.event_id}': source format {fmt} != "
                         f"{REQUIRED_SOURCE_FORMAT}", event.event_id)
            project.skipped_events.append({"event_id": event.event_id,
                                           "reason": "format_mismatch"})
            continue
        valid.append(event)

    # --- overlap detection per track (recorded as validation findings) ---
    valid.sort(key=lambda e: (e.track, e.start, e.event_id))
    per_track: dict[str, list] = {}
    for ev in valid:
        per_track.setdefault(ev.track, []).append(ev)
    for track, evs in per_track.items():
        for prev, cur in zip(evs, evs[1:]):
            prev_end = prev.start + prev.duration
            if cur.start < prev_end - 1e-9:
                finding = (f"overlap on track '{track}': '{prev.event_id}' ends at "
                           f"{prev_end:.3f}s but '{cur.event_id}' starts at {cur.start:.3f}s")
                project.validation.append({"type": "overlap", "track": track,
                                           "message": finding})
                project.warn("overlap", finding, cur.event_id)

    declared = float(timeline.get("duration", 0.0))
    last_end = max((e.start + e.duration for e in valid), default=0.0)
    project.timeline_duration = max(declared, last_end)
    if project.timeline_duration <= 0:
        raise ProjectError("timeline duration is zero (no usable events)")

    # --- chapter validation against total duration ---
    kept = []
    for ch in project.chapters:
        if ch["start"] < 0 or ch["start"] >= project.timeline_duration:
            project.warn("chapter_out_of_range",
                         f"chapter '{ch['title']}' start {ch['start']:.3f}s outside "
                         f"[0, {project.timeline_duration:.3f}s); dropped")
        else:
            kept.append(ch)
    project.chapters = kept
    project.events = valid
    return project
