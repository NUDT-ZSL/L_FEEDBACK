"""Timeline JSON loading, schema validation and per-segment duration checks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import soundfile as sf

from . import (
    ERR_BAD_TIMING,
    ERR_DURATION_EXCEEDED,
    ERR_FILE_NOT_FOUND,
    ERR_FILE_UNREADABLE,
    ERR_BAD_SCHEMA,
)


class TimelineValidationError(Exception):
    def __init__(self, diagnostics: dict):
        super().__init__(diagnostics["error"]["message"])
        self.diagnostics = diagnostics


@dataclass
class Segment:
    index: int
    start: float
    end: float
    source_start: float
    source_end: float
    file: str
    language: str
    target_lufs: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Timeline:
    project: dict
    segments: list[Segment]
    music_path: str | None
    sfx_paths: list[str] = field(default_factory=list)
    base_dir: str = ""

    @property
    def duration(self) -> float:
        explicit = self.project.get("duration_seconds")
        if explicit is not None:
            return float(explicit)
        candidates = [seg.end for seg in self.segments]
        return max(candidates) if candidates else 0.0


def load_timeline(path: str) -> Timeline:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise TimelineValidationError(
            _diagnostic(ERR_FILE_NOT_FOUND, f"timeline file not found: {path}",
                        {"path": path}))
    except json.JSONDecodeError as exc:
        raise TimelineValidationError(
            _diagnostic(ERR_BAD_SCHEMA, f"invalid JSON: {exc.msg}",
                        {"line": exc.lineno, "column": exc.colno}))

    base_dir = os.path.dirname(os.path.abspath(path))
    return _parse(raw, base_dir)


def _parse(raw: dict, base_dir: str) -> Timeline:
    errors: list[dict] = []
    if not isinstance(raw, dict):
        raise TimelineValidationError(
            _diagnostic(ERR_BAD_SCHEMA, "timeline root must be a JSON object"))

    project = raw.get("project")
    if not isinstance(project, dict):
        errors.append(_segment_error(None, ERR_BAD_SCHEMA,
                                     "missing or invalid 'project' object"))
        project = {}

    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        errors.append(_segment_error(None, ERR_BAD_SCHEMA,
                                     "'segments' must be a non-empty array"))
        raw_segments = []

    segments: list[Segment] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            errors.append(_segment_error(index, ERR_BAD_SCHEMA, "segment must be an object"))
            continue
        parsed = _parse_segment(index, item, base_dir, errors)
        if parsed is not None:
            segments.append(parsed)

    music_path, sfx_paths = _parse_reference_tracks(raw, base_dir, errors)

    if errors:
        raise TimelineValidationError(
            _diagnostic(ERR_BAD_SCHEMA, "timeline validation failed",
                        {"issues": errors, "issue_count": len(errors)}))

    return Timeline(project=project, segments=segments,
                    music_path=music_path, sfx_paths=sfx_paths, base_dir=base_dir)


def _parse_segment(index: int, item: dict, base_dir: str, errors: list[dict]) -> Segment | None:
    required = ("start", "end", "source_range", "file", "language", "target_lufs")
    missing = [key for key in required if key not in item]
    if missing:
        errors.append(_segment_error(index, ERR_BAD_SCHEMA,
                                     f"missing fields: {', '.join(missing)}"))
        return None

    try:
        start = float(item["start"])
        end = float(item["end"])
        target_lufs = float(item["target_lufs"])
        source_range = item["source_range"]
        source_start, source_end = float(source_range[0]), float(source_range[1])
    except (TypeError, ValueError, IndexError):
        errors.append(_segment_error(index, ERR_BAD_SCHEMA,
                                     "start/end/target_lufs/source_range must be numeric"))
        return None

    if start < 0 or end < 0 or end < start:
        errors.append(_segment_error(index, ERR_BAD_TIMING,
                                     "start/end must be non-negative and end >= start",
                                     {"start": start, "end": end}))
        return None
    if source_start < 0 or source_end < 0 or source_end < source_start:
        errors.append(_segment_error(index, ERR_BAD_TIMING,
                                     "source_range must be non-negative with end >= start",
                                     {"source_range": [source_start, source_end]}))
        return None

    file_name = str(item["file"])
    file_path = file_name if os.path.isabs(file_name) else os.path.join(base_dir, file_name)
    if not os.path.isfile(file_path):
        errors.append(_segment_error(index, ERR_FILE_NOT_FOUND,
                                     f"audio file not found: {file_name}",
                                     {"path": file_path}))
        return None

    try:
        info = sf.info(file_path)
        file_duration = info.frames / float(info.samplerate)
    except Exception as exc:  # libsndfile cannot decode the file
        errors.append(_segment_error(index, ERR_FILE_UNREADABLE,
                                     f"audio file unreadable: {exc}",
                                     {"path": file_path}))
        return None

    if source_end > file_duration + 1e-9:
        errors.append(_segment_error(index, ERR_DURATION_EXCEEDED,
                                     "source_range end exceeds source audio duration",
                                     {"source_end": source_end,
                                      "audio_duration": file_duration}))
        return None

    return Segment(index=index, start=start, end=end,
                   source_start=source_start, source_end=source_end,
                   file=file_path,
                   language=str(item["language"]), target_lufs=target_lufs)


def _parse_reference_tracks(raw: dict, base_dir: str, errors: list[dict]):
    tracks = raw.get("reference_tracks", {})
    if not isinstance(tracks, dict):
        errors.append(_segment_error(None, ERR_BAD_SCHEMA,
                                     "'reference_tracks' must be an object"))
        return None, []

    music_path = _resolve_optional_track(tracks.get("music"), base_dir, errors, "music")
    sfx_paths: list[str] = []
    sfx_raw = tracks.get("sfx", [])
    if isinstance(sfx_raw, str):
        sfx_raw = [sfx_raw]
    if not isinstance(sfx_raw, list):
        errors.append(_segment_error(None, ERR_BAD_SCHEMA,
                                     "reference_tracks.sfx must be a string or array"))
    else:
        for name in sfx_raw:
            resolved = _resolve_optional_track(name, base_dir, errors, "sfx")
            if resolved is not None:
                sfx_paths.append(resolved)
    return music_path, sfx_paths


def _resolve_optional_track(name, base_dir: str, errors: list[dict], kind: str) -> str | None:
    if name in (None, ""):
        return None
    path = str(name)
    resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.isfile(resolved):
        errors.append(_segment_error(None, ERR_FILE_NOT_FOUND,
                                     f"{kind} track not found: {name}",
                                     {"path": resolved}))
        return None
    return resolved


def _segment_error(index, code: str, message: str, extra: dict | None = None) -> dict:
    issue = {"code": code, "timeline_index": index, "message": message}
    if extra:
        issue.update(extra)
    return issue


def _diagnostic(code: str, message: str, details: dict | None = None) -> dict:
    payload = {"status": "error", "error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return payload
