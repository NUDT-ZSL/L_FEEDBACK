"""Project description parsing and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ValidationError

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mkv"}
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
MIN_TRACKS = 1
MAX_TRACKS = 8
MIN_OFFSET_MS = -5000
MAX_OFFSET_MS = 5000
MIN_MARKER_DRIFT = -10
MAX_MARKER_DRIFT = 10
MAX_MARKERS = 3


@dataclass(frozen=True)
class Track:
    index: int
    path: Path
    offset_ms: int
    envelope: tuple[float, ...]


@dataclass(frozen=True)
class Marker:
    frame: int
    drift_ms: int


@dataclass(frozen=True)
class Project:
    source_path: Path
    video_path: Path
    output_path: Path
    tracks: tuple[Track, ...]
    markers: tuple[Marker, ...]


def load_project(project_file: str | Path) -> Project:
    path = Path(project_file)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read project file: {exc}", "PROJECT_READ_ERROR")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc}", "PROJECT_PARSE_ERROR")
    if not isinstance(data, dict):
        raise ValidationError("project root must be a JSON object")
    return _build_project(data, path)


def _build_project(data: dict, project_file: Path) -> Project:
    base_dir = project_file.resolve().parent

    video_value = data.get("video_path")
    if not isinstance(video_value, str) or not video_value:
        raise ValidationError("'video_path' is required and must be a string")
    video_path = _resolve_path(video_value, base_dir)
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValidationError(
            f"unsupported video container '{video_path.suffix}'; "
            "only MP4/MKV are supported"
        )
    if not video_path.is_file():
        raise ValidationError(f"video file does not exist: {video_path}")

    output_value = data.get("output_path")
    if not isinstance(output_value, str) or not output_value:
        raise ValidationError("'output_path' is required and must be a string")
    output_path = _resolve_path(output_value, base_dir)
    if output_path.suffix.lower() != ".mp4":
        raise ValidationError("output_path must use the .mp4 extension")

    tracks_value = data.get("audio_tracks")
    if not isinstance(tracks_value, list):
        raise ValidationError("'audio_tracks' must be an array")
    if not (MIN_TRACKS <= len(tracks_value) <= MAX_TRACKS):
        raise ValidationError(
            f"audio track count must be between {MIN_TRACKS} and {MAX_TRACKS}"
        )

    tracks: list[Track] = []
    for index, item in enumerate(tracks_value):
        tracks.append(_parse_track(item, index, base_dir))

    markers = _parse_markers(data.get("frame_markers", []))

    return Project(
        source_path=project_file.resolve(),
        video_path=video_path,
        output_path=output_path,
        tracks=tuple(tracks),
        markers=tuple(markers),
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _parse_track(item: object, index: int, base_dir: Path) -> Track:
    label = f"audio_tracks[{index}]"
    if not isinstance(item, dict):
        raise ValidationError(f"{label} must be an object")

    path_value = item.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValidationError(f"{label}.path is required and must be a string")
    track_path = _resolve_path(path_value, base_dir)
    if track_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValidationError(
            f"{label}: unsupported audio format '{track_path.suffix}'"
        )
    if not track_path.is_file():
        raise ValidationError(f"{label}: audio file does not exist: {track_path}")

    offset = item.get("offset_ms")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValidationError(f"{label}.offset_ms must be an integer")
    if not (MIN_OFFSET_MS <= offset <= MAX_OFFSET_MS):
        raise ValidationError(
            f"{label}.offset_ms must be within [{MIN_OFFSET_MS}, {MAX_OFFSET_MS}]"
        )

    envelope = item.get("volume_envelope")
    if not isinstance(envelope, list) or len(envelope) < 2:
        raise ValidationError(
            f"{label}.volume_envelope must be an array of at least 2 floats"
        )
    normalized: list[float] = []
    for point in envelope:
        if isinstance(point, bool) or not isinstance(point, (int, float)):
            raise ValidationError(
                f"{label}.volume_envelope entries must be numbers"
            )
        value = float(point)
        if not (0.0 <= value <= 1.0):
            raise ValidationError(
                f"{label}.volume_envelope entries must be within [0.0, 1.0]"
            )
        normalized.append(value)

    return Track(
        index=index,
        path=track_path,
        offset_ms=offset,
        envelope=tuple(normalized),
    )


def _parse_markers(value: object) -> list[Marker]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError("'frame_markers' must be an array")
    if len(value) > MAX_MARKERS:
        raise ValidationError(f"at most {MAX_MARKERS} frame markers are allowed")
    markers: list[Marker] = []
    for index, item in enumerate(value):
        label = f"frame_markers[{index}]"
        if not isinstance(item, dict):
            raise ValidationError(f"{label} must be an object")
        frame = item.get("frame")
        drift = item.get("drift_ms")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValidationError(f"{label}.frame must be a non-negative integer")
        if isinstance(drift, bool) or not isinstance(drift, int):
            raise ValidationError(f"{label}.drift_ms must be an integer")
        if not (MIN_MARKER_DRIFT <= drift <= MAX_MARKER_DRIFT):
            raise ValidationError(
                f"{label}.drift_ms must be within "
                f"[{MIN_MARKER_DRIFT}, {MAX_MARKER_DRIFT}]"
            )
        markers.append(Marker(frame=frame, drift_ms=drift))
    return markers
