"""Project description parsing and schema validation.

The project is a deterministic JSON document describing the render session:

{
  "project_id": "demo",
  "sample_rate": 48000,
  "channels": 2,
  "duration_ms": 1000,
  "target_lufs": -23.0,
  "output": {
    "path": "out/mix.wav",
    "format": "wav",
    "bit_depth": 16
  },
  "compressor": {
    "threshold_db": -18.0,
    "ratio": 4.0,
    "knee_db": 6.0,
    "attack_ms": 5.0,
    "release_ms": 50.0,
    "lookahead_ms": 0.0
  },
  "tracks": [
    {
      "track_id": "t1",
      "source": "samples/t1.wav",
      "gain_db": -1.0,
      "muted": false,
      "events": [
        {"event_id": "e1", "start_ms": 0, "duration_ms": 500,
         "source_offset_ms": 0, "gain_db": 0.0}
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CineMixerError

SUPPORTED_CHANNELS = frozenset({1, 2})
SUPPORTED_FORMATS = frozenset({"wav"})
SUPPORTED_BIT_DEPTHS = frozenset({16, 24, 32})
MIN_SAMPLE_RATE = 1000
MAX_SAMPLE_RATE = 192000


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    start_ms: float
    duration_ms: float
    source_offset_ms: float = 0.0
    gain_db: float = 0.0


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    source: str
    gain_db: float
    muted: bool
    events: tuple[EventSpec, ...]


@dataclass(frozen=True)
class CompressorSpec:
    threshold_db: float = -18.0
    ratio: float = 4.0
    knee_db: float = 6.0
    attack_ms: float = 5.0
    release_ms: float = 50.0
    lookahead_ms: float = 0.0


@dataclass(frozen=True)
class OutputSpec:
    path: str
    format: str = "wav"
    bit_depth: int = 16


@dataclass(frozen=True)
class Project:
    project_id: str
    sample_rate: int
    channels: int
    duration_ms: float
    target_lufs: float
    output: OutputSpec
    compressor: CompressorSpec
    tracks: tuple[TrackSpec, ...]
    source_path: Path = field(default=Path("project.json"))


def _schema_error(message: str) -> CineMixerError:
    return CineMixerError("E_PROJECT_SCHEMA", message, 2)


def _require_type(obj: Any, types: tuple[type, ...], path: str) -> Any:
    if not isinstance(obj, types):
        name = " or ".join(t.__name__ for t in types)
        raise _schema_error(f"field {path} must be of type {name}")
    return obj


def _finite_number(obj: Any, path: str) -> float:
    _require_type(obj, (int, float), path)
    value = float(obj)
    if not math.isfinite(value):
        raise _schema_error(f"field {path} must be a finite number")
    return value


def _parse_compressor(raw: Any) -> CompressorSpec:
    if raw is None:
        return CompressorSpec()
    _require_type(raw, (dict,), "compressor")
    spec = CompressorSpec(
        threshold_db=_finite_number(raw.get("threshold_db", -18.0), "compressor.threshold_db"),
        ratio=_finite_number(raw.get("ratio", 4.0), "compressor.ratio"),
        knee_db=_finite_number(raw.get("knee_db", 6.0), "compressor.knee_db"),
        attack_ms=_finite_number(raw.get("attack_ms", 5.0), "compressor.attack_ms"),
        release_ms=_finite_number(raw.get("release_ms", 50.0), "compressor.release_ms"),
        lookahead_ms=_finite_number(raw.get("lookahead_ms", 0.0), "compressor.lookahead_ms"),
    )
    if not (-80.0 <= spec.threshold_db <= 0.0):
        raise _schema_error("compressor.threshold_db must be within [-80, 0] dB")
    if spec.ratio < 1.0 or spec.ratio > 100.0:
        raise _schema_error("compressor.ratio must be within [1, 100]")
    if spec.knee_db < 0.0 or spec.knee_db > 24.0:
        raise _schema_error("compressor.knee_db must be within [0, 24] dB")
    if spec.attack_ms <= 0.0 or spec.attack_ms > 500.0:
        raise _schema_error("compressor.attack_ms must be within (0, 500] ms")
    if spec.release_ms <= 0.0 or spec.release_ms > 5000.0:
        raise _schema_error("compressor.release_ms must be within (0, 5000] ms")
    if spec.lookahead_ms < 0.0 or spec.lookahead_ms > 5.0:
        raise _schema_error("compressor.lookahead_ms must be within [0, 5] ms (max latency)")
    return spec


def _parse_event(raw: Any, track_id: str, index: int) -> EventSpec:
    path = f"tracks[{track_id}].events[{index}]"
    _require_type(raw, (dict,), path)
    event_id = raw.get("event_id")
    _require_type(event_id, (str,), f"{path}.event_id")
    if not event_id:
        raise _schema_error(f"{path}.event_id must be non-empty")
    start_ms = _finite_number(raw.get("start_ms"), f"{path}.start_ms")
    duration_ms = _finite_number(raw.get("duration_ms"), f"{path}.duration_ms")
    if duration_ms < 0.0:
        raise _schema_error(f"{path}.duration_ms must be >= 0")
    source_offset_ms = _finite_number(raw.get("source_offset_ms", 0.0), f"{path}.source_offset_ms")
    if source_offset_ms < 0.0:
        raise _schema_error(f"{path}.source_offset_ms must be >= 0")
    gain_db = _finite_number(raw.get("gain_db", 0.0), f"{path}.gain_db")
    if not (-120.0 <= gain_db <= 60.0):
        raise _schema_error(f"{path}.gain_db must be within [-120, 60] dB")
    return EventSpec(event_id, start_ms, duration_ms, source_offset_ms, gain_db)


def _parse_track(raw: Any, index: int) -> TrackSpec:
    path = f"tracks[{index}]"
    _require_type(raw, (dict,), path)
    track_id = raw.get("track_id")
    _require_type(track_id, (str,), f"{path}.track_id")
    if not track_id:
        raise _schema_error(f"{path}.track_id must be non-empty")
    source = raw.get("source")
    _require_type(source, (str,), f"{path}.source")
    if not source:
        raise _schema_error(f"{path}.source must be non-empty")
    gain_db = _finite_number(raw.get("gain_db", 0.0), f"{path}.gain_db")
    if not (-120.0 <= gain_db <= 60.0):
        raise _schema_error(f"{path}.gain_db must be within [-120, 60] dB")
    muted = raw.get("muted", False)
    _require_type(muted, (bool,), f"{path}.muted")
    raw_events = raw.get("events")
    _require_type(raw_events, (list,), f"{path}.events")
    events = tuple(_parse_event(item, track_id, i) for i, item in enumerate(raw_events))
    return TrackSpec(track_id, source, gain_db, muted, events)


def parse_project(path: str | Path) -> Project:
    """Load and validate a project JSON file."""
    project_path = Path(path)
    try:
        text = project_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CineMixerError("E_PROJECT_NOT_FOUND", f"project file not found: {project_path}", 3) from exc
    except OSError as exc:
        raise CineMixerError("E_PROJECT_SCHEMA", f"cannot read project file: {exc}", 2) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _schema_error(f"invalid JSON: {exc.msg} at line {exc.lineno}") from exc
    return build_project(raw, project_path)


def build_project(raw: Any, source_path: str | Path = Path("project.json")) -> Project:
    """Validate an already-decoded project dictionary."""
    _require_type(raw, (dict,), "$")

    project_id = raw.get("project_id", "unnamed")
    _require_type(project_id, (str,), "project_id")

    sample_rate = raw.get("sample_rate")
    _require_type(sample_rate, (int,), "sample_rate")
    if not (MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE):
        raise _schema_error(
            f"sample_rate must be within [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}] Hz"
        )

    channels = raw.get("channels")
    _require_type(channels, (int,), "channels")
    if channels not in SUPPORTED_CHANNELS:
        raise _schema_error("channels must be 1 (mono) or 2 (stereo)")

    duration_ms = _finite_number(raw.get("duration_ms"), "duration_ms")
    if duration_ms <= 0.0:
        raise _schema_error("duration_ms must be > 0")

    target_lufs = _finite_number(raw.get("target_lufs"), "target_lufs")
    if not (-70.0 <= target_lufs <= -5.0):
        raise _schema_error("target_lufs must be within [-70, -5] LUFS")

    raw_output = raw.get("output")
    _require_type(raw_output, (dict,), "output")
    out_path = raw_output.get("path")
    _require_type(out_path, (str,), "output.path")
    if not out_path:
        raise _schema_error("output.path must be non-empty")
    out_format = raw_output.get("format", "wav")
    if out_format not in SUPPORTED_FORMATS:
        raise _schema_error(f"output.format must be one of {sorted(SUPPORTED_FORMATS)}")
    bit_depth = raw_output.get("bit_depth", 16)
    _require_type(bit_depth, (int,), "output.bit_depth")
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise _schema_error(f"output.bit_depth must be one of {sorted(SUPPORTED_BIT_DEPTHS)}")
    output = OutputSpec(out_path, out_format, bit_depth)

    compressor = _parse_compressor(raw.get("compressor"))

    raw_tracks = raw.get("tracks")
    _require_type(raw_tracks, (list,), "tracks")
    if not raw_tracks:
        raise _schema_error("tracks must contain at least one track")
    tracks = tuple(_parse_track(item, i) for i, item in enumerate(raw_tracks))

    track_ids = [track.track_id for track in tracks]
    if len(set(track_ids)) != len(track_ids):
        raise _schema_error("track_id values must be unique")
    seen_events: set[tuple[str, str]] = set()
    for track in tracks:
        event_ids = [event.event_id for event in track.events]
        if len(set(event_ids)) != len(event_ids):
            raise _schema_error(f"event_id values must be unique within track {track.track_id}")
        for event in track.events:
            key = (track.track_id, event.event_id)
            if key in seen_events:  # pragma: no cover - guarded by per-track check
                raise _schema_error(f"duplicate event {event.event_id}")
            seen_events.add(key)

    return Project(
        project_id=project_id,
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
        target_lufs=target_lufs,
        output=output,
        compressor=compressor,
        tracks=tracks,
        source_path=Path(source_path),
    )


def timeline_length_samples(project: Project) -> int:
    """Total render length in samples, rounded from the millisecond duration."""
    return int(round(project.duration_ms * project.sample_rate / 1000.0))
