"""Manifest parsing and validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import err
from .timing import as_decimal, frames_to_ms, ms_to_frames

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 384_000
VALID_SUBTYPES = {"PCM_16", "PCM_24", "FLOAT"}
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    path: Path
    sample_rate: int
    channels: int
    start_ms: Decimal
    gain_db: float
    role: str
    priority: int

    @property
    def is_dialogue(self) -> bool:
        role = self.role.lower()
        return role == "dialogue" or role.startswith("dialogue_") or role == "voice"


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    version: str
    segment_id: str
    sample_rate: int
    duration_ms: Decimal
    channels: int
    duration_frames: int


@dataclass(frozen=True)
class LoudnessSpec:
    target_lufs: float
    true_peak_db: float


@dataclass(frozen=True)
class DuckingSpec:
    duck_db: float
    safe_headroom_db: float


@dataclass(frozen=True)
class OutputSpec:
    master: Path | None
    report: Path | None
    sample_format: str


@dataclass(frozen=True)
class Manifest:
    source_path: Path
    sha256: str
    schema_version: int
    project: ProjectSpec
    loudness: LoudnessSpec
    ducking: DuckingSpec
    output: OutputSpec
    tracks: tuple[TrackSpec, ...]

    @property
    def manifest_dir(self) -> Path:
        return self.source_path.parent

    @property
    def project_key(self) -> str:
        identity = f"{self.project.name}-{self.project.segment_id}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity).strip("._-")[:72]
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
        return f"{safe or 'project'}-{digest}"


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise err(
            "E_BAD_MANIFEST",
            f"{field_name} must be a JSON object",
            field=field_name,
        )
    return value


def _required(mapping: dict[str, Any], key: str, field_name: str) -> Any:
    if key not in mapping:
        raise err(
            "E_BAD_MANIFEST",
            f"missing required field {field_name}.{key}",
            field=f"{field_name}.{key}",
        )
    return mapping[key]


def _string(value: Any, field_name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise err("E_BAD_MANIFEST", f"{field_name} must be a non-empty string", field=field_name)
    return value.strip()


def _int(
    value: Any,
    field_name: str,
    low: int,
    high: int,
    *,
    code: str | None = None,
) -> int:
    error_code = code or (
        "E_BAD_RATE" if field_name.endswith("sample_rate") else "E_BAD_MANIFEST"
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise err(error_code, f"{field_name} must be an integer", field=field_name)
    if value < low or value > high:
        raise err(
            error_code,
            f"{field_name} must be in [{low}, {high}]",
            field=field_name,
            value=value,
        )
    return value


def _finite_float(
    value: Any,
    field_name: str,
    *,
    low: float | None = None,
    high: float | None = None,
    code: str = "E_BAD_MANIFEST",
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise err(code, f"{field_name} must be a finite number", field=field_name)
    result = float(value)
    if not math.isfinite(result):
        raise err(code, f"{field_name} must be finite", field=field_name)
    if low is not None and result < low:
        raise err(code, f"{field_name} must be >= {low}", field=field_name, value=result)
    if high is not None and result > high:
        raise err(code, f"{field_name} must be <= {high}", field=field_name, value=result)
    return result


def _resolve_path(base: Path, value: Any, field_name: str) -> Path:
    text = _string(value, field_name)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def load_manifest(path: Path | str) -> Manifest:
    source_path = Path(path).resolve(strict=False)
    if not source_path.is_file():
        raise err(
            "E_MANIFEST_MISSING",
            "timeline manifest does not exist",
            exit_code=3,
            path=str(source_path),
        )

    raw = source_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        root = json.loads(text, parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise err(
            "E_BAD_MANIFEST",
            "timeline manifest is not valid UTF-8 JSON",
            line=getattr(exc, "lineno", None),
            column=getattr(exc, "colno", None),
        ) from exc

    root = _require_mapping(root, "manifest")
    schema_version = _int(root.get("schema_version", 1), "schema_version", 1, 1)
    project_raw = _require_mapping(_required(root, "project", "manifest"), "project")

    name = _string(_required(project_raw, "name", "project"), "project.name")
    version = _string(_required(project_raw, "version", "project"), "project.version")
    segment_id = _string(
        _required(project_raw, "segment_id", "project"), "project.segment_id"
    )
    sample_rate = _int(
        _required(project_raw, "sample_rate", "project"),
        "project.sample_rate",
        MIN_SAMPLE_RATE,
        MAX_SAMPLE_RATE,
    )
    channels = _int(
        project_raw.get("channels", 2),
        "project.channels",
        1,
        32,
        code="E_BAD_CHANNELS",
    )
    if channels != 2:
        raise err(
            "E_UNSUPPORTED_OUTPUT_CHANNELS",
            "master output must be stereo",
            field="project.channels",
            value=channels,
        )

    timeline_raw = root.get("timeline")
    if timeline_raw is None:
        timeline_raw = {}
    timeline_raw = _require_mapping(timeline_raw, "timeline")
    duration_value = project_raw.get("duration_ms", timeline_raw.get("duration_ms"))
    if duration_value is None:
        raise err(
            "E_BAD_MANIFEST",
            "project.duration_ms is required",
            field="project.duration_ms",
        )
    if isinstance(duration_value, bool) or not isinstance(
        duration_value, (int, float, Decimal)
    ):
        raise err("E_BAD_MANIFEST", "duration_ms must be numeric")
    duration_ms = as_decimal(duration_value)
    if not duration_ms.is_finite() or duration_ms <= 0:
        raise err("E_BAD_TIMELINE", "duration_ms must be finite and positive")
    duration_frames = ms_to_frames(duration_ms, sample_rate)
    if duration_frames <= 0:
        raise err("E_BAD_TIMELINE", "timeline is shorter than one sample")

    loudness_raw = root.get("loudness", {})
    loudness_raw = _require_mapping(loudness_raw, "loudness")
    target_lufs = _finite_float(
        loudness_raw.get("target_lufs", root.get("target_lufs")),
        "loudness.target_lufs",
        low=-70.0,
        high=0.0,
    )
    true_peak_db = _finite_float(
        loudness_raw.get("true_peak_db", root.get("true_peak_db")),
        "loudness.true_peak_db",
        low=-30.0,
        high=0.0,
    )

    ducking_raw = root.get("dialogue_ducking", {})
    ducking_raw = _require_mapping(ducking_raw, "dialogue_ducking")
    duck_db = _finite_float(
        ducking_raw.get("duck_db", -12.0),
        "dialogue_ducking.duck_db",
        low=-96.0,
        high=0.0,
    )
    safe_headroom_db = _finite_float(
        ducking_raw.get("safe_headroom_db", -0.5),
        "dialogue_ducking.safe_headroom_db",
        low=-20.0,
        high=0.0,
    )

    output_raw = root.get("output", {})
    output_raw = _require_mapping(output_raw, "output")
    master_value = output_raw.get("master")
    report_value = output_raw.get("report")
    sample_format = _string(
        output_raw.get("sample_format", "PCM_24"), "output.sample_format"
    ).upper()
    if sample_format not in VALID_SUBTYPES:
        raise err(
            "E_BAD_OUTPUT_FORMAT",
            f"output.sample_format must be one of {sorted(VALID_SUBTYPES)}",
            value=sample_format,
        )

    tracks_raw = root.get("tracks")
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise err("E_BAD_MANIFEST", "tracks must be a non-empty array", field="tracks")

    tracks: list[TrackSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(tracks_raw):
        field_name = f"tracks[{index}]"
        item = _require_mapping(item, field_name)
        track_id = _string(_required(item, "id", field_name), f"{field_name}.id")
        if not ID_RE.match(track_id):
            raise err(
                "E_BAD_MANIFEST",
                "track id may contain only letters, digits, dot, underscore and hyphen",
                field=f"{field_name}.id",
                value=track_id,
            )
        if track_id in seen_ids:
            raise err(
                "E_DUPLICATE_TRACK",
                "track ids must be unique",
                track_id=track_id,
            )
        seen_ids.add(track_id)

        track_path = _resolve_path(
            source_path.parent,
            _required(item, "path", field_name),
            f"{field_name}.path",
        )
        track_rate = _int(
            _required(item, "sample_rate", field_name),
            f"{field_name}.sample_rate",
            MIN_SAMPLE_RATE,
            MAX_SAMPLE_RATE,
        )
        track_channels = _int(
            _required(item, "channels", field_name),
            f"{field_name}.channels",
            1,
            32,
            code="E_BAD_CHANNELS",
        )
        start_value = _required(item, "start_ms", field_name)
        if isinstance(start_value, bool) or not isinstance(
            start_value, (int, float, Decimal)
        ):
            raise err(
                "E_BAD_START",
                f"{field_name}.start_ms must be numeric",
                field=f"{field_name}.start_ms",
            )
        start_ms = as_decimal(start_value)
        if not start_ms.is_finite() or start_ms < 0:
            raise err(
                "E_BAD_START",
                f"{field_name}.start_ms must be finite and non-negative",
                field=f"{field_name}.start_ms",
            )
        gain_db = _finite_float(
            item.get("gain_db", 0.0),
            f"{field_name}.gain_db",
            low=-120.0,
            high=24.0,
        )
        role = _string(item.get("role", "sfx"), f"{field_name}.role").lower()
        priority = _int(item.get("priority", 0), f"{field_name}.priority", 0, 1_000_000)
        tracks.append(
            TrackSpec(
                track_id=track_id,
                path=track_path,
                sample_rate=track_rate,
                channels=track_channels,
                start_ms=start_ms,
                gain_db=gain_db,
                role=role,
                priority=priority,
            )
        )

    output = OutputSpec(
        master=_resolve_path(source_path.parent, master_value, "output.master")
        if master_value is not None
        else None,
        report=_resolve_path(source_path.parent, report_value, "output.report")
        if report_value is not None
        else None,
        sample_format=sample_format,
    )
    project = ProjectSpec(
        name=name,
        version=version,
        segment_id=segment_id,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        channels=channels,
        duration_frames=duration_frames,
    )
    return Manifest(
        source_path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        project=project,
        loudness=LoudnessSpec(target_lufs=target_lufs, true_peak_db=true_peak_db),
        ducking=DuckingSpec(duck_db=duck_db, safe_headroom_db=safe_headroom_db),
        output=output,
        tracks=tuple(tracks),
    )


def track_timing_summary(track: TrackSpec, manifest: Manifest) -> dict[str, Any]:
    frame = ms_to_frames(track.start_ms, manifest.project.sample_rate)
    return {
        "track_id": track.track_id,
        "timeline_start_frame": frame,
        "requested_start_ms": float(track.start_ms),
        "resolved_start_ms": frames_to_ms(frame, manifest.project.sample_rate),
        "start_rounding_error_ms": abs(
            frames_to_ms(frame, manifest.project.sample_rate) - float(track.start_ms)
        ),
    }
