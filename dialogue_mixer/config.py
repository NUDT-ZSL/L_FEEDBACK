"""Project JSON parsing and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .envelope import SUPPORTED_CURVES


VALID_TRACK_KINDS = ("dialogue", "music", "sfx", "other")
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 384000


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


class ProjectFatalError(RuntimeError):
    def __init__(self, code: str, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


@dataclass
class TrackConfig:
    id: str
    kind: str
    gain: float


@dataclass
class ClipConfig:
    id: str
    track_id: str
    file_name: str
    path: Path
    start_seconds: float
    gain: float


@dataclass
class DialogueConfig:
    id: str
    start_seconds: float
    end_seconds: float
    stretch_ratio: Optional[float]
    voice_clip_id: Optional[str]
    duck_tracks: List[str]
    depth_db: Optional[float] = None
    attack_seconds: Optional[float] = None
    release_seconds: Optional[float] = None
    attack_curve: Optional[str] = None
    release_curve: Optional[str] = None


@dataclass
class DuckingConfig:
    enabled: bool
    target_tracks: List[str]
    depth_db: float
    attack_seconds: float
    release_seconds: float
    attack_curve: str
    release_curve: str


@dataclass
class CrossfadeConfig:
    from_clip_id: str
    to_clip_id: str
    duration_seconds: float


@dataclass
class NormalizationConfig:
    peak_ceiling_dbfs: float


@dataclass
class OutputConfig:
    audio_path: Path
    index_path: Path
    log_path: Path


@dataclass
class Project:
    name: str
    source_path: Path
    sample_rate: int
    duration_seconds: float
    total_samples: int
    tracks: Dict[str, TrackConfig]
    clips: List[ClipConfig]
    dialogues: List[DialogueConfig]
    ducking: DuckingConfig
    crossfades: List[CrossfadeConfig]
    normalization: NormalizationConfig
    output: OutputConfig


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_finite_float(
    value: Any,
    field_name: str,
    issues: List[Issue],
    default: Optional[float] = None,
) -> Optional[float]:
    if not _is_number(value):
        issues.append(
            Issue(
                "INVALID_PROJECT_FIELD",
                "error",
                "{0} must be a finite number".format(field_name),
                {"field": field_name, "value": repr(value)},
            )
        )
        return default
    return float(value)


def _resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _parse_curve(
    raw: Dict[str, Any],
    key: str,
    default: str,
    issues: List[Issue],
    context: Dict[str, Any],
) -> str:
    value = str(raw.get(key, default)).strip().lower()
    if value == "equal_power":
        value = "sine"
    if value not in SUPPORTED_CURVES:
        issues.append(
            Issue(
                "INVALID_ENVELOPE_CURVE",
                "error",
                "unsupported {0}; using cosine".format(key),
                dict(context, curve=value, fallback="cosine"),
            )
        )
        return "cosine"
    return value


def parse_project(path: Path) -> tuple[Project, List[Issue]]:
    """Parse a project or raise ProjectFatalError when no timeline is possible."""

    path = Path(path).expanduser().resolve()
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ProjectFatalError(
            "PROJECT_READ_ERROR",
            "cannot read project file: {0}".format(exc),
            {"path": str(path)},
        ) from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProjectFatalError(
            "INVALID_PROJECT_JSON",
            "project is not valid JSON: {0}".format(exc),
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(raw, dict):
        raise ProjectFatalError(
            "INVALID_PROJECT_ROOT", "project root must be a JSON object", {"path": str(path)}
        )

    issues: List[Issue] = []
    project_dir = path.parent

    sample_rate_value = raw.get("sample_rate")
    if (
        not _is_number(sample_rate_value)
        or float(sample_rate_value) != int(float(sample_rate_value))
        or int(sample_rate_value) < MIN_SAMPLE_RATE
        or int(sample_rate_value) > MAX_SAMPLE_RATE
    ):
        raise ProjectFatalError(
            "INVALID_SAMPLE_RATE",
            "sample_rate must be an integer from {0} to {1}".format(
                MIN_SAMPLE_RATE, MAX_SAMPLE_RATE
            ),
            {"sample_rate": repr(sample_rate_value)},
        )
    sample_rate = int(sample_rate_value)

    duration_value = raw.get("duration_seconds")
    if not _is_number(duration_value) or float(duration_value) <= 0.0:
        raise ProjectFatalError(
            "INVALID_DURATION",
            "duration_seconds must be a positive finite number",
            {"duration_seconds": repr(duration_value)},
        )
    duration_seconds = float(duration_value)
    total_samples = int(round(duration_seconds * sample_rate))
    if total_samples <= 0:
        raise ProjectFatalError(
            "INVALID_DURATION",
            "duration_seconds rounds to zero samples",
            {"duration_seconds": duration_seconds, "sample_rate": sample_rate},
        )

    tracks_raw = raw.get("tracks")
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise ProjectFatalError(
            "MISSING_TRACKS", "tracks must be a non-empty array", {"path": str(path)}
        )

    tracks: Dict[str, TrackConfig] = {}
    for index, item in enumerate(tracks_raw):
        context = {"index": index}
        if not isinstance(item, dict):
            issues.append(
                Issue("INVALID_TRACK", "error", "track entry must be an object", context)
            )
            continue
        track_id = str(item.get("id", "")).strip()
        if not track_id:
            issues.append(
                Issue("INVALID_TRACK", "error", "track id is required", context)
            )
            continue
        context["track_id"] = track_id
        if track_id in tracks:
            issues.append(
                Issue(
                    "DUPLICATE_TRACK_ID",
                    "error",
                    "duplicate track id; later definition ignored",
                    context,
                )
            )
            continue
        kind = str(item.get("kind", "other")).strip().lower()
        if kind not in VALID_TRACK_KINDS:
            issues.append(
                Issue(
                    "INVALID_TRACK_KIND",
                    "error",
                    "unknown track kind; treated as 'other'",
                    dict(context, kind=kind),
                )
            )
            kind = "other"
        gain_value = item.get("gain", 1.0)
        gain = _as_finite_float(gain_value, "tracks[].gain", issues, default=1.0)
        if gain is not None and gain < 0.0:
            issues.append(
                Issue(
                    "INVALID_GAIN",
                    "error",
                    "track gain must be non-negative; using 1.0",
                    dict(context, gain=gain),
                )
            )
            gain = 1.0
        tracks[track_id] = TrackConfig(id=track_id, kind=kind, gain=float(gain or 0.0))

    ducking_raw = raw.get("ducking", {})
    if not isinstance(ducking_raw, dict):
        issues.append(
            Issue(
                "INVALID_DUCKING_CONFIG",
                "error",
                "ducking must be an object; defaults used",
            )
        )
        ducking_raw = {}
    enabled = bool(ducking_raw.get("enabled", True))
    depth_db = _as_finite_float(
        ducking_raw.get("depth_db", -12.0),
        "ducking.depth_db",
        issues,
        default=-12.0,
    )
    if depth_db is None:
        depth_db = -12.0
    if depth_db > 0.0:
        issues.append(
            Issue(
                "INVALID_DUCK_DEPTH",
                "error",
                "ducking depth_db cannot be positive; clamped to 0 dB",
                {"depth_db": depth_db},
            )
        )
        depth_db = 0.0
    attack_seconds = _as_finite_float(
        ducking_raw.get("attack_seconds", 0.05),
        "ducking.attack_seconds",
        issues,
        default=0.05,
    )
    release_seconds = _as_finite_float(
        ducking_raw.get("release_seconds", 0.30),
        "ducking.release_seconds",
        issues,
        default=0.30,
    )
    if attack_seconds is None or attack_seconds < 0.0:
        attack_seconds = 0.05
        issues.append(
            Issue(
                "INVALID_DUCK_TIME",
                "error",
                "ducking attack_seconds must be non-negative; using 0.05",
            )
        )
    if release_seconds is None or release_seconds < 0.0:
        release_seconds = 0.30
        issues.append(
            Issue(
                "INVALID_DUCK_TIME",
                "error",
                "ducking release_seconds must be non-negative; using 0.30",
            )
        )
    target_tracks_raw = ducking_raw.get("target_tracks", [])
    if not isinstance(target_tracks_raw, list):
        issues.append(
            Issue(
                "INVALID_DUCK_TARGETS",
                "error",
                "ducking.target_tracks must be an array; using []",
            )
        )
        target_tracks_raw = []
    target_tracks = [str(item).strip() for item in target_tracks_raw if str(item).strip()]
    attack_curve = _parse_curve(
        ducking_raw, "attack_curve", "cosine", issues, {"section": "ducking"}
    )
    release_curve = _parse_curve(
        ducking_raw, "release_curve", "cosine", issues, {"section": "ducking"}
    )

    clips_raw = raw.get("clips", [])
    if not isinstance(clips_raw, list):
        issues.append(
            Issue("INVALID_CLIPS", "error", "clips must be an array; no clips loaded")
        )
        clips_raw = []
    clips: List[ClipConfig] = []
    clip_ids = set()
    for index, item in enumerate(clips_raw):
        context = {"index": index}
        if not isinstance(item, dict):
            issues.append(
                Issue("INVALID_CLIP", "error", "clip entry must be an object", context)
            )
            continue
        clip_id = str(item.get("id", "")).strip()
        track_id = str(item.get("track", item.get("track_id", ""))).strip()
        file_name = str(item.get("file", "")).strip()
        context.update({"clip_id": clip_id, "track_id": track_id, "file": file_name})
        if not clip_id or not track_id or not file_name:
            issues.append(
                Issue(
                    "INVALID_CLIP",
                    "error",
                    "clip id, track and file are required",
                    context,
                )
            )
            continue
        if clip_id in clip_ids:
            issues.append(
                Issue(
                    "DUPLICATE_CLIP_ID",
                    "error",
                    "duplicate clip id; later clip ignored",
                    context,
                )
            )
            continue
        start = _as_finite_float(
            item.get("start", item.get("start_seconds", 0.0)),
            "clips[].start",
            issues,
            default=None,
        )
        gain = _as_finite_float(
            item.get("gain", 1.0), "clips[].gain", issues, default=None
        )
        if start is None or start < 0.0:
            issues.append(
                Issue(
                    "INVALID_CLIP_START",
                    "error",
                    "clip start must be a non-negative number",
                    context,
                )
            )
            continue
        if gain is None or gain < 0.0:
            issues.append(
                Issue(
                    "INVALID_GAIN",
                    "error",
                    "clip gain must be a non-negative number",
                    context,
                )
            )
            continue
        clip_ids.add(clip_id)
        clips.append(
            ClipConfig(
                id=clip_id,
                track_id=track_id,
                file_name=file_name,
                path=_resolve_path(file_name, project_dir),
                start_seconds=float(start),
                gain=float(gain),
            )
        )

    dialogues_raw = raw.get("dialogues", [])
    if not isinstance(dialogues_raw, list):
        issues.append(
            Issue(
                "INVALID_DIALOGUES",
                "error",
                "dialogues must be an array; no dialogues loaded",
            )
        )
        dialogues_raw = []
    dialogues: List[DialogueConfig] = []
    dialogue_ids = set()
    for index, item in enumerate(dialogues_raw):
        context = {"index": index}
        if not isinstance(item, dict):
            issues.append(
                Issue(
                    "INVALID_DIALOGUE",
                    "error",
                    "dialogue entry must be an object",
                    context,
                )
            )
            continue
        dialogue_id = str(item.get("id", "")).strip()
        if not dialogue_id:
            dialogue_id = "dialogue_{0}".format(index + 1)
        context["dialogue_id"] = dialogue_id
        if dialogue_id in dialogue_ids:
            issues.append(
                Issue(
                    "DUPLICATE_DIALOGUE_ID",
                    "error",
                    "duplicate dialogue id; later dialogue ignored",
                    context,
                )
            )
            continue
        start = _as_finite_float(
            item.get("start", item.get("start_seconds")),
            "dialogues[].start",
            issues,
            default=None,
        )
        end = _as_finite_float(
            item.get("end", item.get("end_seconds")),
            "dialogues[].end",
            issues,
            default=None,
        )
        if start is None or end is None or start < 0.0 or end <= start:
            issues.append(
                Issue(
                    "INVALID_DIALOGUE_INTERVAL",
                    "error",
                    "dialogue requires 0 <= start < end",
                    dict(context, start=start, end=end),
                )
            )
            continue
        if end * sample_rate > total_samples + 1e-6:
            issues.append(
                Issue(
                    "DIALOGUE_OUTSIDE_TIMELINE",
                    "error",
                    "dialogue ends outside the project duration; dialogue ignored",
                    dict(context, start=start, end=end, total_seconds=total_samples / sample_rate),
                )
            )
            continue
        stretch_ratio = None
        if "stretch_ratio" in item and item.get("stretch_ratio") is not None:
            stretch_value = _as_finite_float(
                item.get("stretch_ratio"),
                "dialogues[].stretch_ratio",
                issues,
                default=None,
            )
            if stretch_value is None or stretch_value <= 0.0:
                issues.append(
                    Issue(
                        "INVALID_STRETCH_RATIO",
                        "error",
                        "stretch_ratio must be positive; dialogue ignored",
                        context,
                    )
                )
                continue
            stretch_ratio = float(stretch_value)

        voice_clip_id = item.get("voice_clip_id")
        if voice_clip_id is not None:
            voice_clip_id = str(voice_clip_id).strip() or None
        duck_tracks_raw = item.get("duck_tracks", None)
        if duck_tracks_raw is None:
            duck_tracks = list(target_tracks)
        elif isinstance(duck_tracks_raw, list):
            duck_tracks = [
                str(track).strip() for track in duck_tracks_raw if str(track).strip()
            ]
        else:
            issues.append(
                Issue(
                    "INVALID_DUCK_TARGETS",
                    "error",
                    "dialogue duck_tracks must be an array; global targets used",
                    context,
                )
            )
            duck_tracks = list(target_tracks)

        depth_override = None
        if item.get("depth_db") is not None:
            depth_override = _as_finite_float(
                item.get("depth_db"), "dialogues[].depth_db", issues, default=None
            )
            if depth_override is None or depth_override > 0.0:
                issues.append(
                    Issue(
                        "INVALID_DUCK_DEPTH",
                        "error",
                        "dialogue depth_db must be <= 0; global value used",
                        context,
                    )
                )
                depth_override = None

        attack_override = None
        if item.get("attack_seconds") is not None:
            attack_override = _as_finite_float(
                item.get("attack_seconds"),
                "dialogues[].attack_seconds",
                issues,
                default=None,
            )
            if attack_override is None or attack_override < 0.0:
                issues.append(
                    Issue(
                        "INVALID_DUCK_TIME",
                        "error",
                        "dialogue attack_seconds must be non-negative; global value used",
                        context,
                    )
                )
                attack_override = None
        release_override = None
        if item.get("release_seconds") is not None:
            release_override = _as_finite_float(
                item.get("release_seconds"),
                "dialogues[].release_seconds",
                issues,
                default=None,
            )
            if release_override is None or release_override < 0.0:
                issues.append(
                    Issue(
                        "INVALID_DUCK_TIME",
                        "error",
                        "dialogue release_seconds must be non-negative; global value used",
                        context,
                    )
                )
                release_override = None

        dialogue_ids.add(dialogue_id)
        dialogues.append(
            DialogueConfig(
                id=dialogue_id,
                start_seconds=float(start),
                end_seconds=float(end),
                stretch_ratio=stretch_ratio,
                voice_clip_id=voice_clip_id,
                duck_tracks=duck_tracks,
                depth_db=depth_override,
                attack_seconds=attack_override,
                release_seconds=release_override,
                attack_curve=(
                    str(item.get("attack_curve")).lower()
                    if item.get("attack_curve") is not None
                    else None
                ),
                release_curve=(
                    str(item.get("release_curve")).lower()
                    if item.get("release_curve") is not None
                    else None
                ),
            )
        )

    for dialogue in dialogues:
        if dialogue.attack_curve is not None and dialogue.attack_curve == "equal_power":
            dialogue.attack_curve = "sine"
        if dialogue.release_curve is not None and dialogue.release_curve == "equal_power":
            dialogue.release_curve = "sine"
        if dialogue.attack_curve is not None and dialogue.attack_curve not in SUPPORTED_CURVES:
            issues.append(
                Issue(
                    "INVALID_ENVELOPE_CURVE",
                    "error",
                    "dialogue attack_curve is unsupported; global value used",
                    {"dialogue_id": dialogue.id, "curve": dialogue.attack_curve},
                )
            )
            dialogue.attack_curve = None
        if dialogue.release_curve is not None and dialogue.release_curve not in SUPPORTED_CURVES:
            issues.append(
                Issue(
                    "INVALID_ENVELOPE_CURVE",
                    "error",
                    "dialogue release_curve is unsupported; global value used",
                    {"dialogue_id": dialogue.id, "curve": dialogue.release_curve},
                )
            )
            dialogue.release_curve = None

    crossfades_raw = raw.get("crossfades", [])
    if not isinstance(crossfades_raw, list):
        issues.append(
            Issue(
                "INVALID_CROSSFADES",
                "error",
                "crossfades must be an array; no crossfades loaded",
            )
        )
        crossfades_raw = []
    crossfades: List[CrossfadeConfig] = []
    for index, item in enumerate(crossfades_raw):
        context = {"index": index}
        if not isinstance(item, dict):
            issues.append(
                Issue(
                    "INVALID_CROSSFADE",
                    "error",
                    "crossfade entry must be an object",
                    context,
                )
            )
            continue
        from_id = str(item.get("from_clip_id", "")).strip()
        to_id = str(item.get("to_clip_id", "")).strip()
        duration = _as_finite_float(
            item.get("duration_seconds", item.get("duration")),
            "crossfades[].duration_seconds",
            issues,
            default=None,
        )
        context.update({"from_clip_id": from_id, "to_clip_id": to_id})
        if not from_id or not to_id or duration is None or duration <= 0.0:
            issues.append(
                Issue(
                    "INVALID_CROSSFADE",
                    "error",
                    "crossfade requires from_clip_id, to_clip_id and positive duration",
                    context,
                )
            )
            continue
        crossfades.append(
            CrossfadeConfig(
                from_clip_id=from_id,
                to_clip_id=to_id,
                duration_seconds=float(duration),
            )
        )

    normalization_raw = raw.get("normalization", {})
    if not isinstance(normalization_raw, dict):
        issues.append(
            Issue(
                "INVALID_NORMALIZATION",
                "error",
                "normalization must be an object; default used",
            )
        )
        normalization_raw = {}
    ceiling = _as_finite_float(
        normalization_raw.get("peak_ceiling_dbfs", 0.0),
        "normalization.peak_ceiling_dbfs",
        issues,
        default=0.0,
    )
    if ceiling is None:
        ceiling = 0.0
    if ceiling > 0.0:
        issues.append(
            Issue(
                "INVALID_PEAK_CEILING",
                "error",
                "peak ceiling cannot exceed 0 dBFS; clamped to 0 dBFS",
                {"peak_ceiling_dbfs": ceiling},
            )
        )
        ceiling = 0.0

    output_raw = raw.get("output", {})
    if not isinstance(output_raw, dict):
        issues.append(
            Issue(
                "INVALID_OUTPUT_CONFIG",
                "error",
                "output must be an object; default paths used",
            )
        )
        output_raw = {}
    stem = path.stem
    audio_name = str(output_raw.get("audio", stem + ".mix.wav"))
    index_name = str(output_raw.get("index", stem + ".track_index.json"))
    log_name = str(output_raw.get("log", stem + ".run_log.json"))

    project = Project(
        name=str(raw.get("name", path.stem)),
        source_path=path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        total_samples=total_samples,
        tracks=tracks,
        clips=clips,
        dialogues=dialogues,
        ducking=DuckingConfig(
            enabled=enabled,
            target_tracks=target_tracks,
            depth_db=float(depth_db),
            attack_seconds=float(attack_seconds),
            release_seconds=float(release_seconds),
            attack_curve=attack_curve,
            release_curve=release_curve,
        ),
        crossfades=crossfades,
        normalization=NormalizationConfig(peak_ceiling_dbfs=float(ceiling)),
        output=OutputConfig(
            audio_path=_resolve_path(audio_name, project_dir),
            index_path=_resolve_path(index_name, project_dir),
            log_path=_resolve_path(log_name, project_dir),
        ),
    )
    return project, issues


def find_project_log_path(path: Path) -> Path:
    """Best-effort log path discovery for parse failures."""

    path = Path(path).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict) and isinstance(raw.get("output"), dict):
            value = raw["output"].get("log")
            if value:
                return _resolve_path(str(value), path.resolve().parent)
    except (OSError, ValueError, TypeError):
        pass
    return path.resolve().with_name(path.stem + ".run_log.json")

