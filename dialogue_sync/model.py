"""Configuration and timeline models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError, TimecodeError
from .timecode import parse_timecode


@dataclass
class AudioConfig:
    sample_rate: int = 48_000
    bit_depth: int = 24
    channels: int = 2
    target_lufs: float = -23.0
    max_true_peak_dbtp: float = -1.0
    loudness_range: float = 11.0
    stretch_min: float = 0.8
    stretch_max: float = 1.25
    min_stretch_delta_ms: int = 25
    max_sample_peak: float = 0.999

    @property
    def channel_layout(self) -> str:
        return "mono" if self.channels == 1 else "stereo"


@dataclass
class OutputConfig:
    audio_filename: str = "dialogue_mix.wav"
    subtitle_filename: str = "dialogue_mix.srt"
    container_filename: str = "dialogue_mix.mkv"
    report_filename: str = "report.json"
    language: str = "eng"
    subtitle_language: str = "eng"
    title: str = "Dialogue Conform"
    normalize: bool = True


@dataclass
class DialogueEntry:
    ordinal: int
    entry_id: str
    source_path: Path
    source_file_as_given: str
    source_start_ms: int
    source_offset_ms: int
    target_start_ms: int
    subtitle_index: int
    source_end_ms: int | None = None
    target_end_ms: int | None = None
    target_duration_ms: int | None = None
    weight: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)
    field_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Project:
    timeline_path: Path
    srt_path: Path
    entries: list[DialogueEntry]
    audio: AudioConfig
    output: OutputConfig
    total_duration_ms: int | None
    timeline_raw: dict[str, Any]
    input_sha256: dict[str, str] = field(default_factory=dict)


_ENTRY_LIST_KEYS = (
    "dialogue_entries",
    "entries",
    "dialogues",
    "segments",
    "dialogue_segments",
)
_SOURCE_KEYS = ("source", "source_wav", "source_file", "sourceWav", "sourceFile", "wav")
_SOURCE_START_KEYS = (
    "source_start",
    "source_start_tc",
    "source_timecode",
    "sourceStart",
)
_TARGET_START_KEYS = (
    "target_start",
    "target_start_tc",
    "target_timecode",
    "targetStart",
)
_SUBTITLE_KEYS = (
    "subtitle_index",
    "subtitle",
    "subtitleIndex",
    "cue_index",
    "srt_index",
)
_SOURCE_END_KEYS = ("source_end", "source_end_tc", "sourceEnd")
_SOURCE_OFFSET_KEYS = (
    "source_offset",
    "source_file_start",
    "source_trim_start",
    "sourceOffset",
)
_TARGET_END_KEYS = ("target_end", "target_end_tc", "targetEnd")
_TARGET_DURATION_KEYS = ("target_duration", "target_duration_tc", "targetDuration")


def _first(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read timeline JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("timeline JSON root must be an object")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _find_entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    for key in _ENTRY_LIST_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            return value
    timeline = raw.get("timeline")
    if isinstance(timeline, dict):
        for key in _ENTRY_LIST_KEYS:
            value = timeline.get(key)
            if isinstance(value, list):
                return value
    raise ConfigError("timeline JSON must contain a dialogue_entries list")


def _find_total(raw: dict[str, Any]) -> Any:
    for container in (raw, raw.get("timeline")):
        if isinstance(container, dict):
            for key in (
                "total_duration",
                "duration",
                "timeline_duration",
                "totalDuration",
                "length",
            ):
                if container.get(key) is not None:
                    return container[key]
    return None


def _parse_positive_duration(value: Any, *, field_name: str) -> int:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be an HH:MM:SS.mmm string")
    try:
        ms = parse_timecode(value, field=field_name)
    except TimecodeError as exc:
        raise ConfigError(str(exc)) from exc
    if ms <= 0:
        raise ConfigError(f"{field_name} must be greater than zero")
    return ms


def _entry_timecode(
    value: Any,
    *,
    field_name: str,
    default: int | None,
    errors: list[dict[str, str]],
) -> int | None:
    try:
        return parse_timecode(value, field=field_name)
    except TimecodeError as exc:
        errors.append(
            {
                "code": "E_TIMECODE",
                "field": field_name,
                "value": str(value),
                "message": str(exc),
            }
        )
        return default


def _as_channels(value: Any) -> int:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"mono", "1", "1.0"}:
            return 1
        if lowered in {"stereo", "2", "2.0"}:
            return 2
    if isinstance(value, (int, float)) and int(value) in {1, 2}:
        return int(value)
    raise ConfigError("output channels must be 1/mono or 2/stereo")


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{field_name} must be finite")
    return number


def _parse_audio_config(raw: dict[str, Any]) -> AudioConfig:
    audio = AudioConfig()
    audio_raw = raw.get("audio") if isinstance(raw.get("audio"), dict) else {}
    output_raw = raw.get("output") if isinstance(raw.get("output"), dict) else {}
    top_level = {
        key: raw[key]
        for key in (
            "sample_rate",
            "bit_depth",
            "channels",
            "channel_layout",
            "target_lufs",
            "max_true_peak_dbtp",
            "loudness_range",
            "stretch_min",
            "stretch_max",
            "min_stretch_delta_ms",
        )
        if key in raw
    }
    merged = {**top_level, **audio_raw, **output_raw}
    if "channels" not in merged:
        for alias in ("output_channels", "audio_channels"):
            if alias in merged:
                merged["channels"] = merged[alias]
                break
    if "sample_rate" in merged:
        audio.sample_rate = int(merged["sample_rate"])
    if "bit_depth" in merged:
        audio.bit_depth = int(merged["bit_depth"])
    if "channels" in merged:
        audio.channels = _as_channels(merged["channels"])
    elif "channel_layout" in merged:
        audio.channels = _as_channels(merged["channel_layout"])
    if "target_lufs" in merged:
        audio.target_lufs = _finite_float(merged["target_lufs"], field_name="target_lufs")
    if "max_true_peak_dbtp" in merged:
        audio.max_true_peak_dbtp = _finite_float(
            merged["max_true_peak_dbtp"], field_name="max_true_peak_dbtp"
        )
    if "loudness_range" in merged:
        audio.loudness_range = _finite_float(
            merged["loudness_range"], field_name="loudness_range"
        )
    if "stretch_min" in merged:
        audio.stretch_min = _finite_float(merged["stretch_min"], field_name="stretch_min")
    if "stretch_max" in merged:
        audio.stretch_max = _finite_float(merged["stretch_max"], field_name="stretch_max")
    if "min_stretch_delta_ms" in merged:
        audio.min_stretch_delta_ms = int(merged["min_stretch_delta_ms"])

    if audio.sample_rate != 48_000:
        raise ConfigError("this delivery profile requires a 48 kHz output sample rate")
    if audio.bit_depth != 24:
        raise ConfigError("this delivery profile requires 24-bit output PCM")
    if audio.channels not in (1, 2):
        raise ConfigError("output must be mono or stereo")
    if not (audio.max_true_peak_dbtp < 0):
        raise ConfigError("max_true_peak_dbtp must be below 0 dBTP")
    if not (0 < audio.stretch_min <= 1 <= audio.stretch_max):
        raise ConfigError("stretch limits must satisfy 0 < min <= 1 <= max")
    if audio.min_stretch_delta_ms < 0:
        raise ConfigError("min_stretch_delta_ms cannot be negative")
    return audio


def _parse_output_config(raw: dict[str, Any]) -> OutputConfig:
    output = OutputConfig()
    output_raw = raw.get("output") if isinstance(raw.get("output"), dict) else {}
    container_raw = raw.get("container") if isinstance(raw.get("container"), dict) else {}
    top_level = {
        key: raw[key]
        for key in (
            "language",
            "audio_language",
            "subtitle_language",
            "title",
            "audio_filename",
            "subtitle_filename",
            "container_filename",
            "report_filename",
        )
        if key in raw
    }
    merged = {**top_level, **output_raw, **container_raw}
    if "language" not in merged and "audio_language" in merged:
        merged["language"] = merged["audio_language"]
    for attr in (
        "audio_filename",
        "subtitle_filename",
        "container_filename",
        "report_filename",
    ):
        if attr in merged:
            setattr(output, attr, str(merged[attr]))
    for attr in ("language", "subtitle_language", "title"):
        if attr in merged:
            setattr(output, attr, str(merged[attr]))
    for attr in ("audio_filename", "subtitle_filename", "container_filename", "report_filename"):
        value = getattr(output, attr)
        if not value or Path(value).name != value:
            raise ConfigError(f"{attr} must be a filename, not a path")
    filenames = [
        output.audio_filename,
        output.subtitle_filename,
        output.container_filename,
        output.report_filename,
    ]
    if len(set(filenames)) != len(filenames):
        raise ConfigError("audio, subtitle, container, and report filenames must differ")
    if not output.language.strip():
        raise ConfigError("container language cannot be empty")
    if not output.subtitle_language.strip():
        raise ConfigError("subtitle language cannot be empty")
    return output


def load_project(
    timeline_path: Path,
    *,
    srt_path: Path | None = None,
    channels: int | str | None = None,
    language: str | None = None,
    subtitle_language: str | None = None,
    total_duration: str | None = None,
) -> Project:
    timeline_path = timeline_path.expanduser().resolve()
    raw = _load_json(timeline_path)
    base_dir = timeline_path.parent

    srt_value = srt_path
    if srt_value is None:
        candidate = (
            raw.get("srt")
            or raw.get("srt_path")
            or raw.get("subtitle_file")
            or raw.get("subtitles")
        )
        if isinstance(candidate, dict):
            candidate = candidate.get("path") or candidate.get("file")
        if isinstance(candidate, str):
            srt_value = _resolve_path(candidate, base_dir)
    if srt_value is None:
        raise ConfigError("SRT path must be supplied by --srt or timeline JSON")
    srt_path = srt_value.expanduser().resolve()

    if channels is not None:
        raw.setdefault("output", {})
        if not isinstance(raw["output"], dict):
            raise ConfigError("output must be an object")
        raw["output"]["channels"] = channels
        raw["output"].pop("channel_layout", None)
    if language is not None:
        raw.setdefault("output", {})
        if not isinstance(raw["output"], dict):
            raise ConfigError("output must be an object")
        raw["output"]["language"] = language
    if subtitle_language is not None:
        raw.setdefault("output", {})
        if not isinstance(raw["output"], dict):
            raise ConfigError("output must be an object")
        raw["output"]["subtitle_language"] = subtitle_language
    if total_duration is not None:
        raw["total_duration"] = total_duration

    audio = _parse_audio_config(raw)
    output = _parse_output_config(raw)
    entries_raw = _find_entries(raw)
    if not entries_raw:
        raise ConfigError("dialogue entry list is empty")

    entries: list[DialogueEntry] = []
    for position, item in enumerate(entries_raw, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"dialogue entry {position} must be an object")
        source_value = _first(item, _SOURCE_KEYS)
        if not isinstance(source_value, str) or not source_value.strip():
            raise ConfigError(f"dialogue entry {position} has no source WAV")
        source_start_value = _first(item, _SOURCE_START_KEYS)
        target_start_value = _first(item, _TARGET_START_KEYS)
        subtitle_value = _first(item, _SUBTITLE_KEYS)
        if source_start_value is None:
            raise ConfigError(f"dialogue entry {position} has no source_start")
        if target_start_value is None:
            raise ConfigError(f"dialogue entry {position} has no target_start")
        if subtitle_value is None:
            raise ConfigError(f"dialogue entry {position} has no subtitle_index")
        field_errors: list[dict[str, str]] = []
        source_start_ms = _entry_timecode(
            source_start_value,
            field_name=f"entry {position} source_start",
            default=0,
            errors=field_errors,
        )
        target_start_ms = _entry_timecode(
            target_start_value,
            field_name=f"entry {position} target_start",
            default=0,
            errors=field_errors,
        )
        assert source_start_ms is not None and target_start_ms is not None
        if isinstance(subtitle_value, dict):
            subtitle_value = subtitle_value.get("index")
        try:
            subtitle_index = int(subtitle_value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"entry {position} subtitle_index must be an integer") from exc
        if subtitle_index <= 0:
            raise ConfigError(f"entry {position} subtitle_index must be positive")

        source_end_ms = None
        source_end_value = _first(item, _SOURCE_END_KEYS)
        if source_end_value is not None:
            source_end_ms = _entry_timecode(
                source_end_value,
                field_name=f"entry {position} source_end",
                default=None,
                errors=field_errors,
            )
            if source_end_ms is not None and source_end_ms <= source_start_ms:
                field_errors.append(
                    {
                        "code": "E_TIMECODE",
                        "field": f"entry {position} source_end",
                        "value": str(source_end_value),
                        "message": "source_end must follow source_start",
                    }
                )

        source_offset_ms = 0
        source_offset_value = _first(item, _SOURCE_OFFSET_KEYS)
        if source_offset_value is not None:
            parsed_offset = _entry_timecode(
                source_offset_value,
                field_name=f"entry {position} source_offset",
                default=0,
                errors=field_errors,
            )
            source_offset_ms = parsed_offset or 0

        target_end_ms = None
        target_end_value = _first(item, _TARGET_END_KEYS)
        if target_end_value is not None:
            target_end_ms = _entry_timecode(
                target_end_value,
                field_name=f"entry {position} target_end",
                default=None,
                errors=field_errors,
            )
            if target_end_ms is not None and target_end_ms <= target_start_ms:
                field_errors.append(
                    {
                        "code": "E_TIMECODE",
                        "field": f"entry {position} target_end",
                        "value": str(target_end_value),
                        "message": "target_end must follow target_start",
                    }
                )

        target_duration_ms = None
        target_duration_value = _first(item, _TARGET_DURATION_KEYS)
        if target_duration_value is not None:
            target_duration_ms = _entry_timecode(
                target_duration_value,
                field_name=f"entry {position} target_duration",
                default=None,
                errors=field_errors,
            )
            if target_duration_ms is not None and target_duration_ms <= 0:
                field_errors.append(
                    {
                        "code": "E_TIMECODE",
                        "field": f"entry {position} target_duration",
                        "value": str(target_duration_value),
                        "message": "target_duration must be positive",
                    }
                )
        if target_end_ms is not None:
            implied_duration = target_end_ms - target_start_ms
            if (
                target_duration_ms is not None
                and abs(implied_duration - target_duration_ms) > 1
            ):
                field_errors.append(
                    {
                        "code": "E_TIMECODE",
                        "field": f"entry {position} target_end",
                        "value": str(target_end_value),
                        "message": "target_end and target_duration disagree",
                    }
                )
            target_duration_ms = implied_duration

        weight = _finite_float(item.get("weight", 1.0), field_name=f"entry {position} weight")
        if weight < 0 or weight > 4:
            raise ConfigError(f"entry {position} weight must be between 0 and 4")

        entry_id_values = (
            str(item.get("id") or item.get("entry_id") or "").strip(),
        )
        entry_id = entry_id_values[0] or f"entry_{position:03d}"
        source_path = _resolve_path(source_value, base_dir)
        entries.append(
            DialogueEntry(
                ordinal=position,
                entry_id=entry_id,
                source_path=source_path,
                source_file_as_given=source_value,
                source_start_ms=source_start_ms,
                source_offset_ms=source_offset_ms,
                target_start_ms=target_start_ms,
                subtitle_index=subtitle_index,
                source_end_ms=source_end_ms,
                target_end_ms=target_end_ms,
                target_duration_ms=target_duration_ms,
                weight=weight,
                raw=item,
                field_errors=field_errors,
            )
        )

    total_raw = _find_total(raw) if total_duration is None else total_duration
    total_ms = None
    if total_raw is not None:
        total_ms = _parse_positive_duration(total_raw, field_name="total_duration")
    return Project(
        timeline_path=timeline_path,
        srt_path=srt_path,
        entries=entries,
        audio=audio,
        output=output,
        total_duration_ms=total_ms,
        timeline_raw=raw,
    )


def validate_timeline_against_subtitles(project: Project, cues: dict[int, Any]) -> None:
    """Reject ambiguous cue references before processing starts."""

    referenced: dict[int, int] = {}
    for entry in project.entries:
        cue = cues.get(entry.subtitle_index)
        if cue is None:
            raise ConfigError(
                f"entry {entry.ordinal} references missing subtitle {entry.subtitle_index}"
            )
        if entry.subtitle_index in referenced:
            raise ConfigError(
                f"subtitle {entry.subtitle_index} is referenced by entries "
                f"{referenced[entry.subtitle_index]} and {entry.ordinal}"
            )
        referenced[entry.subtitle_index] = entry.ordinal
        if entry.target_duration_ms is None:
            entry.target_duration_ms = cue.end_ms - cue.start_ms
        if entry.target_duration_ms <= 0:
            raise ConfigError(f"entry {entry.ordinal} resolves to a non-positive duration")


def derive_total_duration(project: Project) -> int:
    if project.total_duration_ms is not None:
        total_ms = project.total_duration_ms
    else:
        ends: list[int] = []
        for entry in project.entries:
            duration = entry.target_duration_ms
            if duration is None:
                raise ConfigError(
                    f"entry {entry.ordinal} has no target_duration and no subtitle duration"
                )
            ends.append(entry.target_start_ms + duration)
        total_ms = max(ends, default=0)
    if total_ms <= 0:
        raise ConfigError("target timeline duration must be positive")
    return total_ms

