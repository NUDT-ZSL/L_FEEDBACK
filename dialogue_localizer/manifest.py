"""Manifest parsing and strict validation.

The canonical schema is documented in README.md.  A small set of aliases is
accepted so older localization hand-offs can be consumed without rewriting
their timeline files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import EXIT_MANIFEST, BusinessError


@dataclass(frozen=True)
class VideoSpec:
    duration_sec: float
    fps: float | None
    width: int | None
    height: int | None
    path: str | None


@dataclass(frozen=True)
class SourceAudioSpec:
    sample_rate: int
    channels: int
    path: str
    validate_targets: bool = True


@dataclass(frozen=True)
class TargetSegment:
    segment_id: str
    source_start_sec: float
    source_end_sec: float
    target_start_sec: float
    target_end_sec: float
    text: str | dict[str, Any]
    target_text: str | None
    audio_path: str
    expected_sample_rate: int
    expected_channels: int


@dataclass
class Manifest:
    path: str
    video: VideoSpec
    source_audio: SourceAudioSpec
    segments: list[TargetSegment] = field(default_factory=list)


def _manifest_error(
    code: str,
    message: str,
    *,
    path: str,
    field_path: str | None = None,
    details: dict[str, Any] | None = None,
) -> BusinessError:
    merged = dict(details or {})
    if field_path is not None:
        merged["field"] = field_path
    return BusinessError(
        code=code,
        message=message,
        exit_code=EXIT_MANIFEST,
        file=path,
        details=merged,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _as_object(value: Any, *, path: str, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _manifest_error(
            "MISSING_FIELD",
            f"'{field_path}' must be a JSON object",
            path=path,
            field_path=field_path,
        )
    return value


def _require(
    obj: dict[str, Any],
    key: str,
    *,
    path: str,
    field_path: str,
) -> Any:
    if key not in obj or obj[key] is None:
        raise _manifest_error(
            "MISSING_FIELD",
            f"required field '{field_path}' is missing",
            path=path,
            field_path=field_path,
        )
    return obj[key]


def _first_present(
    obj: dict[str, Any],
    names: tuple[str, ...],
) -> tuple[str, Any] | None:
    for name in names:
        if name in obj and obj[name] is not None:
            return name, obj[name]
    return None


def _number(
    value: Any,
    *,
    path: str,
    field_path: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}' must be a finite number",
            path=path,
            field_path=field_path,
        )
    number = float(value)
    if not math.isfinite(number):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}' must be finite",
            path=path,
            field_path=field_path,
        )
    if positive and number <= 0:
        raise _manifest_error(
            "INVALID_FIELD_VALUE",
            f"'{field_path}' must be greater than zero",
            path=path,
            field_path=field_path,
        )
    return number


def _integer(
    value: Any,
    *,
    path: str,
    field_path: str,
    positive: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}' must be an integer",
            path=path,
            field_path=field_path,
        )
    if positive and value <= 0:
        raise _manifest_error(
            "INVALID_FIELD_VALUE",
            f"'{field_path}' must be greater than zero",
            path=path,
            field_path=field_path,
        )
    return value


def _optional_integer(
    value: Any,
    *,
    path: str,
    field_path: str,
) -> int | None:
    if value is None:
        return None
    return _integer(value, path=path, field_path=field_path, positive=True)


def _resolve_audio_path(raw_path: str, manifest_dir: Path) -> str:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    return str(candidate.resolve())


def _parse_video(raw: dict[str, Any], manifest_path: str) -> VideoSpec:
    duration_field = _first_present(
        raw, ("duration", "duration_sec", "duration_seconds")
    )
    if duration_field is None:
        raise _manifest_error(
            "MISSING_FIELD",
            "required field 'video.duration' is missing",
            path=manifest_path,
            field_path="video.duration",
        )
    duration = _number(
        duration_field[1],
        path=manifest_path,
        field_path=f"video.{duration_field[0]}",
        positive=True,
    )
    fps_value = raw.get("fps")
    fps = (
        _number(fps_value, path=manifest_path, field_path="video.fps", positive=True)
        if fps_value is not None
        else None
    )
    width = _optional_integer(
        raw.get("width"), path=manifest_path, field_path="video.width"
    )
    height = _optional_integer(
        raw.get("height"), path=manifest_path, field_path="video.height"
    )
    raw_path = raw.get("path")
    video_path = None
    if raw_path is not None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise _manifest_error(
                "INVALID_FIELD_TYPE",
                "'video.path' must be a non-empty string",
                path=manifest_path,
                field_path="video.path",
            )
        video_path = _resolve_audio_path(raw_path, Path(manifest_path).parent)
    return VideoSpec(
        duration_sec=duration,
        fps=fps,
        width=width,
        height=height,
        path=video_path,
    )


def _parse_source_audio(raw: dict[str, Any], manifest_path: str) -> SourceAudioSpec:
    sample_field = _first_present(raw, ("sample_rate", "sample_rate_hz", "rate"))
    if sample_field is None:
        raise _manifest_error(
            "MISSING_FIELD",
            "required field 'dialogue_track.sample_rate' is missing",
            path=manifest_path,
            field_path="dialogue_track.sample_rate",
        )
    sample_rate = _integer(
        sample_field[1],
        path=manifest_path,
        field_path=f"dialogue_track.{sample_field[0]}",
        positive=True,
    )
    channels = _integer(
        _require(
            raw,
            "channels",
            path=manifest_path,
            field_path="dialogue_track.channels",
        ),
        path=manifest_path,
        field_path="dialogue_track.channels",
        positive=True,
    )
    path_value = _require(
        raw,
        "path",
        path=manifest_path,
        field_path="dialogue_track.path",
    )
    if not isinstance(path_value, str) or not path_value.strip():
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            "'dialogue_track.path' must be a non-empty string",
            path=manifest_path,
            field_path="dialogue_track.path",
        )
    validate_targets = raw.get("validate_targets", True)
    if not isinstance(validate_targets, bool):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            "'dialogue_track.validate_targets' must be a boolean",
            path=manifest_path,
            field_path="dialogue_track.validate_targets",
        )
    return SourceAudioSpec(
        sample_rate=sample_rate,
        channels=channels,
        path=_resolve_audio_path(path_value, Path(manifest_path).parent),
        validate_targets=validate_targets,
    )


def _parse_text(
    segment: dict[str, Any],
    *,
    target_language: str,
    manifest_path: str,
    field_path: str,
) -> tuple[str | dict[str, Any], str | None]:
    text_field = _first_present(segment, ("text", "subtitle", "subtitles"))
    if text_field is None:
        raise _manifest_error(
            "MISSING_FIELD",
            f"required field '{field_path}.text' is missing",
            path=manifest_path,
            field_path=f"{field_path}.text",
        )
    text = text_field[1]
    if isinstance(text, str):
        return text, text
    if not isinstance(text, dict):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}.{text_field[0]}' must be a string or language map",
            path=manifest_path,
            field_path=f"{field_path}.{text_field[0]}",
        )
    language_text = text.get(target_language)
    if language_text is not None and not isinstance(language_text, str):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}.{text_field[0]}.{target_language}' must be a string",
            path=manifest_path,
            field_path=f"{field_path}.{text_field[0]}.{target_language}",
        )
    if language_text is None:
        raise _manifest_error(
            "MISSING_FIELD",
            (
                f"required subtitle '{field_path}.{text_field[0]}."
                f"{target_language}' is missing"
            ),
            path=manifest_path,
            field_path=f"{field_path}.{text_field[0]}.{target_language}",
        )
    return text, language_text


def _parse_target(
    value: Any,
    *,
    segment: dict[str, Any],
    target_language: str,
    source_start: float,
    source_end: float,
    source_audio: SourceAudioSpec,
    manifest_path: str,
    field_path: str,
) -> TargetSegment:
    if isinstance(value, str):
        target: dict[str, Any] = {"path": value}
    elif isinstance(value, dict):
        target = value
    else:
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}' must be an audio path or an object",
            path=manifest_path,
            field_path=field_path,
        )

    path_field = _first_present(target, ("path", "audio_path", "file"))
    if path_field is None:
        raise _manifest_error(
            "MISSING_FIELD",
            f"required field '{field_path}.path' is missing",
            path=manifest_path,
            field_path=f"{field_path}.path",
        )
    audio_path = path_field[1]
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            f"'{field_path}.path' must be a non-empty string",
            path=manifest_path,
            field_path=f"{field_path}.path",
        )

    start_field = _first_present(target, ("start", "target_start", "start_sec"))
    end_field = _first_present(target, ("end", "target_end", "end_sec"))
    if (start_field is None) != (end_field is None):
        missing = "end" if end_field is None else "start"
        raise _manifest_error(
            "MISSING_FIELD",
            f"required field '{field_path}.{missing}' is missing",
            path=manifest_path,
            field_path=f"{field_path}.{missing}",
        )
    if start_field is None:
        target_start = source_start
        target_end = source_end
    else:
        target_start = _number(
            start_field[1],
            path=manifest_path,
            field_path=f"{field_path}.{start_field[0]}",
        )
        target_end = _number(
            end_field[1],
            path=manifest_path,
            field_path=f"{field_path}.{end_field[0]}",
        )
    if target_end <= target_start:
        raise _manifest_error(
            "INVALID_TIME_RANGE",
            f"'{field_path}' target end must be greater than target start",
            path=manifest_path,
            field_path=field_path,
            details={"start_sec": target_start, "end_sec": target_end},
        )

    expected_rate_field = _first_present(
        target, ("sample_rate", "sample_rate_hz", "expected_sample_rate")
    )
    if expected_rate_field is None:
        expected_rate = source_audio.sample_rate
        if not source_audio.validate_targets:
            expected_rate = 0
    else:
        expected_rate = _integer(
            expected_rate_field[1],
            path=manifest_path,
            field_path=f"{field_path}.{expected_rate_field[0]}",
            positive=True,
        )

    expected_channels_field = _first_present(
        target, ("channels", "expected_channels")
    )
    if expected_channels_field is None:
        expected_channels = source_audio.channels
        if not source_audio.validate_targets:
            expected_channels = 0
    else:
        expected_channels = _integer(
            expected_channels_field[1],
            path=manifest_path,
            field_path=f"{field_path}.{expected_channels_field[0]}",
            positive=True,
        )

    text, target_text = _parse_text(
        segment,
        target_language=target_language,
        manifest_path=manifest_path,
        field_path=field_path.rsplit(".targets", 1)[0],
    )
    return TargetSegment(
        segment_id="",
        source_start_sec=source_start,
        source_end_sec=source_end,
        target_start_sec=target_start,
        target_end_sec=target_end,
        text=text,
        target_text=target_text,
        audio_path=_resolve_audio_path(
            audio_path, Path(manifest_path).parent
        ),
        expected_sample_rate=expected_rate,
        expected_channels=expected_channels,
    )


def load_manifest(path: str | Path, target_language: str) -> Manifest:
    manifest_path = str(Path(path).resolve())
    try:
        raw_text = Path(manifest_path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BusinessError(
            code="MANIFEST_NOT_FOUND",
            message=f"manifest file not found: {manifest_path}",
            exit_code=EXIT_MANIFEST,
            file=manifest_path,
        ) from exc
    except OSError as exc:
        raise BusinessError(
            code="MANIFEST_READ_FAILED",
            message=f"could not read manifest: {manifest_path}: {exc}",
            exit_code=EXIT_MANIFEST,
            file=manifest_path,
        ) from exc

    try:
        root = json.loads(raw_text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _manifest_error(
            "INVALID_MANIFEST_JSON",
            f"manifest is not valid JSON: {exc}",
            path=manifest_path,
        ) from exc
    root = _as_object(root, path=manifest_path, field_path="$")

    video_value = _require(
        root, "video", path=manifest_path, field_path="video"
    )
    video = _parse_video(
        _as_object(video_value, path=manifest_path, field_path="video"),
        manifest_path,
    )

    source_field = _first_present(root, ("dialogue_track", "source_audio"))
    if source_field is None:
        raise _manifest_error(
            "MISSING_FIELD",
            "required field 'dialogue_track' is missing",
            path=manifest_path,
            field_path="dialogue_track",
        )
    source_audio = _parse_source_audio(
        _as_object(
            source_field[1],
            path=manifest_path,
            field_path=source_field[0],
        ),
        manifest_path,
    )

    segments_value = _require(
        root, "segments", path=manifest_path, field_path="segments"
    )
    if not isinstance(segments_value, list):
        raise _manifest_error(
            "INVALID_FIELD_TYPE",
            "'segments' must be a JSON array",
            path=manifest_path,
            field_path="segments",
        )
    if not segments_value:
        raise _manifest_error(
            "INVALID_FIELD_VALUE",
            "'segments' must contain at least one segment",
            path=manifest_path,
            field_path="segments",
        )

    segments: list[TargetSegment] = []
    seen_ids: set[str] = set()
    for index, raw_segment in enumerate(segments_value):
        segment_path = f"segments[{index}]"
        segment = _as_object(
            raw_segment, path=manifest_path, field_path=segment_path
        )
        id_key, raw_id = _first_present(segment, ("id", "segment_id")) or (
            None,
            None,
        )
        if raw_id is None:
            segment_id = f"segment_{index + 1:04d}"
        else:
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise _manifest_error(
                    "INVALID_FIELD_TYPE",
                    f"'{segment_path}.id' must be a non-empty string",
                    path=manifest_path,
                    field_path=f"{segment_path}.{id_key}",
                )
            segment_id = raw_id
        if segment_id in seen_ids:
            raise _manifest_error(
                "DUPLICATE_SEGMENT_ID",
                f"duplicate segment id '{segment_id}'",
                path=manifest_path,
                field_path=f"{segment_path}.id",
            )
        seen_ids.add(segment_id)

        start_field = _first_present(segment, ("start", "source_start", "start_sec"))
        end_field = _first_present(segment, ("end", "source_end", "end_sec"))
        if start_field is None:
            raise _manifest_error(
                "MISSING_FIELD",
                f"required field '{segment_path}.start' is missing",
                path=manifest_path,
                field_path=f"{segment_path}.start",
            )
        if end_field is None:
            raise _manifest_error(
                "MISSING_FIELD",
                f"required field '{segment_path}.end' is missing",
                path=manifest_path,
                field_path=f"{segment_path}.end",
            )
        source_start = _number(
            start_field[1],
            path=manifest_path,
            field_path=f"{segment_path}.{start_field[0]}",
        )
        source_end = _number(
            end_field[1],
            path=manifest_path,
            field_path=f"{segment_path}.{end_field[0]}",
        )
        if source_start < 0 or source_end <= source_start:
            raise _manifest_error(
                "INVALID_TIME_RANGE",
                f"'{segment_path}' has an invalid source time range",
                path=manifest_path,
                field_path=segment_path,
                details={"start_sec": source_start, "end_sec": source_end},
            )
        tolerance = 1.0 / max(source_audio.sample_rate, 1)
        if source_end > video.duration_sec + tolerance:
            raise _manifest_error(
                "INVALID_TIME_RANGE",
                f"'{segment_path}' source end exceeds video duration",
                path=manifest_path,
                field_path=segment_path,
                details={
                    "end_sec": source_end,
                    "video_duration_sec": video.duration_sec,
                },
            )

        targets_value = _require(
            segment,
            "targets",
            path=manifest_path,
            field_path=f"{segment_path}.targets",
        )
        targets = _as_object(
            targets_value,
            path=manifest_path,
            field_path=f"{segment_path}.targets",
        )
        if target_language not in targets:
            raise _manifest_error(
                "TARGET_LANGUAGE_NOT_FOUND",
                (
                    f"target language '{target_language}' is missing for "
                    f"segment '{segment_id}'"
                ),
                path=manifest_path,
                field_path=f"{segment_path}.targets.{target_language}",
                details={"available_languages": sorted(targets)},
            )
        target = _parse_target(
            targets[target_language],
            segment=segment,
            target_language=target_language,
            source_start=source_start,
            source_end=source_end,
            source_audio=source_audio,
            manifest_path=manifest_path,
            field_path=f"{segment_path}.targets.{target_language}",
        )
        segments.append(
            TargetSegment(
                segment_id=segment_id,
                source_start_sec=target.source_start_sec,
                source_end_sec=target.source_end_sec,
                target_start_sec=target.target_start_sec,
                target_end_sec=target.target_end_sec,
                text=target.text,
                target_text=target.target_text,
                audio_path=target.audio_path,
                expected_sample_rate=target.expected_sample_rate,
                expected_channels=target.expected_channels,
            )
        )

    return Manifest(
        path=manifest_path,
        video=video,
        source_audio=source_audio,
        segments=segments,
    )
