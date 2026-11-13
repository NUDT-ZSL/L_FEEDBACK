from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localizer.errors import EXIT_TIMELINE, LocalizerError
from localizer.languages import normalize_language_code


MAX_CROSSFADE_MS = 10
CONTINUITY_GAP_MS = 5
SUPPORTED_BIT_DEPTHS = {8, 16, 24, 32}
SUPPORTED_SAMPLE_RATES = {
    8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100,
    48000, 64000, 88200, 96000, 176400, 192000,
}


@dataclass(frozen=True)
class TargetFormat:
    sample_rate: int = 48000
    channels: int = 2
    bit_depth: int = 16


@dataclass(frozen=True)
class Clip:
    clip_id: str
    language: str
    start_ms: int
    duration_ms: int
    path: Path
    text: str

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class LanguageTrack:
    code: str
    name: str
    clips: tuple[Clip, ...]


@dataclass(frozen=True)
class Manifest:
    path: Path
    total_ms: int
    target: TargetFormat
    output_dir: Path
    output_name: str
    languages: tuple[LanguageTrack, ...]


def _timeline_error(message: str, *, field: str = "", **details: Any) -> None:
    payload: dict[str, Any] = {}
    if field:
        payload["field"] = field
    payload.update(details)
    raise LocalizerError(
        "E_TIMELINE",
        message,
        exit_code=EXIT_TIMELINE,
        details=payload,
    )


def _get_first(obj: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in obj:
            return obj[name]
    return default


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _timeline_error(f"{field} must be a JSON object", field=field)
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _timeline_error(f"{field} must be an integer number of milliseconds", field=field)
    return value


def _require_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _timeline_error(f"{field} must be a string", field=field)
    if not allow_empty and not value:
        _timeline_error(f"{field} must not be empty", field=field)
    return value


def _resolve_path(base: Path, raw: Any, field: str) -> Path:
    value = _require_text(raw, field)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_output_name(value: Any) -> str:
    name = _require_text(value, "output.name")
    forbidden = '<>:"/\\|?*'
    if (
        name != Path(name).name
        or name.startswith(".")
        or any(character in forbidden for character in name)
        or any(ord(character) < 32 for character in name)
    ):
        _timeline_error(
            "output.name must be a safe filename without path components",
            field="output.name",
        )
    return name


def _parse_target(data: dict[str, Any]) -> TargetFormat:
    target_raw = _require_object(data.get("target", {}), "target")
    sample_rate = _require_int(
        _get_first(target_raw, ("sample_rate", "sample_rate_hz"), 48000),
        "target.sample_rate",
    )
    channels = _require_int(
        _get_first(target_raw, ("channels", "channel_count"), 2),
        "target.channels",
    )
    bit_depth = _require_int(
        _get_first(target_raw, ("bit_depth", "bits_per_sample"), 16),
        "target.bit_depth",
    )
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        _timeline_error(
            f"unsupported target sample rate: {sample_rate}",
            field="target.sample_rate",
        )
    if channels not in {1, 2}:
        _timeline_error(
            "target.channels must be 1 (mono) or 2 (stereo)",
            field="target.channels",
        )
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        _timeline_error(
            "target.bit_depth must be one of 8, 16, 24, 32",
            field="target.bit_depth",
        )
    return TargetFormat(sample_rate=sample_rate, channels=channels, bit_depth=bit_depth)


def _parse_output(data: dict[str, Any], manifest_dir: Path) -> tuple[Path, str]:
    raw_output = data.get("output", "out")
    output_name = data.get("output_name", "cutscene")
    if isinstance(raw_output, str):
        output_dir_raw = raw_output
    elif isinstance(raw_output, dict):
        output_dir_raw = _get_first(
            raw_output,
            ("directory", "dir", "path"),
            data.get("output_dir", "out"),
        )
        output_name = _get_first(raw_output, ("name", "filename"), output_name)
    else:
        _timeline_error("output must be a path string or JSON object", field="output")

    output_dir_raw = data.get("output_dir", output_dir_raw)
    output_dir = _resolve_path(manifest_dir, output_dir_raw, "output")
    return output_dir, _validate_output_name(output_name)


def _parse_flat_clips(
    raw_clips: list[Any],
) -> tuple[list[tuple[str, str | None, tuple[Any, ...]]], dict[str, str]]:
    grouped: dict[str, list[Any]] = {}
    names: dict[str, str] = {}
    for index, raw_clip in enumerate(raw_clips):
        clip_obj = _require_object(raw_clip, f"clips[{index}]")
        language_raw = _get_first(clip_obj, ("language", "lang", "code"))
        if language_raw is None:
            _timeline_error(
                "flat clips entries must declare language",
                field=f"clips[{index}].language",
            )
        try:
            code = normalize_language_code(language_raw)
        except ValueError as exc:
            _timeline_error(
                str(exc),
                field=f"clips[{index}].language",
            )
        grouped.setdefault(code, []).append(clip_obj)
        name = _get_first(clip_obj, ("language_name", "name"))
        if isinstance(name, str) and name:
            names.setdefault(code, name)
    entries = [(code, names.get(code), tuple(items)) for code, items in grouped.items()]
    return entries, names


def _parse_language_sections(
    raw_languages: Any,
) -> list[tuple[str, str | None, tuple[Any, ...]]]:
    entries: list[tuple[str, str | None, tuple[Any, ...]]] = []
    if isinstance(raw_languages, dict):
        iterable = [
            {"code": code, **(value if isinstance(value, dict) else {"clips": value})}
            for code, value in raw_languages.items()
        ]
    elif isinstance(raw_languages, list):
        iterable = raw_languages
    else:
        _timeline_error("languages must be an array or object", field="languages")

    for index, raw_track in enumerate(iterable):
        track_obj = _require_object(raw_track, f"languages[{index}]")
        language_raw = _get_first(track_obj, ("language", "lang", "code"))
        if language_raw is None:
            _timeline_error(
                "language track must declare language/code",
                field=f"languages[{index}].language",
            )
        try:
            code = normalize_language_code(language_raw)
        except ValueError as exc:
            _timeline_error(str(exc), field=f"languages[{index}].language")
        name_raw = _get_first(track_obj, ("name", "language_name"))
        name = str(name_raw) if isinstance(name_raw, str) and name_raw else None
        clips_raw = track_obj.get("clips")
        if not isinstance(clips_raw, list) or not clips_raw:
            _timeline_error(
                "language track must contain a non-empty clips array",
                field=f"languages[{index}].clips",
            )
        entries.append((code, name, tuple(clips_raw)))
    return entries


def _build_track(
    code: str,
    name: str | None,
    raw_clips: tuple[Any, ...],
    manifest_dir: Path,
) -> LanguageTrack:
    clips: list[Clip] = []
    identifiers: set[str] = set()
    for local_index, raw_clip in enumerate(raw_clips):
        field_base = f"languages[{code}].clips[{local_index}]"
        clip_obj = _require_object(raw_clip, field_base)
        declared_language = _get_first(clip_obj, ("language", "lang", "code"))
        if declared_language is not None:
            try:
                declared_code = normalize_language_code(declared_language)
            except ValueError as exc:
                _timeline_error(str(exc), field=f"{field_base}.language")
            if declared_code != code:
                _timeline_error(
                    "clip language does not match its language track",
                    field=f"{field_base}.language",
                    expected=code,
                    actual=declared_code,
                )

        start_ms = _require_int(
            _get_first(clip_obj, ("start_ms", "start")),
            f"{field_base}.start_ms",
        )
        duration_ms = _require_int(
            _get_first(clip_obj, ("duration_ms", "duration")),
            f"{field_base}.duration_ms",
        )
        if start_ms < 0:
            _timeline_error("clip start_ms must be >= 0", field=f"{field_base}.start_ms")
        if duration_ms <= 0:
            _timeline_error(
                "clip duration_ms must be a positive integer",
                field=f"{field_base}.duration_ms",
            )
        path_raw = _get_first(clip_obj, ("path", "file", "audio"))
        if path_raw is None:
            _timeline_error(f"{field_base}.path is required")
        path = _resolve_path(manifest_dir, path_raw, f"{field_base}.path")
        text_raw = _get_first(clip_obj, ("text", "subtitle", "subtitle_text"))
        if text_raw is None:
            _timeline_error(f"{field_base}.text is required")
        text = _require_text(text_raw, f"{field_base}.text", allow_empty=True)
        clip_id_raw = clip_obj.get("id")
        clip_id = (
            _require_text(clip_id_raw, f"{field_base}.id")
            if clip_id_raw is not None
            else f"clip-{len(clips) + 1:04d}"
        )
        if clip_id in identifiers:
            _timeline_error(
                f"duplicate clip id in language {code}: {clip_id}",
                field=f"{field_base}.id",
            )
        identifiers.add(clip_id)
        clips.append(
            Clip(
                clip_id=clip_id,
                language=code,
                start_ms=start_ms,
                duration_ms=duration_ms,
                path=path,
                text=text,
            )
        )

    ordered = sorted(clips, key=lambda clip: (clip.start_ms, clip.end_ms, clip.clip_id))
    previous: Clip | None = None
    for clip in ordered:
        if previous is not None:
            overlap_ms = previous.end_ms - clip.start_ms
            if overlap_ms > MAX_CROSSFADE_MS:
                _timeline_error(
                    "clips overlap by more than the permitted 10ms crossfade",
                    field=f"languages[{code}]",
                    previous_id=previous.clip_id,
                    current_id=clip.clip_id,
                    overlap_ms=overlap_ms,
                )
            if overlap_ms < 0 and -overlap_ms <= CONTINUITY_GAP_MS:
                pass
        previous = clip

    return LanguageTrack(code=code, name=name or code, clips=tuple(ordered))


def load_manifest(path: Path) -> Manifest:
    resolved = path.expanduser().resolve()
    try:
        raw_text = resolved.read_text(encoding="utf-8-sig")
    except OSError as exc:
        _timeline_error(f"cannot read manifest: {exc}", field="manifest")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _timeline_error(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            field="manifest",
        )
    data = _require_object(data, "manifest")
    manifest_dir = resolved.parent

    target = _parse_target(data)
    output_dir, output_name = _parse_output(data, manifest_dir)

    if "languages" in data or "tracks" in data:
        raw_languages = data.get("languages", data.get("tracks"))
        language_entries = _parse_language_sections(raw_languages)
    elif "clips" in data:
        language_entries, _ = _parse_flat_clips(
            _require_list(data.get("clips"), "clips", non_empty=True)
        )
    else:
        _timeline_error("manifest must declare languages/tracks or clips", field="manifest")

    if not language_entries:
        _timeline_error("manifest contains no language tracks", field="languages")

    tracks: list[LanguageTrack] = []
    seen_codes: set[str] = set()
    for code, name, raw_clips in language_entries:
        if code in seen_codes:
            _timeline_error(
                f"duplicate language after ISO 639-2/B normalization: {code}",
                field="languages",
            )
        seen_codes.add(code)
        tracks.append(_build_track(code, name, raw_clips, manifest_dir))

    max_end_ms = max(clip.end_ms for track in tracks for clip in track.clips)
    timeline_raw = data.get("timeline")
    total_raw: Any = None
    if isinstance(timeline_raw, dict):
        total_raw = _get_first(
            timeline_raw,
            ("duration_ms", "total_ms", "length_ms", "duration", "total_duration"),
        )
    elif isinstance(timeline_raw, int) and not isinstance(timeline_raw, bool):
        total_raw = timeline_raw
    elif timeline_raw is not None:
        _timeline_error("timeline must be an object or integer duration", field="timeline")
    if total_raw is None:
        total_raw = _get_first(
            data,
            (
                "duration_ms",
                "total_duration_ms",
                "timeline_duration_ms",
                "total_duration",
                "timeline_duration",
                "duration",
            ),
            max_end_ms,
        )
    total_ms = _require_int(total_raw, "timeline.duration_ms")
    if total_ms <= 0:
        _timeline_error("timeline duration must be positive", field="timeline.duration_ms")
    if total_ms < max_end_ms:
        _timeline_error(
            "timeline duration is shorter than a clip end time",
            field="timeline.duration_ms",
            duration_ms=total_ms,
            clip_end_ms=max_end_ms,
        )

    return Manifest(
        path=resolved,
        total_ms=total_ms,
        target=target,
        output_dir=output_dir,
        output_name=output_name,
        languages=tuple(tracks),
    )


def _require_list(value: Any, field: str, *, non_empty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _timeline_error(f"{field} must be an array", field=field)
    if non_empty and not value:
        _timeline_error(f"{field} must not be empty", field=field)
    return value
