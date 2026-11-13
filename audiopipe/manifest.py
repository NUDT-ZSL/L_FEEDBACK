"""Manifest parsing and strict validation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .model import Job, Manifest


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_DEFAULT_MAX_ATTENUATION_DB = 24.0
_DEFAULT_MAX_BOOST_DB = 12.0


def _invalid(message: str, **details: Any) -> ValueError:
    error = ValueError(message)
    error.details = details  # type: ignore[attr-defined]
    return error


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(obj: dict[str, Any], name: str, *, minimum: float, maximum: float, required: bool = True,
            default: float | None = None) -> float:
    if name not in obj or obj[name] is None:
        if required:
            raise _invalid(f"missing required numeric field {name!r}", field=name)
        assert default is not None
        return float(default)
    value = obj[name]
    if not _is_number(value):
        raise _invalid(f"field {name!r} must be a finite number", field=name)
    result = float(value)
    if result < minimum or result > maximum:
        raise _invalid(
            f"field {name!r} must be between {minimum} and {maximum}",
            field=name,
            value=result,
        )
    return result


def _track(obj: dict[str, Any], role: str, aliases: Iterable[str], nested_aliases: Iterable[str] = ()) -> str:
    tracks = obj.get("tracks")
    candidates: list[tuple[str, Any]] = []
    track_key = next((name for name in (role, *nested_aliases) if isinstance(tracks, dict) and name in tracks), None)
    if track_key is not None:
        assert isinstance(tracks, dict)
        track_value = tracks[track_key]
        if isinstance(track_value, str):
            candidates.append((f"tracks.{track_key}", track_value))
        elif isinstance(track_value, dict) and "path" in track_value:
            candidates.append((f"tracks.{track_key}.path", track_value["path"]))
        else:
            raise _invalid(f"field tracks.{track_key} must be a path string or an object with path", field=role)
    for alias in aliases:
        if alias in obj:
            candidates.append((alias, obj[alias]))
    if not candidates:
        raise _invalid(f"missing required path for {role} track", field=role)
    selected = candidates[0]
    for _, value in candidates[1:]:
        if value != selected[1]:
            raise _invalid(f"conflicting path declarations for {role} track", field=role)
    value = selected[1]
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"track path {selected[0]!r} must be a non-empty string", field=selected[0])
    if "\x00" in value:
        raise _invalid(f"track path {selected[0]!r} contains NUL", field=selected[0])
    if value.strip() != value:
        raise _invalid(f"track path {selected[0]!r} has leading/trailing whitespace", field=selected[0])
    if Path(value).is_absolute() or value.startswith(("/", "\\")) or _WINDOWS_DRIVE.match(value):
        raise _invalid(f"track path {selected[0]!r} must be relative to the show directory", field=selected[0])
    return value


def _fade_values(obj: dict[str, Any]) -> tuple[float, float]:
    nested = obj.get("music_fade")
    start_names: list[str] = []
    duration_names: list[str] = []
    if isinstance(nested, dict):
        if "start_sec" in nested:
            start_names.append("music_fade.start_sec")
        if "duration_sec" in nested:
            duration_names.append("music_fade.duration_sec")
    start_names.extend(["music_fade_start_sec", "music_fade_start", "fade_start_sec", "fade_start"])
    duration_names.extend(
        ["music_fade_duration_sec", "music_fade_duration", "fade_duration_sec", "fade_duration"]
    )

    def pick(names: list[str], label: str) -> tuple[str, Any]:
        values: list[tuple[str, Any]] = []
        for name in names:
            if "." in name:
                assert isinstance(nested, dict)
                key = name.split(".", 1)[1]
                if key in nested:
                    values.append((name, nested[key]))
            elif name in obj:
                values.append((name, obj[name]))
        if not values:
            raise _invalid(f"missing required field for music fade {label}", field=label)
        selected = values[0]
        for _, value in values[1:]:
            if value != selected[1]:
                raise _invalid(f"conflicting music fade {label} declarations", field=label)
        return selected

    start_name, start_value = pick(start_names, "start")
    duration_name, duration_value = pick(duration_names, "duration")
    if not _is_number(start_value):
        raise _invalid(f"field {start_name!r} must be a finite number", field=start_name)
    if not _is_number(duration_value):
        raise _invalid(f"field {duration_name!r} must be a finite number", field=duration_name)
    start = float(start_value)
    duration = float(duration_value)
    if start < 0.0 or start > 86400.0:
        raise _invalid(f"field {start_name!r} must be between 0 and 86400 seconds", field=start_name)
    if duration <= 0.0 or duration > 86400.0:
        raise _invalid(f"field {duration_name!r} must be greater than 0 and at most 86400 seconds", field=duration_name)
    return start, duration


def parse_manifest(job: Job, raw: bytes | None = None) -> Manifest:
    if raw is None:
        try:
            raw = job.manifest_path.read_bytes()
        except OSError as exc:
            raise _invalid(f"cannot read manifest: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _invalid(f"manifest is not valid UTF-8: {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _invalid(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise _invalid("manifest root must be a JSON object")

    dialogue = _track(obj, "dialogue", ("dialogue_path", "dialogue"))
    music = _track(obj, "music", ("music_path", "music"))
    ambience = _track(
        obj,
        "ambience",
        ("ambience_path", "ambience", "environment_path", "environment"),
        ("environment",),
    )

    target_names = [
        name
        for name in ("target_lufs", "dialogue_target_lufs", "target_dialogue_lufs")
        if name in obj
    ]
    if not target_names:
        raise _invalid("missing required numeric field 'target_lufs'", field="target_lufs")
    target = _number(obj, target_names[0], minimum=-70.0, maximum=0.0)
    for alias_name in target_names[1:]:
        alias_target = _number(obj, alias_name, minimum=-70.0, maximum=0.0)
        if abs(alias_target - target) > 1e-9:
            raise _invalid(f"conflicting {target_names[0]} and {alias_name}", field=target_names[0])

    fade_start, fade_duration = _fade_values(obj)

    music_gain_names = [name for name in ("music_gain_db", "music_gain") if name in obj]
    if not music_gain_names:
        raise _invalid("missing required numeric field 'music_gain_db'", field="music_gain_db")
    music_gain = _number(obj, music_gain_names[0], minimum=-120.0, maximum=24.0)
    for alias_name in music_gain_names[1:]:
        alias_gain = _number(obj, alias_name, minimum=-120.0, maximum=24.0)
        if abs(alias_gain - music_gain) > 1e-9:
            raise _invalid(f"conflicting {music_gain_names[0]} and {alias_name}", field=alias_name)

    ambience_gain_name: str | None = None
    for name in ("ambience_gain_db", "environment_gain_db", "ambience_gain", "environment_gain"):
        if name in obj:
            if ambience_gain_name is not None:
                other = _number(obj, name, minimum=-120.0, maximum=24.0)
                current = _number(obj, ambience_gain_name, minimum=-120.0, maximum=24.0)
                if abs(other - current) > 1e-9:
                    raise _invalid(f"conflicting {ambience_gain_name} and {name}", field=name)
            else:
                ambience_gain_name = name
    if ambience_gain_name is None:
        raise _invalid("missing required numeric field 'ambience_gain_db'", field="ambience_gain_db")
    ambience_gain = _number(obj, ambience_gain_name, minimum=-120.0, maximum=24.0)

    attenuation_names = [name for name in ("max_attenuation_db", "max_attenuation") if name in obj]
    if attenuation_names:
        max_attenuation = _number(obj, attenuation_names[0], minimum=0.0, maximum=120.0)
        for alias_name in attenuation_names[1:]:
            alias_attenuation = _number(obj, alias_name, minimum=0.0, maximum=120.0)
            if abs(alias_attenuation - max_attenuation) > 1e-9:
                raise _invalid(f"conflicting {attenuation_names[0]} and {alias_name}", field=alias_name)
    else:
        max_attenuation = _DEFAULT_MAX_ATTENUATION_DB

    boost_names = [name for name in ("max_boost_db", "max_boost") if name in obj]
    if boost_names:
        max_boost = _number(obj, boost_names[0], minimum=0.0, maximum=48.0)
        for alias_name in boost_names[1:]:
            alias_boost = _number(obj, alias_name, minimum=0.0, maximum=48.0)
            if abs(alias_boost - max_boost) > 1e-9:
                raise _invalid(f"conflicting {boost_names[0]} and {alias_name}", field=alias_name)
    else:
        max_boost = _DEFAULT_MAX_BOOST_DB

    for role, path in (("dialogue", dialogue), ("music", music), ("ambience", ambience)):
        resolved = (job.job_dir / Path(path.replace("\\", "/"))).resolve()
        try:
            resolved.relative_to(job.job_dir.resolve())
        except ValueError:
            raise _invalid(f"{role} track path escapes the show directory", field=f"{role}_path")

    return Manifest(
        dialogue_path=dialogue,
        music_path=music,
        ambience_path=ambience,
        target_lufs=target,
        music_fade_start_sec=fade_start,
        music_fade_duration_sec=fade_duration,
        music_gain_db=music_gain,
        ambience_gain_db=ambience_gain,
        max_attenuation_db=max_attenuation,
        max_boost_db=max_boost,
    )


def manifest_error_details(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    return details if isinstance(details, dict) else {}
