"""JSON playback-configuration parsing and validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .errors import E_INVALID_INPUT, MixerError

_MISSING = object()
_MAX_TRACKS = 8
_VALID_CURVES = {
    "equal_power": "equal_power",
    "equal-power": "equal_power",
    "equalpower": "equal_power",
    "linear": "linear",
    "linear_fade": "linear",
    "linear-fade": "linear",
}


@dataclass(frozen=True)
class TrackSettings:
    path: str
    volume: float
    offset_seconds: float
    loop_start_seconds: Optional[float]
    loop_end_seconds: Optional[float]
    fade_in_ms: float
    fade_out_ms: float
    fade_curve: str


@dataclass(frozen=True)
class MixerSettings:
    duration_seconds: Optional[float]
    tracks: Tuple[TrackSettings, ...]


def _invalid(message: str) -> MixerError:
    return MixerError(E_INVALID_INPUT, message)


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return _MISSING


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise _invalid(f"{label} must be finite")
    return result


def _optional_number(mapping: Mapping[str, Any], names: Sequence[str], label: str) -> Optional[float]:
    value = _first(mapping, names)
    if value is _MISSING or value is None:
        return None
    return _number(value, label)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{label} must be a JSON object")
    return value


def _duration_from_config(root: Mapping[str, Any]) -> Optional[float]:
    seconds = _optional_number(
        root,
        (
            "duration_seconds",
            "play_duration_seconds",
            "playback_duration_seconds",
            "playback_seconds",
            "total_duration_seconds",
            "play_time_seconds",
            "duration",
            "play_duration",
        ),
        "duration_seconds",
    )
    milliseconds = _optional_number(root, ("duration_ms", "play_duration_ms"), "duration_ms")
    if seconds is not None and milliseconds is not None:
        raise _invalid("specify duration in either seconds or milliseconds, not both")
    duration = seconds if seconds is not None else (milliseconds / 1000.0 if milliseconds is not None else None)
    if duration is not None and duration <= 0.0:
        raise _invalid("duration_seconds must be greater than zero")
    return duration


def _loop_values(container: Mapping[str, Any], label: str) -> Tuple[Optional[float], Optional[float]]:
    start: Optional[float] = None
    end: Optional[float] = None
    nested = container.get("loop", _MISSING)
    if nested is not _MISSING and nested is not None:
        if isinstance(nested, (list, tuple)):
            if len(nested) != 2:
                raise _invalid(f"{label}.loop must contain [start_seconds, end_seconds]")
            start = _number(nested[0], f"{label}.loop[0]")
            end = _number(nested[1], f"{label}.loop[1]")
        else:
            loop_map = _mapping(nested, f"{label}.loop")
            start = _optional_number(
                loop_map,
                (
                    "start_seconds",
                    "start",
                    "loop_start_seconds",
                    "loop_start_sec",
                    "loop_start_time_seconds",
                ),
                f"{label}.loop.start_seconds",
            )
            end = _optional_number(
                loop_map,
                (
                    "end_seconds",
                    "end",
                    "loop_end_seconds",
                    "loop_end_sec",
                    "loop_end_time_seconds",
                ),
                f"{label}.loop.end_seconds",
            )

    flat_start = _optional_number(
        container,
        (
            "loop_start_seconds",
            "loop_start_sec",
            "loop_start",
            "loop_start_time",
            "loop_start_time_seconds",
        ),
        f"{label}.loop_start_seconds",
    )
    flat_end = _optional_number(
        container,
        (
            "loop_end_seconds",
            "loop_end_sec",
            "loop_end",
            "loop_end_time",
            "loop_end_time_seconds",
        ),
        f"{label}.loop_end_seconds",
    )
    if flat_start is not None:
        start = flat_start
    if flat_end is not None:
        end = flat_end

    if start is not None and start < 0.0:
        raise _invalid(f"{label} loop start must not be negative")
    if end is not None and end <= 0.0:
        raise _invalid(f"{label} loop end must be greater than zero")
    if start is not None and end is not None and end <= start:
        raise _invalid(f"{label} loop end must be greater than loop start")
    return start, end


def _fade_values(
    container: Mapping[str, Any],
    label: str,
    fallback_in: float,
    fallback_out: float,
    fallback_curve: str,
) -> Tuple[float, float, str]:
    fade_in = fallback_in
    fade_out = fallback_out
    curve = fallback_curve

    nested = container.get("fade", _MISSING)
    if nested is not _MISSING and nested is not None:
        fade_map = _mapping(nested, f"{label}.fade")
        value = _optional_number(
            fade_map,
            ("duration_ms", "fade_duration_ms"),
            f"{label}.fade.duration_ms",
        )
        if value is not None:
            fade_in = value
            fade_out = value
        value = _optional_number(
            fade_map,
            ("in_ms", "fade_in_ms", "fade_in_milliseconds", "in"),
            f"{label}.fade.in_ms",
        )
        if value is not None:
            fade_in = value
        value = _optional_number(
            fade_map,
            ("out_ms", "fade_out_ms", "fade_out_milliseconds", "out"),
            f"{label}.fade.out_ms",
        )
        if value is not None:
            fade_out = value
        value = _first(fade_map, ("curve", "fade_curve", "kind", "type", "fade_type"))
        if value is not _MISSING and value is not None:
            curve = _text(value, f"{label}.fade.curve")

    common = _optional_number(container, ("fade_ms", "fade_duration_ms"), f"{label}.fade_ms")
    if common is not None:
        fade_in = common
        fade_out = common
    value = _optional_number(
        container,
        ("fade_in_ms", "fade_in_milliseconds", "fade_in"),
        f"{label}.fade_in_ms",
    )
    if value is not None:
        fade_in = value
    value = _optional_number(
        container,
        ("fade_out_ms", "fade_out_milliseconds", "fade_out"),
        f"{label}.fade_out_ms",
    )
    if value is not None:
        fade_out = value
    value = _first(container, ("fade_curve", "fade_kind", "fade_type"))
    if value is not _MISSING and value is not None:
        curve = _text(value, f"{label}.fade_curve")

    if fade_in < 0.0 or fade_out < 0.0:
        raise _invalid(f"{label} fade durations must not be negative")
    curve_key = curve.strip().lower().replace(" ", "_")
    normalized = _VALID_CURVES.get(curve_key)
    if normalized is None:
        raise _invalid(f"{label} fade_curve must be 'equal_power' or 'linear'")
    return fade_in, fade_out, normalized


def _find_track_list(root: Mapping[str, Any]) -> Any:
    for name in ("tracks", "audio_tracks", "mix_tracks"):
        if name in root:
            return root[name]
    return _MISSING


def _resolve_path(value: Any, base_dir: str, label: str) -> str:
    path_text = _text(value, label)
    expanded = os.path.expanduser(path_text)
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_dir, expanded)
    return os.path.abspath(expanded)


def load_settings(config_path: str, input_path: str) -> MixerSettings:
    """Load and validate a playback configuration.

    Relative track paths are resolved against the configuration file's
    directory.  The CLI ``--input`` path is resolved against the current
    working directory by the caller before it reaches this function.
    """

    try:
        with open(config_path, "r", encoding="utf-8-sig") as handle:
            root = json.load(handle)
    except FileNotFoundError as exc:
        raise MixerError(E_INVALID_INPUT, f"configuration file not found: {exc}") from exc
    except PermissionError as exc:
        raise MixerError(E_INVALID_INPUT, f"configuration file is not readable: {exc}") from exc
    except OSError as exc:
        raise MixerError(E_INVALID_INPUT, f"cannot read configuration file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _invalid(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    root = _mapping(root, "configuration")
    duration = _duration_from_config(root)
    global_start, global_end = _loop_values(root, "configuration")
    global_fade_in, global_fade_out, global_curve = _fade_values(
        root, "configuration", 10.0, 10.0, "equal_power"
    )

    raw_tracks = _find_track_list(root)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if raw_tracks is _MISSING:
        raw_track_items: List[Any] = [{}]
    else:
        if isinstance(raw_tracks, dict):
            raw_track_items = [raw_tracks]
        elif isinstance(raw_tracks, list):
            raw_track_items = list(raw_tracks)
        else:
            raise _invalid("tracks must be a JSON array")
        if not raw_track_items:
            raw_track_items = [{}]

    if len(raw_track_items) > _MAX_TRACKS:
        raise _invalid(f"at most {_MAX_TRACKS} tracks are supported")

    tracks: List[TrackSettings] = []
    for index, raw_track in enumerate(raw_track_items):
        label = f"tracks[{index}]"
        track_map = _mapping(raw_track, label)

        path_value = _first(
            track_map,
            ("path", "file", "input", "wav", "source_file", "audio_file", "filename"),
        )
        if path_value is _MISSING or path_value is None:
            track_path = os.path.abspath(input_path)
        else:
            track_path = _resolve_path(path_value, config_dir, f"{label}.path")

        volume_value = _first(
            track_map, ("volume", "gain", "linear_volume", "linear_gain")
        )
        if volume_value is _MISSING or volume_value is None:
            volume = 1.0
        else:
            volume = _number(volume_value, f"{label}.volume")
            if volume < 0.0 or volume > 1.0:
                raise _invalid(f"{label}.volume must be between 0.0 and 1.0")

        offset = _optional_number(
            track_map,
            (
                "offset_seconds",
                "offset",
                "start_offset_seconds",
                "start_offset",
                "offset_time_seconds",
                "start_seconds",
            ),
            f"{label}.offset_seconds",
        )
        offset_ms = _optional_number(
            track_map, ("offset_ms", "start_offset_ms"), f"{label}.offset_ms"
        )
        if offset is not None and offset_ms is not None:
            raise _invalid(f"{label} must not specify both offset_seconds and offset_ms")
        if offset is None:
            offset = offset_ms / 1000.0 if offset_ms is not None else 0.0
        if offset < 0.0:
            raise _invalid(f"{label}.offset_seconds must not be negative")

        track_start, track_end = _loop_values(track_map, label)
        loop_start = track_start if track_start is not None else global_start
        loop_end = track_end if track_end is not None else global_end
        if loop_start is not None and loop_end is not None and loop_end <= loop_start:
            raise _invalid(f"{label} loop end must be greater than loop start")

        fade_in, fade_out, curve = _fade_values(
            track_map,
            label,
            global_fade_in,
            global_fade_out,
            global_curve,
        )

        tracks.append(
            TrackSettings(
                path=track_path,
                volume=volume,
                offset_seconds=offset,
                loop_start_seconds=loop_start,
                loop_end_seconds=loop_end,
                fade_in_ms=fade_in,
                fade_out_ms=fade_out,
                fade_curve=curve,
            )
        )

    return MixerSettings(duration_seconds=duration, tracks=tuple(tracks))
