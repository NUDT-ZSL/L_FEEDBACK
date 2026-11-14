"""Parsing and validation for shot descriptions and choice files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ErrorCode, MixerError

SUPPORTED_SAMPLE_RATES = frozenset((44_100, 48_000))
TRACK_TYPES = ("dialogue", "ambience", "music")
DEFAULT_BUS_WEIGHTS = {
    "dialogue": 1.0,
    "ambience": 0.7,
    "music": 0.5,
}
SUPPORTED_SUBTYPES = frozenset(("PCM_16", "PCM_24", "FLOAT"))

_TRACK_ALIASES = {
    "dialogue": "dialogue",
    "dialog": "dialogue",
    "voice": "dialogue",
    "speech": "dialogue",
    "对白": "dialogue",
    "ambience": "ambience",
    "ambient": "ambience",
    "environment": "ambience",
    "environment_sfx": "ambience",
    "sfx": "ambience",
    "音效": "ambience",
    "环境": "ambience",
    "环境音效": "ambience",
    "music": "music",
    "bgm": "music",
    "score": "music",
    "音乐": "music",
}


@dataclass(frozen=True)
class TrackSpec:
    kind: str
    path: Path
    gain_db: float
    fade_in_ms: Decimal
    fade_out_ms: Decimal
    fade_in_samples: int
    fade_out_samples: int


@dataclass(frozen=True)
class ShotSpec:
    shot_id: str
    start_ms: Decimal
    end_ms: Decimal
    start_sample: int
    end_sample: int
    duration_samples: int
    tracks: tuple[TrackSpec, ...]
    allow_overlap: bool
    crossfade_ms: Decimal


@dataclass(frozen=True)
class ProjectConfig:
    source: Path
    target_sample_rate: int
    output_channels: int
    output_subtype: str
    bus_weights: Mapping[str, float]
    limiter_ceiling_dbfs: float
    shots: tuple[ShotSpec, ...]
    shots_by_id: Mapping[str, ShotSpec]


def _business(code: ErrorCode, message: str) -> MixerError:
    return MixerError(code, message)


def _read_json(path: Path, code: ErrorCode, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _business(code, f"cannot read {label} {path}: {exc}") from exc


def _number(value: Any, field: str, code: ErrorCode) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise _business(code, f"{field} must be a finite number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise _business(code, f"{field} must be a finite number") from exc
    if not number.is_finite():
        raise _business(code, f"{field} must be a finite number")
    return number


def _float_number(value: Any, field: str, code: ErrorCode) -> float:
    number = _number(value, field, code)
    try:
        result = float(number)
    except (OverflowError, ValueError) as exc:
        raise _business(code, f"{field} is outside the supported range") from exc
    if not math.isfinite(result):
        raise _business(code, f"{field} must be finite")
    return result


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    raise _business(ErrorCode.E_SHOT_DESC, f"{field} must be a boolean")


def ms_to_samples(value: Decimal | int | float | str, sample_rate: int) -> int:
    """Convert milliseconds to samples using deterministic half-up rounding."""

    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    samples = decimal_value * Decimal(sample_rate) / Decimal(1000)
    return int(samples.to_integral_value(rounding=ROUND_HALF_UP))


def _normalise_track_type(value: Any) -> str:
    if not isinstance(value, str):
        raise _business(ErrorCode.E_SHOT_DESC, "track type must be a string")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _TRACK_ALIASES[key]
    except KeyError as exc:
        raise _business(
            ErrorCode.E_SHOT_DESC,
            f"unsupported track type {value!r}; expected dialogue, ambience, or music",
        ) from exc


def _track_path(raw_path: Any, field: str, base_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise _business(ErrorCode.E_SHOT_DESC, f"{field} must be a non-empty path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _gain_db(track: Mapping[str, Any], field: str) -> float:
    if "gain_db" in track:
        return _float_number(track["gain_db"], f"{field}.gain_db", ErrorCode.E_SHOT_DESC)
    if "gain" in track:
        gain = _float_number(track["gain"], f"{field}.gain", ErrorCode.E_SHOT_DESC)
        if gain < 0:
            raise _business(ErrorCode.E_SHOT_DESC, f"{field}.gain must be non-negative")
        return 20.0 * math.log10(gain) if gain > 0 else -math.inf
    if "volume" in track:
        gain = _float_number(track["volume"], f"{field}.volume", ErrorCode.E_SHOT_DESC)
        if gain < 0:
            raise _business(ErrorCode.E_SHOT_DESC, f"{field}.volume must be non-negative")
        return 20.0 * math.log10(gain) if gain > 0 else -math.inf
    return 0.0


def _fade_value(
    track: Mapping[str, Any],
    shot: Mapping[str, Any],
    track_keys: Iterable[str],
    shot_keys: Iterable[str],
    field: str,
) -> Decimal:
    raw = _first(track, *track_keys)
    if raw is None:
        raw = _first(shot, *shot_keys)
    if raw is None:
        return Decimal(0)
    value = _number(raw, field, ErrorCode.E_FADE_OVERFLOW)
    if value < 0:
        raise _business(ErrorCode.E_FADE_OVERFLOW, f"{field} cannot be negative")
    return value


def _parse_track(
    track: Mapping[str, Any],
    *,
    shot: Mapping[str, Any],
    shot_id: str,
    track_index: int,
    base_dir: Path,
    target_sample_rate: int,
    duration_ms: Decimal,
) -> TrackSpec:
    field = f"shots[{shot_id}].tracks[{track_index}]"
    kind_value = _first(track, "type", "track_type", "bus", "kind")
    kind = _normalise_track_type(kind_value)
    raw_path = _first(track, "path", "media_path", "media", "source", "file")
    path = _track_path(raw_path, f"{field}.path", base_dir)
    gain_db = _gain_db(track, field)

    fade_in_ms = _fade_value(
        track,
        shot,
        ("fade_in_ms", "fade_in"),
        ("fade_in_ms", "fade_in"),
        f"{field}.fade_in_ms",
    )
    fade_out_ms = _fade_value(
        track,
        shot,
        ("fade_out_ms", "fade_out"),
        ("fade_out_ms", "fade_out"),
        f"{field}.fade_out_ms",
    )
    if fade_in_ms + fade_out_ms > duration_ms:
        raise _business(
            ErrorCode.E_FADE_OVERFLOW,
            f"{shot_id}: fade in plus fade out exceeds shot duration",
        )

    fade_in_samples = ms_to_samples(fade_in_ms, target_sample_rate)
    fade_out_samples = ms_to_samples(fade_out_ms, target_sample_rate)
    duration_samples = ms_to_samples(duration_ms, target_sample_rate)
    if fade_in_samples + fade_out_samples > duration_samples:
        raise _business(
            ErrorCode.E_FADE_OVERFLOW,
            f"{shot_id}: sampled fades exceed the shot duration",
        )

    return TrackSpec(
        kind=kind,
        path=path,
        gain_db=gain_db,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
        fade_in_samples=fade_in_samples,
        fade_out_samples=fade_out_samples,
    )


def _tracks_for_shot(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_tracks = raw.get("tracks")
    if raw_tracks is None:
        if _first(raw, "path", "media_path", "media", "source", "file") is not None:
            return [raw]
        grouped: list[Mapping[str, Any]] = []
        for kind in TRACK_TYPES:
            if kind not in raw:
                continue
            value = raw[kind]
            if isinstance(value, str):
                value = {"path": value}
            if not isinstance(value, Mapping):
                raise _business(
                    ErrorCode.E_SHOT_DESC,
                    f"shot track {kind!r} must be an object or a path string",
                )
            item = dict(value)
            item.setdefault("type", kind)
            grouped.append(item)
        if grouped:
            return grouped
        return []
    if isinstance(raw_tracks, Mapping):
        result: list[Mapping[str, Any]] = []
        for kind, value in raw_tracks.items():
            if isinstance(value, str):
                value = {"path": value}
            if not isinstance(value, Mapping):
                raise _business(
                    ErrorCode.E_SHOT_DESC,
                    f"track {kind!r} must be an object or a path string",
                )
            item = dict(value)
            item.setdefault("type", kind)
            result.append(item)
        return result
    if isinstance(raw_tracks, list):
        if not all(isinstance(item, Mapping) for item in raw_tracks):
            raise _business(ErrorCode.E_SHOT_DESC, "every track must be an object")
        return list(raw_tracks)
    raise _business(ErrorCode.E_SHOT_DESC, "tracks must be an array or object")


def _overlap_metadata(raw: Mapping[str, Any], shot_id: str) -> tuple[bool, Decimal]:
    allow = False
    for key in ("allow_overlap", "overlaps_previous"):
        if key in raw:
            allow = _as_bool(raw[key], f"{shot_id}.{key}")
    crossfade_ms = Decimal(0)
    for key in (
        "crossfade_ms",
        "transition_in_ms",
        "overlap_ms",
        "crossfade_duration_ms",
    ):
        if key not in raw:
            continue
        value = _number(raw[key], f"{shot_id}.{key}", ErrorCode.E_SHOT_DESC)
        if value < 0:
            raise _business(
                ErrorCode.E_SHOT_DESC, f"{shot_id}.{key} cannot be negative"
            )
        crossfade_ms = max(crossfade_ms, value)
        if value > 0:
            allow = True

    if "overlap" in raw:
        overlap = raw["overlap"]
        if isinstance(overlap, Mapping):
            value = _first(
                overlap,
                "duration_ms",
                "crossfade_ms",
                "fade_ms",
                "transition_ms",
                "ms",
                "duration",
            )
            if value is not None:
                parsed = _number(value, f"{shot_id}.overlap", ErrorCode.E_SHOT_DESC)
                if parsed < 0:
                    raise _business(
                        ErrorCode.E_SHOT_DESC,
                        f"{shot_id}.overlap cannot be negative",
                    )
                crossfade_ms = max(crossfade_ms, parsed)
                if parsed > 0:
                    allow = True
            elif _first(overlap, "allowed", "allow", "enabled", "declared") is not None:
                allow = _as_bool(
                    _first(overlap, "allowed", "allow", "enabled", "declared"),
                    f"{shot_id}.overlap",
                ) or allow
            else:
                allow = True
        elif isinstance(overlap, (bool, int, str)):
            if isinstance(overlap, (int, str)) and not isinstance(overlap, bool):
                try:
                    numeric_overlap = _number(
                        overlap, f"{shot_id}.overlap", ErrorCode.E_SHOT_DESC
                    )
                except MixerError:
                    allow = _as_bool(overlap, f"{shot_id}.overlap")
                else:
                    if numeric_overlap < 0:
                        raise _business(
                            ErrorCode.E_SHOT_DESC,
                            f"{shot_id}.overlap cannot be negative",
                        )
                    crossfade_ms = max(crossfade_ms, numeric_overlap)
                    if numeric_overlap > 0:
                        allow = True
            else:
                allow = _as_bool(overlap, f"{shot_id}.overlap") or allow
        else:
            raise _business(ErrorCode.E_SHOT_DESC, f"{shot_id}.overlap is invalid")
    transition = raw.get("transition")
    if isinstance(transition, Mapping):
        value = _first(
            transition, "duration_ms", "crossfade_ms", "fade_ms", "ms", "duration"
        )
        if value is not None:
            parsed = _number(
                value, f"{shot_id}.transition", ErrorCode.E_SHOT_DESC
            )
            if parsed < 0:
                raise _business(
                    ErrorCode.E_SHOT_DESC,
                    f"{shot_id}.transition cannot be negative",
                )
            crossfade_ms = max(crossfade_ms, parsed)
            if parsed > 0:
                allow = True
    return allow, crossfade_ms


def _parse_shot(
    raw: Any,
    index: int,
    *,
    base_dir: Path,
    target_sample_rate: int,
) -> ShotSpec:
    if not isinstance(raw, Mapping):
        raise _business(ErrorCode.E_SHOT_DESC, f"shots[{index}] must be an object")
    for unit_key in ("time_unit", "unit"):
        if unit_key not in raw:
            continue
        unit = raw[unit_key]
        if not isinstance(unit, str) or unit.strip().lower() not in (
            "ms",
            "millisecond",
            "milliseconds",
        ):
            raise _business(
                ErrorCode.E_SHOT_DESC,
                f"shots[{index}] must use milliseconds (time_unit='ms')",
            )
    shot_id = _first(raw, "shot_id", "id", "name")
    if not isinstance(shot_id, str) or not shot_id.strip():
        raise _business(ErrorCode.E_SHOT_DESC, f"shots[{index}] has no valid id")
    shot_id = shot_id.strip()

    timeline = raw.get("timeline")
    timeline_mapping = timeline if isinstance(timeline, Mapping) else {}
    raw_start = _first(
        raw, "start_ms", "start_time_ms", "start"
    )
    raw_end = _first(raw, "end_ms", "end_time_ms", "end")
    if raw_start is None:
        raw_start = _first(timeline_mapping, "start_ms", "start")
    if raw_end is None:
        raw_end = _first(timeline_mapping, "end_ms", "end")

    if raw_start is None or raw_end is None:
        duration_value = _first(raw, "duration_ms", "duration")
        if duration_value is None:
            duration_value = _first(timeline_mapping, "duration_ms", "duration")
        if duration_value is None:
            raise _business(
                ErrorCode.E_SHOT_DESC,
                f"{shot_id}: start_ms and end_ms (or duration_ms) are required",
            )
        duration = _number(
            duration_value, f"{shot_id}.duration_ms", ErrorCode.E_SHOT_DESC
        )
        if raw_end is None:
            start_value = (
                Decimal(0)
                if raw_start is None
                else _number(raw_start, f"{shot_id}.start_ms", ErrorCode.E_SHOT_DESC)
            )
            raw_start = start_value
            raw_end = start_value + duration
        else:
            end_value = _number(raw_end, f"{shot_id}.end_ms", ErrorCode.E_SHOT_DESC)
            raw_end = end_value
            raw_start = end_value - duration

    start_ms = _number(raw_start, f"{shot_id}.start_ms", ErrorCode.E_SHOT_DESC)
    end_ms = _number(raw_end, f"{shot_id}.end_ms", ErrorCode.E_SHOT_DESC)
    if start_ms < 0:
        raise _business(ErrorCode.E_SHOT_DESC, f"{shot_id}: start_ms cannot be negative")
    if end_ms <= start_ms:
        raise _business(
            ErrorCode.E_SHOT_DESC, f"{shot_id}: end_ms must be greater than start_ms"
        )

    start_sample = ms_to_samples(start_ms, target_sample_rate)
    end_sample = ms_to_samples(end_ms, target_sample_rate)
    if end_sample <= start_sample:
        raise _business(
            ErrorCode.E_SHOT_DESC,
            f"{shot_id}: shot is shorter than one target sample",
        )

    duration_ms = end_ms - start_ms
    raw_tracks = _tracks_for_shot(raw)
    if not raw_tracks:
        raise _business(ErrorCode.E_SHOT_DESC, f"{shot_id}: at least one track is required")
    tracks: list[TrackSpec] = []
    seen_track_types: set[str] = set()
    for track_index, track in enumerate(raw_tracks):
        parsed_track = _parse_track(
            track,
            shot=raw,
            shot_id=shot_id,
            track_index=track_index,
            base_dir=base_dir,
            target_sample_rate=target_sample_rate,
            duration_ms=duration_ms,
        )
        if parsed_track.kind in seen_track_types:
            raise _business(
                ErrorCode.E_SHOT_DESC,
                f"{shot_id}: duplicate track type {parsed_track.kind}",
            )
        seen_track_types.add(parsed_track.kind)
        tracks.append(parsed_track)

    allow_overlap, crossfade_ms = _overlap_metadata(raw, shot_id)
    return ShotSpec(
        shot_id=shot_id,
        start_ms=start_ms,
        end_ms=end_ms,
        start_sample=start_sample,
        end_sample=end_sample,
        duration_samples=end_sample - start_sample,
        tracks=tuple(tracks),
        allow_overlap=allow_overlap,
        crossfade_ms=crossfade_ms,
    )


def _target_sample_rate(data: Mapping[str, Any]) -> int:
    raw = _first(data, "target_sample_rate", "sample_rate", "output_sample_rate")
    if raw is None:
        return 48_000
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise _business(ErrorCode.E_SHOT_DESC, "target_sample_rate must be an integer")
    if raw not in SUPPORTED_SAMPLE_RATES:
        raise _business(
            ErrorCode.E_SHOT_DESC,
            "target_sample_rate must be 44100 or 48000",
        )
    return raw


def _output_channels(data: Mapping[str, Any]) -> int:
    raw = _first(data, "output_channels", "channels")
    if raw is None:
        return 4
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 8:
        raise _business(ErrorCode.E_SHOT_DESC, "channels must be an integer from 1 to 8")
    return raw


def _output_subtype(data: Mapping[str, Any]) -> str:
    raw = data.get("output_subtype", data.get("subtype", "PCM_16"))
    if not isinstance(raw, str) or raw.upper() not in SUPPORTED_SUBTYPES:
        raise _business(
            ErrorCode.E_SHOT_DESC,
            "output_subtype must be PCM_16, PCM_24, or FLOAT",
        )
    return raw.upper()


def _bus_weights(data: Mapping[str, Any]) -> Mapping[str, float]:
    raw = data.get("bus_weights")
    if raw is None:
        return dict(DEFAULT_BUS_WEIGHTS)
    if not isinstance(raw, Mapping):
        raise _business(ErrorCode.E_SHOT_DESC, "bus_weights must be an object")
    result = dict(DEFAULT_BUS_WEIGHTS)
    for key, value in raw.items():
        normalised = _normalise_track_type(key)
        weight = _float_number(value, f"bus_weights.{key}", ErrorCode.E_SHOT_DESC)
        if weight < 0:
            raise _business(
                ErrorCode.E_SHOT_DESC, f"bus_weights.{key} cannot be negative"
            )
        result[normalised] = weight
    return result


def _limiter_ceiling(data: Mapping[str, Any]) -> float:
    raw = _first(data, "limiter_ceiling_dbfs", "ceiling_dbfs")
    if raw is None:
        return 0.0
    ceiling = _float_number(raw, "limiter_ceiling_dbfs", ErrorCode.E_SHOT_DESC)
    if ceiling > 0:
        raise _business(
            ErrorCode.E_SHOT_DESC, "limiter_ceiling_dbfs cannot be above 0 dBFS"
        )
    return ceiling


def load_project_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    data = _read_json(source, ErrorCode.E_SHOT_DESC, "shot description")
    if not isinstance(data, Mapping):
        raise _business(ErrorCode.E_SHOT_DESC, "shot description root must be an object")

    for unit_key in ("time_unit", "unit"):
        if unit_key in data:
            unit = data[unit_key]
            if not isinstance(unit, str) or unit.strip().lower() not in (
                "ms",
                "millisecond",
                "milliseconds",
            ):
                raise _business(
                    ErrorCode.E_SHOT_DESC,
                    "all shot times must use milliseconds (time_unit='ms')",
                )

    target_sample_rate = _target_sample_rate(data)
    raw_shots = _first(data, "shots", "shot_list")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise _business(ErrorCode.E_SHOT_DESC, "shots must be a non-empty array")

    base_dir = source.parent
    shots: list[ShotSpec] = []
    ids: set[str] = set()
    for index, raw_shot in enumerate(raw_shots):
        try:
            parsed = _parse_shot(
                raw_shot,
                index,
                base_dir=base_dir,
                target_sample_rate=target_sample_rate,
            )
        except MixerError as error:
            if "partial_report" not in error.context:
                error.context["target_sample_rate"] = target_sample_rate
                error.context["partial_report"] = {
                    "schema_version": 1,
                    "status": "error",
                    "target_sample_rate": target_sample_rate,
                    "total_samples": max(
                        (shot.end_sample for shot in shots), default=0
                    ),
                    "shots": [
                        {
                            "shot_id": shot.shot_id,
                            "start_sample": int(shot.start_sample),
                            "end_sample": int(shot.end_sample),
                            "duration_samples": int(shot.duration_samples),
                            "peak_dbfs": None,
                            "raw_peak": None,
                        }
                        for shot in sorted(
                            shots, key=lambda item: (item.start_sample, item.shot_id)
                        )
                    ],
                    "bus_weights": dict(DEFAULT_BUS_WEIGHTS),
                    "bus_peak_dbfs": None,
                    "master_peak_dbfs": None,
                    "limiter_active": False,
                    "limiter_gain": 1.0,
                }
            raise
        if parsed.shot_id in ids:
            raise _business(
                ErrorCode.E_SHOT_DESC, f"duplicate shot id {parsed.shot_id!r}"
            )
        ids.add(parsed.shot_id)
        shots.append(parsed)

    return ProjectConfig(
        source=source,
        target_sample_rate=target_sample_rate,
        output_channels=_output_channels(data),
        output_subtype=_output_subtype(data),
        bus_weights=_bus_weights(data),
        limiter_ceiling_dbfs=_limiter_ceiling(data),
        shots=tuple(shots),
        shots_by_id={shot.shot_id: shot for shot in shots},
    )


def _choice_id(value: Any, index: int) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        raw = _first(value, "shot_id", "id", "choice", "value")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise _business(
        ErrorCode.E_PATH_INVALID, f"choice path entry {index} is not a shot id"
    )


def load_choice_path(path: str | Path) -> list[str]:
    source = Path(path).resolve()
    data = _read_json(source, ErrorCode.E_PATH_INVALID, "choice path")
    if isinstance(data, Mapping):
        data = _first(
            data, "path", "choices", "selection", "selected", "selected_shots"
        )
    if not isinstance(data, list):
        raise _business(
            ErrorCode.E_PATH_INVALID,
            "choice path must be an array or contain a path/choices array",
        )
    if not data:
        raise _business(ErrorCode.E_PATH_INVALID, "choice path cannot be empty")
    return [_choice_id(item, index) for index, item in enumerate(data)]


def resolve_path_shots(
    config: ProjectConfig, choice_path: Iterable[str]
) -> list[ShotSpec]:
    selected: list[ShotSpec] = []
    seen: set[str] = set()
    for shot_id in choice_path:
        shot = config.shots_by_id.get(shot_id)
        if shot is None:
            raise MixerError(
                ErrorCode.E_PATH_INVALID,
                f"choice path references unknown shot {shot_id!r}",
                context={
                    "partial_shot_ids": [item.shot_id for item in selected],
                    "invalid_shot_id": shot_id,
                },
            )
        if shot_id in seen:
            raise MixerError(
                ErrorCode.E_PATH_INVALID,
                f"choice path references shot {shot_id!r} more than once",
                context={"partial_shot_ids": [item.shot_id for item in selected]},
            )
        seen.add(shot_id)
        selected.append(shot)
    return selected
