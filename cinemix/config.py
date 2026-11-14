from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CinemixError, Diagnostics

CURVES = {"linear", "equal_power", "s_curve"}


@dataclass(frozen=True)
class FadeSpec:
    start: float | None = None
    end: float | None = None
    duration: float | None = None
    curve: str = "linear"


@dataclass(frozen=True)
class TrackSpec:
    id: str
    path: Path
    start: float
    gain_db: float
    pan: float
    fade_in: FadeSpec | None
    fade_out: FadeSpec | None
    resample_to: int | None
    source_index: int


@dataclass(frozen=True)
class CrossfadeSpec:
    from_track: str
    to_track: str
    start: float
    end: float
    curve: str
    source_index: int


@dataclass(frozen=True)
class LimiterSpec:
    attack_ms: float
    release_ms: float
    ratio: float
    lookahead_ms: float
    compression_threshold_dbtp: float | None
    safety_margin_db: float


@dataclass(frozen=True)
class AdaptiveGainSpec:
    enabled: bool
    threshold_lufs: float
    max_cut_db: float


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    sample_rate: int | None
    output: Path | None
    target_lufs: float
    true_peak_dbtp: float
    output_channels: int
    duration_seconds: float | None
    resample_mismatched: bool
    tracks: tuple[TrackSpec, ...]
    crossfades: tuple[CrossfadeSpec, ...]
    limiter: LimiterSpec
    adaptive_gain: AdaptiveGainSpec


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CinemixError(
            "E_CONFIG_SCHEMA", f"{path} must be a JSON object", context={"field": path}
        )
    return value


def _alias(mapping: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be a finite number",
            context={"field": path, "value": value},
        )
    number = float(value)
    if not math.isfinite(number):
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be finite",
            context={"field": path, "value": value},
        )
    return number


def _positive_number(value: Any, path: str, *, allow_zero: bool = False) -> float:
    number = _number(value, path)
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be {qualifier}",
            context={"field": path, "value": value},
        )
    return number


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be true or false",
            context={"field": path, "value": value},
        )
    return value


def _sample_rate(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be an integer sample rate",
            context={"field": path, "value": value},
        )
    if value < 8000 or value > 384000:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be between 8000 and 384000 Hz",
            context={"field": path, "value": value},
        )
    return value


def _curve(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be a curve name",
            context={"field": path, "value": value},
        )
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "equalpower": "equal_power",
        "s": "s_curve",
        "scurve": "s_curve",
        "smoothstep": "s_curve",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CURVES:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be one of: linear, equal_power, s_curve",
            context={"field": path, "value": value},
        )
    return normalized


def _parse_fade(value: Any, path: str) -> FadeSpec | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} must be a duration or an object",
            context={"field": path},
        )
    if isinstance(value, (int, float)):
        duration = _positive_number(value, path, allow_zero=True)
        return FadeSpec(duration=duration)
    object_value = _mapping(value, path)
    start = object_value.get("start")
    end = object_value.get("end")
    duration = object_value.get("duration")
    if start is None and end is None and duration is None:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} needs start/end or duration",
            context={"field": path},
        )
    parsed_start = _positive_number(start, f"{path}.start", allow_zero=True) if start is not None else None
    parsed_end = _positive_number(end, f"{path}.end", allow_zero=True) if end is not None else None
    parsed_duration = (
        _positive_number(duration, f"{path}.duration", allow_zero=True)
        if duration is not None
        else None
    )
    if parsed_start is not None and parsed_end is not None and parsed_end <= parsed_start:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path}.end must be greater than {path}.start",
            context={"field": path},
        )
    if (
        parsed_duration is not None
        and parsed_duration == 0
        and parsed_start is None
        and parsed_end is None
    ):
        return None
    return FadeSpec(
        start=parsed_start,
        end=parsed_end,
        duration=parsed_duration,
        curve=_curve(object_value.get("curve", "linear"), f"{path}.curve"),
    )


def _parse_track(value: Any, index: int, config_dir: Path, diagnostics: Diagnostics) -> TrackSpec:
    path = f"tracks[{index}]"
    track = _mapping(value, path)
    track_id = track.get("id")
    if not isinstance(track_id, str) or not track_id.strip():
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path}.id must be a non-empty string",
            context={"field": f"{path}.id"},
        )
    track_id = track_id.strip()

    source = _alias(track, ("path", "file", "filename"))
    if not isinstance(source, str) or not source.strip():
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path}.path must be a non-empty WAV path",
            context={"field": f"{path}.path", "track": track_id},
        )
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (config_dir / source_path).resolve()

    start = _positive_number(
        _alias(track, ("start", "start_time", "timeline_start"), 0.0),
        f"{path}.start",
        allow_zero=True,
    )
    gain_db = _number(_alias(track, ("gain_db", "gain", "volume_db"), 0.0), f"{path}.gain_db")
    if gain_db < -96.0 or gain_db > 24.0:
        original = gain_db
        gain_db = float(min(24.0, max(-96.0, gain_db)))
        diagnostics.warning(
            "W_GAIN_CLAMPED",
            f"Track {track_id} gain {original:.3f} dB is outside [-96, 24]; clamped to {gain_db:.3f} dB",
            track=track_id,
            original_gain_db=original,
            applied_gain_db=gain_db,
        )
    pan = _number(_alias(track, ("pan", "balance"), 0.0), f"{path}.pan")
    if pan < -1.0 or pan > 1.0:
        original = pan
        pan = float(min(1.0, max(-1.0, pan)))
        diagnostics.warning(
            "W_PAN_CLAMPED",
            f"Track {track_id} pan {original:.3f} is outside [-1, 1]; clamped to {pan:.3f}",
            track=track_id,
            original_pan=original,
            applied_pan=pan,
        )

    fade_in = _parse_fade(
        _alias(track, ("fade_in", "fade_in_spec")), f"{path}.fade_in"
    )
    fade_out = _parse_fade(
        _alias(track, ("fade_out", "fade_out_spec")), f"{path}.fade_out"
    )
    resample_value = _alias(track, ("resample_to", "target_sample_rate", "convert_to_sample_rate"))
    resample_to = (
        _sample_rate(resample_value, f"{path}.resample_to")
        if resample_value is not None
        else None
    )
    return TrackSpec(
        id=track_id,
        path=source_path,
        start=start,
        gain_db=gain_db,
        pan=pan,
        fade_in=fade_in,
        fade_out=fade_out,
        resample_to=resample_to,
        source_index=index,
    )


def _parse_crossfade(value: Any, index: int) -> CrossfadeSpec:
    path = f"crossfades[{index}]"
    crossfade = _mapping(value, path)
    from_track = _alias(crossfade, ("from", "from_track", "out_track"))
    to_track = _alias(crossfade, ("to", "to_track", "in_track"))
    for label, track_id in (("from", from_track), ("to", to_track)):
        if not isinstance(track_id, str) or not track_id.strip():
            raise CinemixError(
                "E_CONFIG_FIELD",
                f"{path}.{label} must name a track",
                context={"field": f"{path}.{label}"},
            )
    if from_track == to_track:
        raise CinemixError(
            "E_CROSSFADE_CONFLICT",
            f"{path} cannot crossfade a track with itself",
            context={"field": path},
        )
    start = _positive_number(
        _alias(crossfade, ("start", "start_time")), f"{path}.start", allow_zero=True
    )
    end_value = crossfade.get("end")
    duration_value = crossfade.get("duration")
    if end_value is not None:
        end = _positive_number(end_value, f"{path}.end", allow_zero=True)
    elif duration_value is not None:
        duration = _positive_number(duration_value, f"{path}.duration")
        end = start + duration
    else:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path} needs end or duration",
            context={"field": path},
        )
    if end <= start:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"{path}.end must be greater than start",
            context={"field": path},
        )
    return CrossfadeSpec(
        from_track=str(from_track).strip(),
        to_track=str(to_track).strip(),
        start=start,
        end=end,
        curve=_curve(crossfade.get("curve", "linear"), f"{path}.curve"),
        source_index=index,
    )


def _detect_crossfade_conflicts(
    crossfades: tuple[CrossfadeSpec, ...],
) -> None:
    for left_index, left in enumerate(crossfades):
        left_tracks = {left.from_track, left.to_track}
        for right in crossfades[left_index + 1 :]:
            overlap = min(left.end, right.end) - max(left.start, right.start)
            shared = left_tracks.intersection({right.from_track, right.to_track})
            if shared and overlap > 1e-9:
                raise CinemixError(
                    "E_CROSSFADE_CONFLICT",
                    "Overlapping crossfades share a track",
                    context={
                        "crossfades": [left.source_index, right.source_index],
                        "tracks": sorted(shared),
                        "overlap_seconds": round(overlap, 9),
                    },
                )


def _parse_limiter(root: Mapping[str, Any], true_peak_dbtp: float) -> LimiterSpec:
    section = _mapping(root.get("limiter", {}), "limiter")
    attack_ms = _positive_number(
        _alias(section, ("attack_ms", "attack_time_ms"), 5.0),
        "limiter.attack_ms",
    )
    release_ms = _positive_number(
        _alias(section, ("release_ms", "release_time_ms"), 120.0),
        "limiter.release_ms",
    )
    ratio = _number(_alias(section, ("ratio", "compression_ratio"), 20.0), "limiter.ratio")
    if ratio < 1.0 or ratio > 1000.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "limiter.ratio must be between 1 and 1000",
            context={"field": "limiter.ratio", "value": ratio},
        )
    lookahead_ms = _positive_number(
        _alias(section, ("lookahead_ms", "lookahead_time_ms"), max(attack_ms, 2.0)),
        "limiter.lookahead_ms",
    )
    if lookahead_ms > 200.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "limiter.lookahead_ms must not exceed 200 ms",
            context={"field": "limiter.lookahead_ms", "value": lookahead_ms},
        )
    threshold_value = _alias(
        section,
        ("compression_threshold_dbtp", "limiter_threshold_dbtp", "threshold_dbtp"),
    )
    threshold = (
        _number(threshold_value, "limiter.compression_threshold_dbtp")
        if threshold_value is not None
        else None
    )
    if threshold is not None and threshold > true_peak_dbtp:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "limiter.compression_threshold_dbtp cannot exceed true_peak_dbtp",
            context={
                "field": "limiter.compression_threshold_dbtp",
                "threshold": threshold,
                "true_peak_dbtp": true_peak_dbtp,
            },
        )
    safety_margin = _positive_number(
        _alias(section, ("safety_margin_db", "true_peak_margin_db"), 0.25),
        "limiter.safety_margin_db",
        allow_zero=True,
    )
    if safety_margin > 3.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "limiter.safety_margin_db must not exceed 3 dB",
            context={"field": "limiter.safety_margin_db", "value": safety_margin},
        )
    return LimiterSpec(
        attack_ms=attack_ms,
        release_ms=release_ms,
        ratio=ratio,
        lookahead_ms=lookahead_ms,
        compression_threshold_dbtp=threshold,
        safety_margin_db=safety_margin,
    )


def _parse_adaptive(root: Mapping[str, Any], target_lufs: float) -> AdaptiveGainSpec:
    value = root.get("adaptive_gain", {})
    if isinstance(value, bool):
        return AdaptiveGainSpec(
            enabled=value,
            threshold_lufs=target_lufs,
            max_cut_db=12.0,
        )
    section = _mapping(value, "adaptive_gain")
    enabled = _boolean(section.get("enabled", False), "adaptive_gain.enabled")
    threshold = _number(
        _alias(section, ("threshold_lufs", "trigger_lufs"), target_lufs),
        "adaptive_gain.threshold_lufs",
    )
    if threshold < -70.0 or threshold > 0.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "adaptive_gain.threshold_lufs must be between -70 and 0",
            context={"field": "adaptive_gain.threshold_lufs", "value": threshold},
        )
    max_cut = _positive_number(
        _alias(section, ("max_cut_db", "max_attenuation_db"), 12.0),
        "adaptive_gain.max_cut_db",
        allow_zero=True,
    )
    if max_cut > 60.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "adaptive_gain.max_cut_db must not exceed 60 dB",
            context={"field": "adaptive_gain.max_cut_db", "value": max_cut},
        )
    return AdaptiveGainSpec(
        enabled=enabled, threshold_lufs=threshold, max_cut_db=max_cut
    )


def load_config(path: str | Path, diagnostics: Diagnostics) -> ProjectConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise CinemixError(
            "E_INPUT_MISSING",
            f"Configuration file does not exist: {config_path}",
            context={"path": str(config_path)},
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CinemixError(
            "E_CONFIG_JSON",
            f"Invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}",
            context={
                "path": str(config_path),
                "line": exc.lineno,
                "column": exc.colno,
            },
        ) from exc
    except OSError as exc:
        raise CinemixError(
            "E_INPUT_MISSING",
            f"Cannot read configuration {config_path}: {exc}",
            context={"path": str(config_path)},
        ) from exc
    if not isinstance(raw, Mapping):
        raise CinemixError(
            "E_CONFIG_SCHEMA", "Top-level JSON value must be an object"
        )

    root = _mapping(raw, "root")
    project_value = root.get("project", {})
    project = _mapping(project_value, "project")

    def project_alias(names: Iterable[str], default: Any = None) -> Any:
        value = _alias(project, names, None)
        return _alias(root, names, default) if value is None else value

    sample_rate_value = project_alias(("sample_rate", "output_sample_rate"))
    sample_rate = (
        _sample_rate(sample_rate_value, "project.sample_rate")
        if sample_rate_value is not None
        else None
    )
    target_lufs = _number(
        project_alias(("target_lufs", "target_loudness_lufs", "loudness_target"), -23.0),
        "target_lufs",
    )
    if target_lufs < -70.0 or target_lufs > 0.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "target_lufs must be between -70 and 0",
            context={"field": "target_lufs", "value": target_lufs},
        )
    true_peak_dbtp = _number(
        project_alias(
            ("true_peak_dbtp", "max_true_peak_dbtp", "true_peak_limit_dbtp"), -1.0
        ),
        "true_peak_dbtp",
    )
    if true_peak_dbtp < -20.0 or true_peak_dbtp > 0.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "true_peak_dbtp must be between -20 and 0",
            context={"field": "true_peak_dbtp", "value": true_peak_dbtp},
        )
    output_channels_value = project_alias(("output_channels", "channels"), 2)
    if isinstance(output_channels_value, bool) or not isinstance(output_channels_value, int):
        raise CinemixError(
            "E_CONFIG_FIELD",
            "output_channels must be 1 or 2",
            context={"field": "output_channels", "value": output_channels_value},
        )
    if output_channels_value not in (1, 2):
        raise CinemixError(
            "E_CONFIG_FIELD",
            "output_channels must be 1 or 2",
            context={"field": "output_channels", "value": output_channels_value},
        )
    output_channels = output_channels_value

    duration_value = project_alias(("duration_seconds", "duration"), None)
    duration_seconds = (
        _positive_number(duration_value, "duration_seconds")
        if duration_value is not None
        else None
    )
    resample_value = project_alias(("resample", "auto_resample"), False)
    resample_mismatched = _boolean(resample_value, "project.resample")

    output_value = project_alias(("output", "output_path", "output_file"))
    if output_value is not None and (not isinstance(output_value, str) or not output_value.strip()):
        raise CinemixError(
            "E_CONFIG_FIELD",
            "project.output must be a non-empty path string",
            context={"field": "project.output"},
        )
    output = Path(output_value) if output_value is not None else None
    if output is not None and not output.is_absolute():
        output = (config_path.parent / output).resolve()

    tracks_value = root.get("tracks")
    if not isinstance(tracks_value, list) or not tracks_value:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "tracks must be a non-empty array",
            context={"field": "tracks"},
        )
    tracks: list[TrackSpec] = []
    seen_ids: set[str] = set()
    for index, track_value in enumerate(tracks_value):
        track = _parse_track(track_value, index, config_path.parent, diagnostics)
        if track.id in seen_ids:
            raise CinemixError(
                "E_CONFIG_FIELD",
                f"Duplicate track id {track.id}",
                context={"field": f"tracks[{index}].id", "track": track.id},
            )
        seen_ids.add(track.id)
        tracks.append(track)

    crossfades_value = root.get("crossfades", [])
    if not isinstance(crossfades_value, list):
        raise CinemixError(
            "E_CONFIG_FIELD",
            "crossfades must be an array",
            context={"field": "crossfades"},
        )
    crossfades = tuple(
        _parse_crossfade(value, index)
        for index, value in enumerate(crossfades_value)
    )
    for crossfade in crossfades:
        for track_id in (crossfade.from_track, crossfade.to_track):
            if track_id not in seen_ids:
                raise CinemixError(
                    "E_CONFIG_FIELD",
                    f"crossfades[{crossfade.source_index}] references unknown track {track_id}",
                    context={
                        "field": f"crossfades[{crossfade.source_index}]",
                        "track": track_id,
                    },
                )
    _detect_crossfade_conflicts(crossfades)

    limiter = _parse_limiter(root, true_peak_dbtp)
    adaptive_gain = _parse_adaptive(root, target_lufs)
    return ProjectConfig(
        path=config_path,
        sample_rate=sample_rate,
        output=output,
        target_lufs=target_lufs,
        true_peak_dbtp=true_peak_dbtp,
        output_channels=output_channels,
        duration_seconds=duration_seconds,
        resample_mismatched=resample_mismatched,
        tracks=tuple(tracks),
        crossfades=crossfades,
        limiter=limiter,
        adaptive_gain=adaptive_gain,
    )
