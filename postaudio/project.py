"""Manifest and output-configuration loading with header-only validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from .automation import AutomationError, validate_automation
from .wavio import WavError, WavHeader, read_wav_header


class ProjectError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class Track:
    name: str
    source_path: str
    expected_sample_rate: int
    expected_bit_depth: int
    expected_channels: int
    automation: list[dict[str, float]]
    sync: dict[str, Any] | None
    raw: dict[str, Any]


@dataclass
class Project:
    root: str
    manifest_path: str
    manifest: dict[str, Any]
    tracks: list[Track]
    timeline_seconds: float | None
    config_path: str

    @property
    def sorted_tracks(self) -> list[Track]:
        return sorted(self.tracks, key=lambda track: track.name)


def _read_json(path: str, label: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ProjectError("PROJECT_FILE_NOT_FOUND", f"{label} not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectError("PROJECT_JSON_INVALID", f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProjectError("PROJECT_JSON_INVALID", f"{label} root must be an object")
    return value


def _required_int(mapping: dict[str, Any], keys: tuple[str, ...], label: str) -> int:
    for key in keys:
        if key in mapping:
            try:
                value = int(mapping[key])
            except (TypeError, ValueError) as exc:
                raise ProjectError("MANIFEST_FIELD_INVALID", f"{label} must be an integer") from exc
            return value
    raise ProjectError("MANIFEST_FIELD_MISSING", f"{label} is required")


def load_project(root: str) -> Project:
    manifest_path = os.path.join(root, "manifest.json")
    manifest = _read_json(manifest_path, "manifest")
    raw_tracks = manifest.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ProjectError("MANIFEST_TRACKS_MISSING", "manifest.tracks must be a non-empty array")
    seen_names: set[str] = set()
    tracks: list[Track] = []
    for index, raw in enumerate(raw_tracks):
        if not isinstance(raw, dict):
            raise ProjectError("MANIFEST_TRACK_INVALID", f"track index {index} must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ProjectError("MANIFEST_TRACK_INVALID", f"track index {index} needs a name")
        if name in seen_names:
            raise ProjectError("MANIFEST_DUPLICATE_TRACK", f"duplicate track name {name}")
        seen_names.add(name)
        source = raw.get("file", raw.get("source", raw.get("source_file")))
        if not isinstance(source, str) or not source:
            raise ProjectError("MANIFEST_TRACK_INVALID", f"track {name} needs a source file")
        source_path = source if os.path.isabs(source) else os.path.join(root, source)
        expected_sr = _required_int(
            raw,
            ("expected_sample_rate", "sample_rate", "sample_rate_hz"),
            f"track {name} expected_sample_rate",
        )
        expected_depth = _required_int(
            raw,
            ("expected_bit_depth", "bit_depth", "bit_depth_bits"),
            f"track {name} expected_bit_depth",
        )
        expected_channels = _required_int(
            raw,
            ("expected_channels", "channels", "channel_count"),
            f"track {name} expected_channels",
        )
        try:
            automation = validate_automation(
                name,
                raw.get("gain_automation", raw.get("automation", raw.get("gain_curve"))),
            )
        except AutomationError as exc:
            raise ProjectError(
                exc.code,
                f"track={name} node_index={exc.node_index} {exc.detail}",
            ) from exc
        sync_value = raw.get("sync", raw.get("sync_marker"))
        if sync_value is not None and not isinstance(sync_value, dict):
            raise ProjectError("MANIFEST_TRACK_INVALID", f"track {name} sync must be an object")
        tracks.append(
            Track(
                name=name,
                source_path=source_path,
                expected_sample_rate=expected_sr,
                expected_bit_depth=expected_depth,
                expected_channels=expected_channels,
                automation=automation,
                sync=sync_value,
                raw=raw,
            )
        )
    timeline_seconds = manifest.get("timeline_seconds")
    if timeline_seconds is not None:
        try:
            timeline_seconds = float(timeline_seconds)
        except (TypeError, ValueError) as exc:
            raise ProjectError("MANIFEST_FIELD_INVALID", "timeline_seconds must be numeric") from exc
        if timeline_seconds <= 0.0:
            raise ProjectError("MANIFEST_FIELD_INVALID", "timeline_seconds must be positive")
    config_path = manifest.get("output_config", "output_config.json")
    if not isinstance(config_path, str) or not config_path:
        raise ProjectError("MANIFEST_FIELD_INVALID", "output_config must be a path")
    if not os.path.isabs(config_path):
        config_path = os.path.join(root, config_path)
    return Project(
        root=os.path.abspath(root),
        manifest_path=manifest_path,
        manifest=manifest,
        tracks=tracks,
        timeline_seconds=timeline_seconds,
        config_path=config_path,
    )


@dataclass(frozen=True)
class OutputConfig:
    sample_rate: int
    channels: int
    encoding: str
    peak_limit_dbfs: float
    limiter_lookahead_ms: float
    smooth_window_ms: float
    loudness_target: float | None
    resampler_half_width: int
    raw: dict[str, Any]

    @property
    def peak_limit_linear(self) -> float:
        return float(10.0 ** (self.peak_limit_dbfs / 20.0))


def _normalize_encoding(value: Any) -> str:
    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    aliases = {
        "pcm16": "pcm16",
        "16": "pcm16",
        "pcms16le": "pcm16",
        "s16le": "pcm16",
        "int16": "pcm16",
        "pcm24": "pcm24",
        "24": "pcm24",
        "pcms24le": "pcm24",
        "s24le": "pcm24",
        "int24": "pcm24",
        "pcm32": "pcm32",
        "32": "pcm32",
        "pcms32le": "pcm32",
        "s32le": "pcm32",
        "int32": "pcm32",
        "float32": "float32",
        "float": "float32",
        "f32": "float32",
        "ieee32": "float32",
        "ieeefloat32": "float32",
        "wavfloat32": "float32",
    }
    if text not in aliases:
        raise ProjectError("OUTPUT_ENCODING_INVALID", f"unsupported output encoding {value!r}")
    return aliases[text]


def load_output_config(path: str) -> OutputConfig:
    raw = _read_json(path, "output config")
    settings = raw.get("output") if isinstance(raw.get("output"), dict) else raw
    try:
        sample_rate = int(
            settings.get("sample_rate", settings.get("output_sample_rate", settings.get("sample_rate_hz")))
        )
        channels = int(settings.get("channels", settings.get("channel_count", 1)))
    except (TypeError, ValueError) as exc:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "sample_rate and channels must be integers") from exc
    if sample_rate <= 0:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "sample_rate must be positive")
    if channels not in (1, 2):
        raise ProjectError("OUTPUT_CONFIG_INVALID", "channels must be 1 or 2")
    encoding = _normalize_encoding(
        settings.get("encoding", settings.get("sample_format", settings.get("format", "float32")))
    )
    try:
        peak = float(settings.get("peak_limit_dbfs", settings.get("peak_limit_db", -1.0)))
        lookahead = float(settings.get("limiter_lookahead_ms", settings.get("lookahead_ms", 5.0)))
        smooth = float(settings.get("smooth_window_ms", 10.0))
    except (TypeError, ValueError) as exc:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "numeric limiter/smoothing setting is invalid") from exc
    if peak > 0.0 or peak < -96.0:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "peak_limit_dbfs must be within [-96, 0]")
    if lookahead <= 0.0 or smooth < 0.0:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "lookahead must be positive and smoothing non-negative")
    loudness_raw = settings.get(
        "loudness_target", settings.get("loudness_target_lufs", settings.get("target_loudness"))
    )
    if loudness_raw is None:
        loudness_target = None
    else:
        try:
            loudness_target = float(loudness_raw)
        except (TypeError, ValueError) as exc:
            raise ProjectError("OUTPUT_CONFIG_INVALID", "loudness_target must be numeric or null") from exc
        if loudness_target > 0.0 or loudness_target < -96.0:
            raise ProjectError("OUTPUT_CONFIG_INVALID", "loudness_target must be within [-96, 0]")
    resampler_raw = settings.get("resampler", {})
    if resampler_raw is None:
        resampler_raw = {}
    if not isinstance(resampler_raw, dict):
        raise ProjectError("OUTPUT_CONFIG_INVALID", "resampler must be an object")
    try:
        half_width = int(resampler_raw.get("half_width_samples", 64))
    except (TypeError, ValueError) as exc:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "resampler half_width_samples must be an integer") from exc
    if half_width < 8 or half_width > 512:
        raise ProjectError("OUTPUT_CONFIG_INVALID", "resampler half_width_samples must be within [8, 512]")
    return OutputConfig(
        sample_rate=sample_rate,
        channels=channels,
        encoding=encoding,
        peak_limit_dbfs=peak,
        limiter_lookahead_ms=lookahead,
        smooth_window_ms=smooth,
        loudness_target=loudness_target,
        resampler_half_width=half_width,
        raw=raw,
    )


def validate_headers(project: Project) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    allowed_rates = project.manifest.get("allowed_sample_rates")
    allowed_depths = project.manifest.get("allowed_bit_depths")
    for track in project.sorted_tracks:
        header: WavHeader | None = None
        try:
            header = read_wav_header(track.source_path)
        except (WavError, OSError) as exc:
            code = exc.code if isinstance(exc, WavError) else "SOURCE_READ_ERROR"
            detail = exc.detail if isinstance(exc, WavError) else str(exc)
            errors.append({"track": track.name, "code": code, "detail": detail})
            continue
        if header.sample_rate != track.expected_sample_rate:
            errors.append(
                {
                    "track": track.name,
                    "code": "SAMPLE_RATE_MISMATCH",
                    "detail": f"manifest={track.expected_sample_rate} header={header.sample_rate}",
                }
            )
            continue
        if header.bit_depth != track.expected_bit_depth:
            errors.append(
                {
                    "track": track.name,
                    "code": "BIT_DEPTH_MISMATCH",
                    "detail": f"manifest={track.expected_bit_depth} header={header.bit_depth}",
                }
            )
            continue
        if header.channels != track.expected_channels:
            errors.append(
                {
                    "track": track.name,
                    "code": "CHANNEL_COUNT_MISMATCH",
                    "detail": f"manifest={track.expected_channels} header={header.channels}",
                }
            )
            continue
        if allowed_rates is not None and header.sample_rate not in [int(value) for value in allowed_rates]:
            errors.append(
                {
                    "track": track.name,
                    "code": "SAMPLE_RATE_NOT_ALLOWED",
                    "detail": f"actual={header.sample_rate} allowed={allowed_rates}",
                }
            )
            continue
        if allowed_depths is not None and header.bit_depth not in [int(value) for value in allowed_depths]:
            errors.append(
                {
                    "track": track.name,
                    "code": "BIT_DEPTH_NOT_ALLOWED",
                    "detail": f"actual={header.bit_depth} allowed={allowed_depths}",
                }
            )
            continue
        summaries.append(
            {
                "name": track.name,
                "file": os.path.relpath(track.source_path, project.root).replace(os.sep, "/"),
                "sample_rate": header.sample_rate,
                "bit_depth": header.bit_depth,
                "channels": header.channels,
                "encoding": header.encoding,
                "frames": header.frame_count,
            }
        )
    return summaries, sorted(errors, key=lambda item: (item["track"], item["code"], item["detail"]))


def header_error_to_source_error(error: dict[str, str]) -> dict[str, str]:
    code = error["code"]
    if code in {"SAMPLE_RATE_MISMATCH", "BIT_DEPTH_MISMATCH", "CHANNEL_COUNT_MISMATCH"}:
        return error
    if code in {"SOURCE_READ_ERROR", "SOURCE_NOT_FOUND"}:
        return {"track": error["track"], "code": "SOURCE_READ_ERROR", "detail": error["detail"]}
    return {"track": error["track"], "code": "SOURCE_READ_ERROR", "detail": f"{code}: {error['detail']}"}
