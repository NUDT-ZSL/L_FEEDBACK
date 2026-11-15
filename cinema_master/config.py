"""Configuration loading, normalization, and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .errors import MasterError
from .layouts import normalize_role, resolve_output_layout


GROUP_ALIASES = {
    "dialogue": "dialogue",
    "dialog": "dialogue",
    "dx": "dialogue",
    "voice": "dialogue",
    "music": "music",
    "mx": "music",
    "effects": "effects",
    "effect": "effects",
    "sfx": "effects",
    "fx": "effects",
}


def _require(mapping: Dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Missing required configuration field",
            {"field": "{}.{}".format(context, key) if context else key},
        )
    return mapping[key]


def _number(
    value: Any,
    field: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MasterError(
            "E_CONFIG_SCHEMA", "Configuration value must be numeric", {"field": field}
        )
    result = float(value)
    if minimum is not None and result < minimum:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Configuration value is below its minimum",
            {"field": field, "value": result, "minimum": minimum},
        )
    if maximum is not None and result > maximum:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Configuration value is above its maximum",
            {"field": field, "value": result, "maximum": maximum},
        )
    return result


def _integer(
    value: Any,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MasterError(
            "E_CONFIG_SCHEMA", "Configuration value must be an integer", {"field": field}
        )
    if minimum is not None and value < minimum:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Configuration value is below its minimum",
            {"field": field, "value": value, "minimum": minimum},
        )
    if maximum is not None and value > maximum:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Configuration value is above its maximum",
            {"field": field, "value": value, "maximum": maximum},
        )
    return value


def locate_config(input_dir: str, explicit: Optional[str]) -> str:
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(input_dir, explicit)
        if not os.path.isfile(path):
            raise MasterError(
                "E_CONFIG_MISSING",
                "Configuration file does not exist",
                {"config": path},
            )
        return os.path.abspath(path)

    default = os.path.join(input_dir, "config.json")
    if os.path.isfile(default):
        return os.path.abspath(default)

    try:
        candidates = sorted(
            name
            for name in os.listdir(input_dir)
            if name.lower().endswith(".json")
            and not name.lower().endswith("_manifest.json")
        )
    except OSError as exc:
        raise MasterError(
            "E_CONFIG_MISSING",
            "Unable to inspect the input directory for configuration files",
            {"input_dir": os.path.abspath(input_dir), "error": str(exc)},
        )
    if not candidates:
        raise MasterError(
            "E_CONFIG_MISSING",
            "No JSON configuration file found in the input directory",
            {"input_dir": os.path.abspath(input_dir)},
        )
    if len(candidates) > 1:
        raise MasterError(
            "E_CONFIG_MISSING",
            "Multiple JSON files found; name the mastering config config.json",
            {"candidates": candidates},
        )
    return os.path.abspath(os.path.join(input_dir, candidates[0]))


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise MasterError(
            "E_CONFIG_SCHEMA", "Configuration section must be an object", {"field": name}
        )
    return value


def _first(mapping: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _normalize_track(
    raw_track: Any, index: int, input_dir: str
) -> Dict[str, Any]:
    field_prefix = "tracks[{}]".format(index)
    if not isinstance(raw_track, dict):
        raise MasterError(
            "E_CONFIG_SCHEMA", "Each track must be an object", {"field": field_prefix}
        )
    filename = _first(raw_track, ["file", "path", "filename"])
    if not isinstance(filename, str) or not filename.strip():
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Track file is required",
            {"field": field_prefix + ".file"},
        )
    raw_group = _first(raw_track, ["group", "bus", "stem"])
    if not isinstance(raw_group, str):
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Track group is required",
            {"field": field_prefix + ".group"},
        )
    group = GROUP_ALIASES.get(raw_group.strip().lower())
    if group is None:
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "Unsupported track group",
            {
                "field": field_prefix + ".group",
                "group": raw_group,
                "supported": ["dialogue", "music", "effects"],
            },
        )

    channel_map = raw_track.get("channel_map")
    normalized_map: Optional[Any] = None
    if channel_map is not None:
        if isinstance(channel_map, list):
            normalized_map = [
                normalize_role(item, allow_none=True) for item in channel_map
            ]
        elif isinstance(channel_map, dict):
            normalized_map = {}
            for role, source_index in channel_map.items():
                normalized_role = normalize_role(role)
                assert normalized_role is not None
                normalized_map[normalized_role] = _integer(
                    source_index,
                    field_prefix + ".channel_map." + role,
                    minimum=0,
                )
        else:
            raise MasterError(
                "E_CONFIG_SCHEMA",
                "channel_map must be a list or an object",
                {"field": field_prefix + ".channel_map"},
            )

    source_layout = raw_track.get("input_layout", raw_track.get("layout"))
    resolved_layout = None
    if source_layout is not None:
        if not isinstance(source_layout, list):
            raise MasterError(
                "E_CONFIG_SCHEMA",
                "input_layout must be a list of channel roles",
                {"field": field_prefix + ".input_layout"},
            )
        resolved_layout = [normalize_role(item) for item in source_layout]

    gain_db = _number(
        raw_track.get("gain_db", 0.0),
        field_prefix + ".gain_db",
        minimum=-96.0,
        maximum=24.0,
    )
    return {
        "file": filename.strip(),
        "absolute_path": os.path.abspath(os.path.join(input_dir, filename.strip())),
        "group": group,
        "channel_map": normalized_map,
        "input_layout": resolved_layout,
        "gain_db": gain_db,
    }


def normalize_config(raw: Dict[str, Any], input_dir: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise MasterError(
            "E_CONFIG_SCHEMA", "The root of the JSON configuration must be an object"
        )

    target_sr = _integer(
        _first(raw, ["target_sample_rate", "sample_rate"]),
        "target_sample_rate",
        minimum=8000,
        maximum=384000,
    )
    layout_name, output_roles, output_mask = resolve_output_layout(
        _first(raw, ["output_layout", "layout"])
    )

    loudness_section = _section(raw, "loudness")
    target_lufs = _number(
        _first(raw, ["target_lufs"], loudness_section.get("target_lufs")),
        "target_lufs",
        minimum=-70.0,
        maximum=0.0,
    )
    true_peak_limit = _number(
        _first(
            raw,
            ["true_peak_limit_dbtp", "true_peak_limit"],
            loudness_section.get(
                "true_peak_limit_dbtp", loudness_section.get("true_peak_limit")
            ),
        ),
        "true_peak_limit_dbtp",
        minimum=-30.0,
        maximum=-0.01,
    )
    max_relative_gain = _number(
        _first(
            raw,
            [
                "max_relative_gain_db",
                "max_gain_offset_db",
                "gain_offset_limit_db",
            ],
            loudness_section.get(
                "max_relative_gain_db",
                loudness_section.get("max_gain_offset_db"),
            ),
        ),
        "max_relative_gain_db",
        minimum=0.0,
        maximum=60.0,
    )
    tolerance_lu = _number(
        loudness_section.get("tolerance_lu", 0.5),
        "loudness.tolerance_lu",
        minimum=0.01,
        maximum=2.0,
    )
    normalization_iterations = _integer(
        loudness_section.get(
            "iteration_limit", loudness_section.get("normalization_iteration_limit", 8)
        ),
        "loudness.iteration_limit",
        minimum=1,
        maximum=30,
    )
    max_normalization_gain = loudness_section.get("max_gain_db")
    if max_normalization_gain is not None:
        max_normalization_gain = _number(
            max_normalization_gain,
            "loudness.max_gain_db",
            minimum=0.0,
            maximum=60.0,
        )

    compression_raw = _section(raw, "compression")
    compression = {
        "enabled": bool(compression_raw.get("enabled", True)),
        "threshold_dbfs": _number(
            _first(compression_raw, ["threshold_dbfs", "threshold"]),
            "compression.threshold_dbfs",
            minimum=-96.0,
            maximum=0.0,
        ),
        "ratio": _number(
            _require(compression_raw, "ratio", "compression"),
            "compression.ratio",
            minimum=1.0,
            maximum=40.0,
        ),
        "attack_ms": _number(
            _require(compression_raw, "attack_ms", "compression"),
            "compression.attack_ms",
            minimum=0.1,
            maximum=2000.0,
        ),
        "release_ms": _number(
            _require(compression_raw, "release_ms", "compression"),
            "compression.release_ms",
            minimum=1.0,
            maximum=10000.0,
        ),
        "knee_db": _number(
            compression_raw.get("knee_db", 6.0),
            "compression.knee_db",
            minimum=0.0,
            maximum=60.0,
        ),
    }

    detection_raw = _section(raw, "dialogue_detection")
    detection = {
        "frame_ms": _number(
            detection_raw.get("frame_ms", 400.0),
            "dialogue_detection.frame_ms",
            minimum=10.0,
            maximum=5000.0,
        ),
        "overlap": _number(
            detection_raw.get("overlap", 0.75),
            "dialogue_detection.overlap",
            minimum=0.0,
            maximum=0.99,
        ),
        "threshold_dbfs": _number(
            detection_raw.get("threshold_dbfs", -45.0),
            "dialogue_detection.threshold_dbfs",
            minimum=-160.0,
            maximum=0.0,
        ),
        "min_segment_ms": _number(
            detection_raw.get("min_segment_ms", 200.0),
            "dialogue_detection.min_segment_ms",
            minimum=0.0,
            maximum=10000.0,
        ),
    }

    limiter_raw = _section(raw, "limiter")
    limiter = {
        "lookahead_ms": _number(
            limiter_raw.get("lookahead_ms", 10.0),
            "limiter.lookahead_ms",
            minimum=0.5,
            maximum=200.0,
        ),
        "release_ms": _number(
            limiter_raw.get("release_ms", 100.0),
            "limiter.release_ms",
            minimum=1.0,
            maximum=5000.0,
        ),
        "oversample": _integer(
            limiter_raw.get("oversample", 4),
            "limiter.oversample",
            minimum=2,
            maximum=8,
        ),
        "iteration_limit": _integer(
            limiter_raw.get("iteration_limit", 8),
            "limiter.iteration_limit",
            minimum=1,
            maximum=30,
        ),
        "safety_margin_db": _number(
            limiter_raw.get("safety_margin_db", 0.25),
            "limiter.safety_margin_db",
            minimum=0.0,
            maximum=3.0,
        ),
    }

    output_raw = _section(raw, "output")
    basename = str(output_raw.get("basename", "master")).strip()
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", basename):
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "output.basename may only contain letters, digits, dot, dash and underscore",
            {"basename": basename},
        )
    codec = str(output_raw.get("audio_codec", output_raw.get("codec", "aac"))).lower()
    if codec in ("pcm24", "pcm_s24le", "lpcm"):
        codec = "pcm"
    if codec not in ("aac", "pcm"):
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "output.audio_codec must be aac or pcm",
            {"audio_codec": codec},
        )
    bitrate = str(output_raw.get("aac_bitrate", "256k"))
    if not re.match(r"^[1-9][0-9]*(k|K|m|M)$", bitrate):
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "output.aac_bitrate must look like 192k or 256k",
            {"aac_bitrate": bitrate},
        )
    bits_per_sample = _integer(
        output_raw.get("bits_per_sample", 24),
        "output.bits_per_sample",
        minimum=16,
        maximum=24,
    )
    if bits_per_sample not in (16, 24):
        raise MasterError(
            "E_CONFIG_SCHEMA",
            "output.bits_per_sample must be 16 or 24",
            {"bits_per_sample": bits_per_sample},
        )
    duration_tolerance_ms = output_raw.get("duration_tolerance_ms")
    if duration_tolerance_ms is not None:
        duration_tolerance_ms = _number(
            duration_tolerance_ms,
            "output.duration_tolerance_ms",
            minimum=0.0,
            maximum=500.0,
        )
    output = {
        "basename": basename,
        "container": "mp4",
        "audio_codec": codec,
        "aac_bitrate": bitrate,
        "bits_per_sample": bits_per_sample,
        "duration_tolerance_ms": duration_tolerance_ms,
        "write_metadata": bool(output_raw.get("write_metadata", True)),
    }

    raw_tracks = _first(raw, ["tracks", "stems"])
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise MasterError(
            "E_CONFIG_SCHEMA", "tracks must be a non-empty list", {"field": "tracks"}
        )
    tracks = [_normalize_track(item, i, input_dir) for i, item in enumerate(raw_tracks)]
    if not any(track["group"] == "dialogue" for track in tracks):
        raise MasterError(
            "E_CONFIG_SCHEMA", "At least one dialogue track is required"
        )

    return {
        "target_sample_rate": target_sr,
        "output_layout_name": layout_name,
        "output_roles": output_roles,
        "output_channel_mask": output_mask,
        "target_lufs": target_lufs,
        "true_peak_limit_dbtp": true_peak_limit,
        "max_relative_gain_db": max_relative_gain,
        "loudness_tolerance_lu": tolerance_lu,
        "normalization_iteration_limit": normalization_iterations,
        "max_normalization_gain_db": max_normalization_gain,
        "compression": compression,
        "dialogue_detection": detection,
        "limiter": limiter,
        "output": output,
        "tracks": tracks,
    }


def load_config(
    input_dir: str, explicit_config: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    config_path = locate_config(input_dir, explicit_config)
    try:
        with open(config_path, "rb") as handle:
            raw_bytes = handle.read()
    except OSError as exc:
        raise MasterError(
            "E_CONFIG_MISSING",
            "Unable to read configuration file",
            {"config": config_path, "error": str(exc)},
        )
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MasterError(
            "E_CONFIG_INVALID_JSON",
            "Configuration is not valid UTF-8 JSON",
            {"config": config_path, "error": str(exc)},
        )
    normalized = normalize_config(raw, input_dir)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return raw, normalized, config_path, digest

