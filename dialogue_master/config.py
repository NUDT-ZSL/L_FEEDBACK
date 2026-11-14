from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


SUPPORTED_BIT_DEPTHS = (8, 16, 24, 32)

# Standard interleaved channel order used by WAV/WAVEX for each supported
# layout. WAVEX carries these as a channel mask via libsndfile.
CHANNEL_LAYOUTS: Mapping[str, Tuple[str, ...]] = {
    "mono": ("FC",),
    "stereo": ("FL", "FR"),
    "2.1": ("FL", "FR", "LFE"),
    "3.0": ("FL", "FR", "FC"),
    "quad": ("FL", "FR", "BL", "BR"),
    "4.0": ("FL", "FR", "BL", "BR"),
    "5.0": ("FL", "FR", "FC", "BL", "BR"),
    "5.1": ("FL", "FR", "FC", "LFE", "BL", "BR"),
    "7.1": ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"),
}

DEFAULT_LAYOUTS = {
    1: "mono",
    2: "stereo",
    3: "3.0",
    4: "quad",
    5: "5.0",
    6: "5.1",
    8: "7.1",
}


class ConfigError(ValueError):
    """A deterministic configuration error (business code E_CFG)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__("{}: {}".format(field, message))
        self.field = field
        self.message = message


@dataclass(frozen=True)
class ClipConfig:
    index: int
    clip_id: str
    input_path: str
    resolved_path: Path
    start_ms: Any
    target_lufs: float
    fade_in_ms: float
    fade_out_ms: float


@dataclass(frozen=True)
class SessionConfig:
    timeline_ms: Any
    timeline_samples: int
    sample_rate: int
    bit_depth: int
    channels: int
    channel_layout: str
    channel_labels: Tuple[str, ...]
    clips: Tuple[ClipConfig, ...]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any, field: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ConfigError(field, "must be a finite number")
    return float(value)


def _nonnegative_number(value: Any, field: str) -> float:
    number = _number(value, field)
    if number < 0:
        raise ConfigError(field, "must be greater than or equal to 0")
    return number


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(field, "must be an integer")
    return value


def milliseconds_to_samples(value: Any, sample_rate: int, field: str) -> int:
    """Convert milliseconds to a sample index using round-half-up."""

    _number(value, field)
    if float(value) < 0:
        raise ConfigError(field, "must be greater than or equal to 0")
    milliseconds = Decimal(str(value))
    samples = milliseconds * Decimal(sample_rate) / Decimal(1000)
    return int(samples.to_integral_value(rounding=ROUND_HALF_UP))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        raise ConfigError("config", "cannot read file: {}".format(exc))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("config", "invalid UTF-8 JSON: {}".format(exc))
    if not isinstance(value, dict):
        raise ConfigError("config", "top-level value must be an object")
    return value


def load_config(path: Path) -> SessionConfig:
    raw = _load_json(path)

    if "timeline_ms" not in raw:
        raise ConfigError("timeline_ms", "missing required field")
    timeline_ms = raw["timeline_ms"]
    _number(timeline_ms, "timeline_ms")
    if float(timeline_ms) <= 0:
        raise ConfigError("timeline_ms", "must be greater than 0")

    if "sample_rate" not in raw:
        raise ConfigError("sample_rate", "missing required field")
    sample_rate = _integer(raw["sample_rate"], "sample_rate")
    if sample_rate < 1000 or sample_rate > 768000:
        raise ConfigError("sample_rate", "must be between 1000 and 768000")

    if "bit_depth" not in raw:
        raise ConfigError("bit_depth", "missing required field")
    bit_depth = _integer(raw["bit_depth"], "bit_depth")
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ConfigError("bit_depth", "must be one of 8, 16, 24, 32")

    if "channels" not in raw:
        raise ConfigError("channels", "missing required field")
    channels = _integer(raw["channels"], "channels")
    if channels < 1 or channels > 16:
        raise ConfigError("channels", "must be between 1 and 16")

    channel_layout = raw.get("channel_layout")
    if channel_layout is None:
        channel_layout = DEFAULT_LAYOUTS.get(channels)
        if channel_layout is None:
            raise ConfigError(
                "channel_layout",
                "required when channels has no standard default",
            )
    if not isinstance(channel_layout, str) or not channel_layout:
        raise ConfigError("channel_layout", "must be a non-empty string")
    channel_layout = channel_layout.strip().lower()
    labels = CHANNEL_LAYOUTS.get(channel_layout)
    if labels is None:
        raise ConfigError(
            "channel_layout",
            "unsupported layout; use one of {}".format(
                ", ".join(sorted(CHANNEL_LAYOUTS))
            ),
        )
    if len(labels) != channels:
        raise ConfigError(
            "channels",
            "must be {} for layout {}".format(len(labels), channel_layout),
        )

    clips_raw = raw.get("clips")
    if not isinstance(clips_raw, list):
        raise ConfigError("clips", "must be an array")

    timeline_samples = milliseconds_to_samples(
        timeline_ms, sample_rate, "timeline_ms"
    )
    if timeline_samples <= 0:
        raise ConfigError("timeline_ms", "rounds to zero samples")

    config_dir = path.parent
    seen_ids = set()
    clips: List[ClipConfig] = []
    for index, item in enumerate(clips_raw):
        prefix = "clips[{}]".format(index)
        if not isinstance(item, dict):
            raise ConfigError(prefix, "must be an object")

        clip_id = item.get("id")
        if clip_id is None:
            clip_id = "clip_{:04d}".format(index + 1)
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ConfigError(prefix + ".id", "must be a non-empty string")
        clip_id = clip_id.strip()
        if clip_id in seen_ids:
            raise ConfigError(prefix + ".id", "duplicate clip id")
        seen_ids.add(clip_id)

        if "input_path" not in item:
            raise ConfigError(prefix + ".input_path", "missing required field")
        input_path = item["input_path"]
        if not isinstance(input_path, str) or not input_path.strip():
            raise ConfigError(
                prefix + ".input_path", "must be a non-empty string"
            )
        input_path = input_path.strip()
        resolved_path = Path(input_path)
        if not resolved_path.is_absolute():
            resolved_path = config_dir / resolved_path

        if "start_ms" not in item:
            raise ConfigError(prefix + ".start_ms", "missing required field")
        start_ms = item["start_ms"]
        _nonnegative_number(start_ms, prefix + ".start_ms")

        if "target_lufs" not in item:
            raise ConfigError(
                prefix + ".target_lufs", "missing required field"
            )
        target_lufs = _number(
            item["target_lufs"], prefix + ".target_lufs"
        )

        if "fade_in_ms" not in item:
            raise ConfigError(prefix + ".fade_in_ms", "missing required field")
        fade_in_ms = _nonnegative_number(
            item["fade_in_ms"], prefix + ".fade_in_ms"
        )

        if "fade_out_ms" not in item:
            raise ConfigError(prefix + ".fade_out_ms", "missing required field")
        fade_out_ms = _nonnegative_number(
            item["fade_out_ms"], prefix + ".fade_out_ms"
        )

        clips.append(
            ClipConfig(
                index=index,
                clip_id=clip_id,
                input_path=input_path,
                resolved_path=resolved_path,
                start_ms=start_ms,
                target_lufs=target_lufs,
                fade_in_ms=fade_in_ms,
                fade_out_ms=fade_out_ms,
            )
        )

    return SessionConfig(
        timeline_ms=timeline_ms,
        timeline_samples=timeline_samples,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=channels,
        channel_layout=channel_layout,
        channel_labels=labels,
        clips=tuple(clips),
    )
