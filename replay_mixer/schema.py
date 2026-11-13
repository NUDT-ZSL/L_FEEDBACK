from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    AppError,
    channel_count_error,
    invalid_config,
    missing_input,
    unsupported_bit_depth,
    unsupported_sample_rate,
)
from .util import (
    canonical_json,
    is_int,
    is_number,
    resolve_relative_path,
    sha256_bytes,
)


SUPPORTED_SAMPLE_RATES = frozenset(
    {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000}
)
SUPPORTED_OUTPUT_BIT_DEPTHS = frozenset({16})
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Anchor:
    replay_ms: float
    source_ms: float


@dataclass(frozen=True)
class SilenceSpec:
    window_ms: float
    threshold_dbfs: float
    min_duration_ms: float


@dataclass(frozen=True)
class TrackSpec:
    role: str
    name: str
    path: Path
    path_value: str
    channels: int
    start_ms: float
    end_ms: float
    gain_db: float
    pan: float
    anchors: tuple[Anchor, ...] = ()
    silence: SilenceSpec | None = None


@dataclass(frozen=True)
class MixSpec:
    target_rms_dbfs: float
    max_gain_db: float
    peak_limit_dbfs: float
    soft_knee_ratio: float
    block_frames: int


@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    base_dir: Path
    version: int
    sample_rate: int
    bit_depth: int
    output_channels: int
    output_path: Path
    output_path_value: str
    duration_ms: float
    frame_timestamps_ms: tuple[float, ...]
    mix: MixSpec
    commentary: tuple[TrackSpec, ...]
    engine: TrackSpec
    ambience: TrackSpec
    raw: dict[str, Any]

    @property
    def total_frames(self) -> int:
        return max(1, int(round(self.duration_ms * self.sample_rate / 1000.0)))


@dataclass(frozen=True)
class InputMetadata:
    role: str
    name: str
    path_value: str
    sha256: str
    wav_bytes: int
    sample_rate: int
    channels: int
    frames: int


class _DuplicateJSONKey(ValueError):
    pass


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise missing_input(str(path), "configuration")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise invalid_config("cannot read configuration file", path=str(path), error=str(exc)) from None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise invalid_config("configuration is not valid JSON", path=str(path), error=str(exc)) from None
    return _obj(value, "configuration")


def _obj(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid_config(f"{context} must be a JSON object", field=context)
    return value


def _check_keys(
    value: dict[str, Any],
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise invalid_config(
            f"{context} is missing required field(s)",
            field=context,
            missing=missing,
        )
    if unknown:
        raise invalid_config(
            f"{context} contains unknown field(s)",
            field=context,
            unknown=unknown,
        )


def _num(value: Any, context: str) -> float:
    if not is_number(value):
        raise invalid_config(f"{context} must be a finite number", field=context)
    return float(value)


def _integer(value: Any, context: str) -> int:
    if not is_int(value):
        raise invalid_config(f"{context} must be an integer", field=context)
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise invalid_config(f"{context} must be a non-empty string", field=context)
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise invalid_config(f"{context} must be an array", field=context)
    return value


def _parse_anchor(value: Any, context: str) -> Anchor:
    obj = _obj(value, context)
    _check_keys(obj, context, {"replay_ms", "source_ms"})
    replay_ms = _num(obj["replay_ms"], f"{context}.replay_ms")
    source_ms = _num(obj["source_ms"], f"{context}.source_ms")
    if source_ms < 0.0:
        raise invalid_config(f"{context}.source_ms must be non-negative", field=context)
    return Anchor(replay_ms=replay_ms, source_ms=source_ms)


def _parse_silence(value: Any, context: str) -> SilenceSpec:
    obj = _obj(value, context)
    _check_keys(obj, context, {"window_ms", "threshold_dbfs", "min_duration_ms"})
    window_ms = _num(obj["window_ms"], f"{context}.window_ms")
    threshold_dbfs = _num(obj["threshold_dbfs"], f"{context}.threshold_dbfs")
    min_duration_ms = _num(obj["min_duration_ms"], f"{context}.min_duration_ms")
    if window_ms <= 0.0:
        raise invalid_config(f"{context}.window_ms must be positive", field=context)
    if not (-160.0 <= threshold_dbfs <= 0.0):
        raise invalid_config(
            f"{context}.threshold_dbfs must be within [-160, 0]",
            field=context,
            value=threshold_dbfs,
        )
    if min_duration_ms < 0.0:
        raise invalid_config(f"{context}.min_duration_ms must be non-negative", field=context)
    return SilenceSpec(
        window_ms=window_ms,
        threshold_dbfs=threshold_dbfs,
        min_duration_ms=min_duration_ms,
    )


def _parse_track(
    value: Any,
    role: str,
    context: str,
    base_dir: Path,
    duration_ms: float,
) -> TrackSpec:
    obj = _obj(value, context)
    required = {"name", "path", "channels", "start_ms", "end_ms", "gain_db", "pan"}
    optional: set[str] = set()
    if role == "commentary":
        required |= {"anchors", "silence"}
    _check_keys(obj, context, required, optional)

    name = _string(obj["name"], f"{context}.name")
    raw_path = obj["path"]
    track_path = resolve_relative_path(base_dir, raw_path, f"{context}.path")
    channels = _integer(obj["channels"], f"{context}.channels")
    if channels not in (1, 2):
        raise channel_count_error(
            "track channel count must be 1 or 2",
            field=f"{context}.channels",
            value=channels,
        )

    start_ms = _num(obj["start_ms"], f"{context}.start_ms")
    end_ms = _num(obj["end_ms"], f"{context}.end_ms")
    if start_ms < 0.0:
        raise invalid_config(f"{context}.start_ms must be non-negative", field=context)
    if end_ms > duration_ms:
        raise invalid_config(
            f"{context}.end_ms must not exceed replay duration_ms",
            field=context,
            end_ms=end_ms,
            duration_ms=duration_ms,
        )
    if end_ms <= start_ms:
        raise invalid_config(f"{context}.end_ms must be greater than start_ms", field=context)

    gain_db = _num(obj["gain_db"], f"{context}.gain_db")
    if not (-120.0 <= gain_db <= 60.0):
        raise invalid_config(
            f"{context}.gain_db must be within [-120, 60]",
            field=context,
            value=gain_db,
        )
    pan = _num(obj["pan"], f"{context}.pan")
    if not (-1.0 <= pan <= 1.0):
        raise invalid_config(f"{context}.pan must be within [-1, 1]", field=context, value=pan)

    anchors: tuple[Anchor, ...] = ()
    silence: SilenceSpec | None = None
    if role == "commentary":
        anchor_values = _list(obj["anchors"], f"{context}.anchors")
        if len(anchor_values) < 2:
            raise invalid_config(f"{context}.anchors must contain at least two anchors", field=context)
        anchors = tuple(
            _parse_anchor(item, f"{context}.anchors[{index}]")
            for index, item in enumerate(anchor_values)
        )
        for index in range(1, len(anchors)):
            previous = anchors[index - 1]
            current = anchors[index]
            if not current.replay_ms > previous.replay_ms:
                raise invalid_config(
                    f"{context}.anchors replay_ms must be strictly increasing",
                    field=context,
                    index=index,
                )
            if not current.source_ms > previous.source_ms:
                raise invalid_config(
                    f"{context}.anchors source_ms must be strictly increasing",
                    field=context,
                    index=index,
                )
        for index, anchor in enumerate(anchors):
            if not (start_ms <= anchor.replay_ms <= end_ms):
                raise invalid_config(
                    f"{context}.anchors[{index}].replay_ms must lie within track start/end",
                    field=context,
                    index=index,
                )
        silence = _parse_silence(obj["silence"], f"{context}.silence")

    return TrackSpec(
        role=role,
        name=name,
        path=track_path,
        path_value=raw_path,
        channels=channels,
        start_ms=start_ms,
        end_ms=end_ms,
        gain_db=gain_db,
        pan=pan,
        anchors=anchors,
        silence=silence,
    )


def load_config(path_value: str) -> ProjectConfig:
    source_path = Path(path_value).resolve()
    raw = _load_json(source_path)
    _check_keys(
        raw,
        "configuration",
        {
            "version",
            "sample_rate",
            "bit_depth",
            "channels",
            "output_path",
            "replay",
            "mix",
            "tracks",
        },
    )

    version = _integer(raw["version"], "version")
    if version != SCHEMA_VERSION:
        raise invalid_config("unsupported configuration version", version=version)

    sample_rate = _integer(raw["sample_rate"], "sample_rate")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise unsupported_sample_rate(
            "sample rate is not supported",
            field="sample_rate",
            value=sample_rate,
            supported=sorted(SUPPORTED_SAMPLE_RATES),
        )

    bit_depth = _integer(raw["bit_depth"], "bit_depth")
    if bit_depth not in SUPPORTED_OUTPUT_BIT_DEPTHS:
        raise unsupported_bit_depth(
            "only 16-bit PCM output is supported",
            field="bit_depth",
            value=bit_depth,
        )

    output_channels = _integer(raw["channels"], "channels")
    if output_channels not in (1, 2):
        raise channel_count_error(
            "output channel count must be 1 or 2",
            field="channels",
            value=output_channels,
        )

    base_dir = source_path.parent.resolve()
    output_path_value = raw["output_path"]
    output_path = resolve_relative_path(base_dir, output_path_value, "output_path")
    if output_path.suffix.lower() != ".wav":
        raise invalid_config("output_path must end with .wav", field="output_path")

    replay = _obj(raw["replay"], "replay")
    _check_keys(replay, "replay", {"duration_ms", "frame_timestamps_ms"})
    duration_ms = _num(replay["duration_ms"], "replay.duration_ms")
    if duration_ms <= 0.0:
        raise invalid_config("replay.duration_ms must be positive", field="replay.duration_ms")
    timestamps_raw = _list(replay["frame_timestamps_ms"], "replay.frame_timestamps_ms")
    if len(timestamps_raw) < 2:
        raise invalid_config(
            "replay.frame_timestamps_ms must contain at least two timestamps",
            field="replay.frame_timestamps_ms",
        )
    timestamps = tuple(
        _num(item, f"replay.frame_timestamps_ms[{index}]")
        for index, item in enumerate(timestamps_raw)
    )
    if timestamps[0] < 0.0:
        raise invalid_config("frame timestamps must be non-negative", field="replay.frame_timestamps_ms")
    for index in range(1, len(timestamps)):
        if not timestamps[index] > timestamps[index - 1]:
            raise invalid_config(
                "frame timestamps must be strictly increasing",
                field="replay.frame_timestamps_ms",
                index=index,
            )
    if timestamps[-1] > duration_ms:
        raise invalid_config(
            "frame timestamp exceeds replay.duration_ms",
            field="replay.frame_timestamps_ms",
            timestamp_ms=timestamps[-1],
            duration_ms=duration_ms,
        )

    mix = _obj(raw["mix"], "mix")
    _check_keys(
        mix,
        "mix",
        {
            "target_rms_dbfs",
            "max_gain_db",
            "peak_limit_dbfs",
            "soft_knee_ratio",
            "block_frames",
        },
    )
    target_rms_dbfs = _num(mix["target_rms_dbfs"], "mix.target_rms_dbfs")
    if not (-160.0 <= target_rms_dbfs <= 0.0):
        raise invalid_config(
            "mix.target_rms_dbfs must be within [-160, 0]",
            field="mix.target_rms_dbfs",
            value=target_rms_dbfs,
        )
    max_gain_db = _num(mix["max_gain_db"], "mix.max_gain_db")
    if not (0.0 <= max_gain_db <= 120.0):
        raise invalid_config(
            "mix.max_gain_db must be within [0, 120]",
            field="mix.max_gain_db",
            value=max_gain_db,
        )
    peak_limit_dbfs = _num(mix["peak_limit_dbfs"], "mix.peak_limit_dbfs")
    if not (-60.0 <= peak_limit_dbfs <= 0.0):
        raise invalid_config(
            "mix.peak_limit_dbfs must be within [-60, 0]",
            field="mix.peak_limit_dbfs",
            value=peak_limit_dbfs,
        )
    soft_knee_ratio = _num(mix["soft_knee_ratio"], "mix.soft_knee_ratio")
    if not (0.5 <= soft_knee_ratio < 1.0):
        raise invalid_config(
            "mix.soft_knee_ratio must be within [0.5, 1.0)",
            field="mix.soft_knee_ratio",
            value=soft_knee_ratio,
        )
    block_frames = _integer(mix["block_frames"], "mix.block_frames")
    if not (1 <= block_frames <= 1048576):
        raise invalid_config(
            "mix.block_frames must be within [1, 1048576]",
            field="mix.block_frames",
            value=block_frames,
        )

    tracks = _obj(raw["tracks"], "tracks")
    _check_keys(tracks, "tracks", {"commentary", "engine", "ambience"})
    commentary_values = _list(tracks["commentary"], "tracks.commentary")
    if not commentary_values:
        raise invalid_config("tracks.commentary must contain at least one track", field="tracks.commentary")
    commentary = tuple(
        _parse_track(
            item,
            "commentary",
            f"tracks.commentary[{index}]",
            base_dir,
            duration_ms,
        )
        for index, item in enumerate(commentary_values)
    )
    engine = _parse_track(tracks["engine"], "engine", "tracks.engine", base_dir, duration_ms)
    ambience = _parse_track(tracks["ambience"], "ambience", "tracks.ambience", base_dir, duration_ms)

    names: set[str] = set()
    for track in (*commentary, engine, ambience):
        if track.name in names:
            raise invalid_config("track names must be unique", name=track.name)
        names.add(track.name)

    all_inputs = {
        os.path.normcase(str(track.path.resolve()))
        for track in (*commentary, engine, ambience)
    }
    if os.path.normcase(str(output_path.resolve())) in all_inputs:
        raise invalid_config(
            "output_path must not overwrite an input WAV",
            field="output_path",
            value=output_path_value,
        )

    mix_spec = MixSpec(
        target_rms_dbfs=target_rms_dbfs,
        max_gain_db=max_gain_db,
        peak_limit_dbfs=peak_limit_dbfs,
        soft_knee_ratio=soft_knee_ratio,
        block_frames=block_frames,
    )
    return ProjectConfig(
        source_path=source_path,
        base_dir=base_dir,
        version=version,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        output_channels=output_channels,
        output_path=output_path,
        output_path_value=output_path_value,
        duration_ms=duration_ms,
        frame_timestamps_ms=timestamps,
        mix=mix_spec,
        commentary=commentary,
        engine=engine,
        ambience=ambience,
        raw=raw,
    )


def build_config_digest(config: ProjectConfig, inputs: list[InputMetadata]) -> str:
    payload = {
        "algorithm": "replay-mixer-stdlib-v1",
        "config": config.raw,
        "inputs": [
            {
                "role": item.role,
                "name": item.name,
                "path": item.path_value,
                "sha256": item.sha256,
                "wav_bytes": item.wav_bytes,
                "sample_rate": item.sample_rate,
                "channels": item.channels,
                "frames": item.frames,
            }
            for item in inputs
        ],
    }
    return sha256_bytes(canonical_json(payload))
