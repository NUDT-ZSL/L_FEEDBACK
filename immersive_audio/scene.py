from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .constants import (
    MAX_FRAME_LENGTH,
    MAX_OBJECTS,
    MAX_SAMPLE_RATE,
    MIN_FRAME_LENGTH,
    MIN_OBJECTS,
    MIN_SAMPLE_RATE,
)
from .errors import (
    SCENE_DUPLICATE_ID,
    SCENE_DURATION,
    SCENE_FRAME_LENGTH,
    SCENE_INVALID_JSON,
    SCENE_NOT_FOUND,
    SCENE_OBJECT_COUNT,
    SCENE_SAMPLE_RATE,
    SCENE_SCHEMA,
    SCENE_TIMESTAMPS,
    PipelineError,
)


@dataclass(frozen=True)
class Keyframe:
    time: Decimal
    azimuth: float
    elevation: float
    distance: float
    gain: float
    size: float


@dataclass(frozen=True)
class AudioClip:
    path: Path
    display_path: str
    start: Decimal
    loops: int


@dataclass(frozen=True)
class AudioObject:
    index: int
    object_id: str
    keyframes: tuple[Keyframe, ...]
    clips: tuple[AudioClip, ...]


@dataclass(frozen=True)
class Scene:
    path: Path
    sample_rate: int
    frame_length: int
    objects: tuple[AudioObject, ...]
    duration: Decimal | None

    @property
    def object_count(self) -> int:
        return len(self.objects)


@dataclass(frozen=True)
class _PositionPoint:
    time: Decimal
    azimuth: float
    elevation: float
    distance: float
    size: float


@dataclass(frozen=True)
class _GainPoint:
    time: Decimal
    gain: float


def _schema(message: str) -> PipelineError:
    return PipelineError(SCENE_SCHEMA, message)


def _as_dict(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _schema(f"{location} must be an object")
    return value


def _as_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise _schema(f"{location} must be an array")
    return value


def _pick(data: dict[str, Any], names: tuple[str, ...], location: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise _schema(f"{location} is required")


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise _schema(f"{location} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _schema(f"{location} must be a finite number") from exc
    if not math.isfinite(result):
        raise _schema(f"{location} must be a finite number")
    return result


def _decimal_time(value: Any, location: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise _schema(f"{location} must be a JSON number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise _schema(f"{location} must be a JSON number") from exc
    if not result.is_finite() or result < 0:
        raise _schema(f"{location} must be a finite non-negative number")
    return result


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool):
        raise _schema(f"{location} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    raise _schema(f"{location} must be an integer")


def _position_point(raw: Any, location: str) -> _PositionPoint:
    point = _as_dict(raw, location)
    time = _decimal_time(_pick(point, ("time", "time_seconds", "t"), location), f"{location}.time")
    azimuth = _finite_number(
        _pick(point, ("azimuth", "azimuth_deg", "az"), location), f"{location}.azimuth"
    )
    elevation = _finite_number(
        _pick(point, ("elevation", "elevation_deg", "el"), location),
        f"{location}.elevation",
    )
    distance = _finite_number(
        _pick(point, ("distance", "radius"), location), f"{location}.distance"
    )
    size = _finite_number(point.get("size", 0.0), f"{location}.size")
    if distance < 0:
        raise _schema(f"{location}.distance must be non-negative")
    if size < 0:
        raise _schema(f"{location}.size must be non-negative")
    return _PositionPoint(time, azimuth, elevation, distance, size)


def _gain_point(raw: Any, location: str) -> _GainPoint:
    point = _as_dict(raw, location)
    time = _decimal_time(_pick(point, ("time", "time_seconds", "t"), location), f"{location}.time")
    gain = _finite_number(_pick(point, ("gain", "value"), location), f"{location}.gain")
    return _GainPoint(time, gain)


def _validate_monotonic(times: list[Decimal], location: str) -> None:
    for previous, current in zip(times, times[1:]):
        if current <= previous:
            raise PipelineError(
                SCENE_TIMESTAMPS,
                f"{location} timestamps must be strictly increasing",
            )


def _linear_points(times: list[Decimal], values: list[float], at: Decimal) -> float:
    if at <= times[0]:
        return values[0]
    if at >= times[-1]:
        return values[-1]
    lo = 0
    hi = len(times) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if times[mid] <= at:
            lo = mid
        else:
            hi = mid
    span = times[hi] - times[lo]
    ratio = float((at - times[lo]) / span)
    return values[lo] + (values[hi] - values[lo]) * ratio


def _parse_object(raw: Any, index: int, base_dir: Path) -> AudioObject:
    location = f"objects[{index}]"
    obj = _as_dict(raw, location)
    object_id = obj.get("id", obj.get("name", f"object_{index}"))
    if not isinstance(object_id, str) or not object_id.strip():
        raise _schema(f"{location}.id must be a non-empty string")

    position_raw = obj.get("keyframes")
    if position_raw is None:
        position_raw = obj.get("position_keyframes", obj.get("positions"))
    if position_raw is None:
        raise _schema(f"{location}.keyframes is required")
    position_list = _as_list(position_raw, f"{location}.keyframes")
    if not position_list:
        raise _schema(f"{location}.keyframes must not be empty")
    positions = [
        _position_point(point, f"{location}.keyframes[{point_index}]")
        for point_index, point in enumerate(position_list)
    ]
    _validate_monotonic([point.time for point in positions], f"{location}.keyframes")

    gain_raw = obj.get("gain_curve")
    gains: list[_GainPoint] = []
    if gain_raw is not None:
        gain_list = _as_list(gain_raw, f"{location}.gain_curve")
        if not gain_list:
            raise _schema(f"{location}.gain_curve must not be empty")
        gains = [
            _gain_point(point, f"{location}.gain_curve[{point_index}]")
            for point_index, point in enumerate(gain_list)
        ]
        _validate_monotonic([point.time for point in gains], f"{location}.gain_curve")

    position_times = [point.time for point in positions]
    position_azimuths = [point.azimuth for point in positions]
    position_elevations = [point.elevation for point in positions]
    position_distances = [point.distance for point in positions]
    position_sizes = [point.size for point in positions]

    keyframes: list[Keyframe] = []
    if gains:
        gain_times = [point.time for point in gains]
        gain_values = [point.gain for point in gains]
        all_times = sorted(set(position_times + gain_times))
        for time in all_times:
            keyframes.append(
                Keyframe(
                    time=time,
                    azimuth=_linear_points(position_times, position_azimuths, time),
                    elevation=_linear_points(position_times, position_elevations, time),
                    distance=_linear_points(position_times, position_distances, time),
                    gain=_linear_points(gain_times, gain_values, time),
                    size=_linear_points(position_times, position_sizes, time),
                )
            )
    else:
        for point_index, point in enumerate(positions):
            point_dict = _as_dict(
                position_list[point_index], f"{location}.keyframes[{point_index}]"
            )
            gain = _finite_number(
                point_dict.get("gain", 1.0),
                f"{location}.keyframes[{point_index}].gain",
            )
            keyframes.append(
                Keyframe(
                    time=point.time,
                    azimuth=point.azimuth,
                    elevation=point.elevation,
                    distance=point.distance,
                    gain=gain,
                    size=point.size,
                )
            )

    clips_raw = obj.get("clips", [])
    clips_list = _as_list(clips_raw, f"{location}.clips")
    clips: list[AudioClip] = []
    for clip_index, raw_clip in enumerate(clips_list):
        clip_location = f"{location}.clips[{clip_index}]"
        clip = _as_dict(raw_clip, clip_location)
        display_path = _pick(clip, ("path", "file"), clip_location)
        if not isinstance(display_path, str) or not display_path:
            raise _schema(f"{clip_location}.path must be a non-empty string")
        clip_path = Path(display_path)
        if not clip_path.is_absolute():
            clip_path = base_dir / clip_path
        start = _decimal_time(
            clip.get("start", clip.get("start_time", clip.get("offset", 0.0))),
            f"{clip_location}.start",
        )
        loops = _integer(
            clip.get("loops", clip.get("loop_count", clip.get("repeat", 1))),
            f"{clip_location}.loops",
        )
        if loops < 1:
            raise _schema(f"{clip_location}.loops must be at least 1")
        clips.append(
            AudioClip(
                path=clip_path.resolve(),
                display_path=display_path.replace("\\", "/"),
                start=start,
                loops=loops,
            )
        )
    return AudioObject(index, object_id, tuple(keyframes), tuple(clips))


def load_scene(path: str | Path) -> Scene:
    scene_path = Path(path)
    if not scene_path.is_file():
        raise PipelineError(SCENE_NOT_FOUND, f"scene file does not exist: {scene_path}")
    try:
        text = scene_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise PipelineError(SCENE_NOT_FOUND, f"cannot read scene file: {scene_path}") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        raw = json.loads(
            text,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(SCENE_INVALID_JSON, f"invalid JSON: {exc}") from exc

    root = _as_dict(raw, "scene")
    sample_rate = _integer(
        _pick(root, ("sample_rate", "sampleRate"), "scene"), "scene.sample_rate"
    )
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise PipelineError(
            SCENE_SAMPLE_RATE,
            f"sample_rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}",
        )

    frame_length = _integer(
        _pick(root, ("frame_length", "frameLength"), "scene"), "scene.frame_length"
    )
    if not MIN_FRAME_LENGTH <= frame_length <= MAX_FRAME_LENGTH:
        raise PipelineError(
            SCENE_FRAME_LENGTH,
            f"frame_length must be between {MIN_FRAME_LENGTH} and {MAX_FRAME_LENGTH}",
        )

    objects_raw = _as_list(_pick(root, ("objects",), "scene"), "scene.objects")
    if not MIN_OBJECTS <= len(objects_raw) <= MAX_OBJECTS:
        raise PipelineError(
            SCENE_OBJECT_COUNT,
            f"object count must be between {MIN_OBJECTS} and {MAX_OBJECTS}",
        )

    base_dir = scene_path.resolve().parent
    objects = tuple(
        _parse_object(raw_object, index, base_dir)
        for index, raw_object in enumerate(objects_raw)
    )
    identifiers = [obj.object_id for obj in objects]
    if len(set(identifiers)) != len(identifiers):
        raise PipelineError(SCENE_DUPLICATE_ID, "object ids must be unique")

    duration_raw = root.get("duration_seconds", root.get("duration"))
    duration = None
    if duration_raw is not None:
        try:
            duration = _decimal_time(duration_raw, "scene.duration_seconds")
        except PipelineError as exc:
            raise PipelineError(SCENE_DURATION, exc.message) from exc

    return Scene(
        path=scene_path.resolve(),
        sample_rate=sample_rate,
        frame_length=frame_length,
        objects=objects,
        duration=duration,
    )
