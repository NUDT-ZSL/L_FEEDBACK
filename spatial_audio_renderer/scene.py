"""Scene JSON parsing, interpolation, and strict timeline validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import ParameterError


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, label: str, minimum: Optional[int] = None) -> int:
    if not _is_int(value):
        raise ParameterError("%s must be an integer" % label)
    if minimum is not None and value < minimum:
        raise ParameterError("%s must be >= %d" % (label, minimum))
    return int(value)


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError("%s must be a finite number" % label)
    result = float(value)
    if not math.isfinite(result):
        raise ParameterError("%s must be a finite number" % label)
    return result


def _require_vec3(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ParameterError("%s must contain exactly three numbers" % label)
    return np.array(
        [_require_float(component, label) for component in value],
        dtype=np.float64,
    )


def _require_vec4(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ParameterError("%s must contain exactly four numbers" % label)
    return np.array(
        [_require_float(component, label) for component in value],
        dtype=np.float64,
    )


def _key_value(item: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in item:
            return item[name]
    return None


def _key_time(item: Dict[str, Any], label: str) -> int:
    value = _key_value(item, ("time", "t", "sample", "frame"))
    if value is None:
        raise ParameterError("%s is missing an integer time" % label)
    return _require_int(value, "%s time" % label, 0)


def _ensure_strict_times(times: Sequence[int], label: str) -> None:
    if not times:
        raise ParameterError("%s must contain at least one key" % label)
    for index in range(1, len(times)):
        if times[index] <= times[index - 1]:
            raise ParameterError(
                "%s timestamps must be strictly increasing (index %d: %d <= %d)"
                % (label, index, times[index], times[index - 1])
            )


def _coerce_source_path(raw: str, scene_directory: Path) -> Path:
    if os.name != "nt":
        raw = raw.replace("\\", "/")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if not path.is_absolute():
        path = scene_directory / path
    return path


@dataclass
class QuaternionTrack:
    times: np.ndarray
    quaternions: np.ndarray

    def sample(self, sample_times: np.ndarray) -> np.ndarray:
        if len(self.times) == 1:
            return np.repeat(self.quaternions[:1], len(sample_times), axis=0)
        right = np.searchsorted(self.times, sample_times, side="right")
        left = np.clip(right - 1, 0, len(self.times) - 2)
        t0 = self.times[left].astype(np.float64)
        t1 = self.times[left + 1].astype(np.float64)
        amount = np.clip(
            (sample_times.astype(np.float64) - t0) / np.maximum(t1 - t0, 1.0),
            0.0,
            1.0,
        )
        q0 = self.quaternions[left]
        q1 = self.quaternions[left + 1]
        dot = np.sum(q0 * q1, axis=1)
        q1 = q1.copy()
        q1[dot < 0.0] *= -1.0
        dot = np.abs(dot)
        near = dot > 0.9995
        result = np.empty_like(q0)
        if np.any(near):
            interpolated = q0[near] + amount[near, None] * (
                q1[near] - q0[near]
            )
            result[near] = interpolated / np.maximum(
                np.linalg.norm(interpolated, axis=1, keepdims=True), 1.0e-12
            )
        if np.any(~near):
            theta = np.arccos(np.clip(dot[~near], -1.0, 1.0))
            sin_theta = np.sin(theta)
            first = np.sin((1.0 - amount[~near]) * theta) / sin_theta
            second = np.sin(amount[~near] * theta) / sin_theta
            result[~near] = (
                first[:, None] * q0[~near] + second[:, None] * q1[~near]
            )
        return result / np.maximum(
            np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12
        )


@dataclass
class PositionTrack:
    times: np.ndarray
    positions: np.ndarray

    def sample(self, sample_times: np.ndarray) -> np.ndarray:
        if len(self.times) == 1:
            return np.repeat(self.positions[:1], len(sample_times), axis=0)
        right = np.searchsorted(self.times, sample_times, side="right")
        left = np.clip(right - 1, 0, len(self.times) - 2)
        t0 = self.times[left].astype(np.float64)
        t1 = self.times[left + 1].astype(np.float64)
        amount = np.clip(
            (sample_times.astype(np.float64) - t0) / np.maximum(t1 - t0, 1.0),
            0.0,
            1.0,
        )
        return (
            self.positions[left]
            + amount[:, None]
            * (self.positions[left + 1] - self.positions[left])
        )

    def velocity_per_sample(self, sample_times: np.ndarray) -> np.ndarray:
        """Piecewise-linear path velocity in world units per sample."""
        result = np.zeros((len(sample_times), 3), dtype=np.float64)
        if len(self.times) == 1:
            return result
        before = sample_times < self.times[0]
        after = sample_times >= self.times[-1]
        active = ~(before | after)
        if np.any(active):
            right = np.searchsorted(
                self.times, sample_times[active], side="right"
            )
            left = np.clip(right - 1, 0, len(self.times) - 2)
            duration = (self.times[left + 1] - self.times[left]).astype(
                np.float64
            )
            result[active] = (
                self.positions[left + 1] - self.positions[left]
            ) / np.maximum(duration[:, None], 1.0)
        return result


@dataclass(frozen=True)
class HeadSpec:
    position: PositionTrack
    rotation: QuaternionTrack


@dataclass(frozen=True)
class SourceSpec:
    name: str
    file_path: Path
    gain: float
    loop: bool
    start_sample: int
    position: PositionTrack


@dataclass(frozen=True)
class OccluderSpec:
    name: str
    minimum: np.ndarray
    maximum: np.ndarray
    attenuation: float
    cutoff_hz: float


@dataclass(frozen=True)
class SceneSpec:
    path: Path
    sample_rate: int
    frame_length: int
    duration_samples: Optional[int]
    head: HeadSpec
    sources: Tuple[SourceSpec, ...]
    occluders: Tuple[OccluderSpec, ...]
    speed_of_sound: float
    reference_distance: float
    head_radius: float
    occlusion_base_cutoff_hz: float

    def timeline_end_sample(self) -> int:
        candidates = [0]
        if self.duration_samples is not None:
            candidates.append(self.duration_samples)
        candidates.extend(int(value) for value in self.head.rotation.times)
        candidates.extend(int(value) for value in self.head.position.times)
        for source in self.sources:
            candidates.extend(int(value) for value in source.position.times)
            candidates.append(source.start_sample)
        return max(candidates)


def _parse_quaternion_track(
    raw_head: Dict[str, Any], frame_length: int, duration_samples: Optional[int]
) -> QuaternionTrack:
    raw_keys = _key_value(
        raw_head, ("quaternion_keys", "rotation_keys", "quaternions", "orientation")
    )
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ParameterError(
            "head.quaternion_keys must be a non-empty list"
        )
    times: List[int] = []
    quaternions: List[np.ndarray] = []
    for index, item in enumerate(raw_keys):
        label = "head.quaternion_keys[%d]" % index
        if isinstance(item, dict):
            timestamp = _key_time(item, label)
            raw_quaternion = _key_value(item, ("quaternion", "q", "value"))
            if raw_quaternion is None:
                raise ParameterError("%s is missing a quaternion" % label)
        else:
            timestamp = index * frame_length
            raw_quaternion = item
        quaternion = _require_vec4(raw_quaternion, label)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-10:
            raise ParameterError("%s quaternion must not be zero" % label)
        times.append(timestamp)
        quaternions.append(quaternion / norm)
    _ensure_strict_times(times, "head.quaternion_keys")
    if duration_samples is not None and times[-1] > duration_samples:
        raise ParameterError(
            "head.quaternion_keys extends past duration_samples"
        )
    return QuaternionTrack(
        np.asarray(times, dtype=np.int64), np.asarray(quaternions, dtype=np.float64)
    )


def _parse_position_track(
    raw_value: Any,
    label: str,
    frame_length: int,
    duration_samples: Optional[int],
) -> PositionTrack:
    if not isinstance(raw_value, list) or not raw_value:
        raise ParameterError("%s must be a non-empty list" % label)
    times: List[int] = []
    positions: List[np.ndarray] = []
    for index, item in enumerate(raw_value):
        item_label = "%s[%d]" % (label, index)
        if isinstance(item, dict):
            timestamp = _key_time(item, item_label)
            raw_position = _key_value(item, ("position", "p", "value"))
            if raw_position is None:
                raise ParameterError("%s is missing a position" % item_label)
        else:
            timestamp = index * frame_length
            raw_position = item
        times.append(timestamp)
        positions.append(_require_vec3(raw_position, item_label))
    _ensure_strict_times(times, label)
    if duration_samples is not None and times[-1] > duration_samples:
        raise ParameterError("%s extends past duration_samples" % label)
    return PositionTrack(
        np.asarray(times, dtype=np.int64), np.asarray(positions, dtype=np.float64)
    )


def _parse_head(
    raw_head: Any, frame_length: int, duration_samples: Optional[int]
) -> HeadSpec:
    if not isinstance(raw_head, dict):
        raise ParameterError("head must be an object")
    rotation = _parse_quaternion_track(
        raw_head, frame_length, duration_samples
    )
    raw_position_keys = _key_value(
        raw_head, ("position_keys", "positions", "position_curve")
    )
    if raw_position_keys is None:
        static_position = _require_vec3(
            raw_head.get("position", [0.0, 0.0, 0.0]),
            "head.position",
        )
        position = PositionTrack(
            np.array([0], dtype=np.int64), static_position[None, :]
        )
    else:
        position = _parse_position_track(
            raw_position_keys,
            "head.position_keys",
            frame_length,
            duration_samples,
        )
    return HeadSpec(position=position, rotation=rotation)


def _parse_sources(
    raw_sources: Any,
    scene_directory: Path,
    frame_length: int,
    duration_samples: Optional[int],
) -> Tuple[SourceSpec, ...]:
    if raw_sources is None:
        return tuple()
    if not isinstance(raw_sources, list):
        raise ParameterError("sources must be a list")
    names = set()
    result: List[SourceSpec] = []
    for index, item in enumerate(raw_sources):
        label = "sources[%d]" % index
        if not isinstance(item, dict):
            raise ParameterError("%s must be an object" % label)
        name = item.get("name", item.get("id"))
        if not isinstance(name, str) or not name.strip():
            raise ParameterError("%s.name must be a non-empty string" % label)
        if name in names:
            raise ParameterError("source name %r is duplicated" % name)
        names.add(name)
        raw_file = item.get("file", item.get("path", item.get("wav")))
        if not isinstance(raw_file, str) or not raw_file:
            raise ParameterError("%s.file must be a non-empty path" % label)
        has_linear_gain = "gain" in item
        has_db_gain = "gain_db" in item
        if has_linear_gain and has_db_gain:
            raise ParameterError(
                "%s must not define both gain and gain_db" % label
            )
        if has_db_gain:
            gain_db = _require_float(
                item["gain_db"], "%s.gain_db" % label
            )
            try:
                gain = 10.0 ** (gain_db / 20.0)
            except OverflowError as exc:
                raise ParameterError(
                    "%s.gain_db is outside the supported range" % label
                ) from exc
            if not math.isfinite(gain):
                raise ParameterError(
                    "%s.gain_db is outside the supported range" % label
                )
        else:
            gain = _require_float(
                item.get("gain", 1.0), "%s.gain" % label
            )
            if gain < 0.0:
                raise ParameterError("%s.gain must be >= 0" % label)
        loop_value = item.get("loop", False)
        if not isinstance(loop_value, bool):
            raise ParameterError("%s.loop must be a boolean" % label)
        start_sample = _require_int(
            item.get("start_sample", item.get("start", 0)),
            "%s.start_sample" % label,
            0,
        )
        raw_positions = _key_value(
            item, ("position_keys", "positions", "position_curve")
        )
        if raw_positions is None:
            raw_positions = [
                {
                    "time": 0,
                    "position": item.get("position", [0.0, 0.0, 1.0]),
                }
            ]
        positions = _parse_position_track(
            raw_positions,
            "%s.position_keys" % label,
            frame_length,
            duration_samples,
        )
        result.append(
            SourceSpec(
                name=name,
                file_path=_coerce_source_path(raw_file, scene_directory),
                gain=gain,
                loop=loop_value,
                start_sample=start_sample,
                position=positions,
            )
        )
    return tuple(result)


def _parse_occluders(raw_occluders: Any) -> Tuple[OccluderSpec, ...]:
    if raw_occluders is None:
        return tuple()
    if not isinstance(raw_occluders, list):
        raise ParameterError("occluders must be a list")
    result: List[OccluderSpec] = []
    for index, item in enumerate(raw_occluders):
        label = "occluders[%d]" % index
        if not isinstance(item, dict):
            raise ParameterError("%s must be an object" % label)
        name = item.get("name", "occluder-%d" % index)
        if not isinstance(name, str) or not name:
            raise ParameterError("%s.name must be a non-empty string" % label)
        raw_min = _key_value(item, ("min", "minimum", "min_corner"))
        raw_max = _key_value(item, ("max", "maximum", "max_corner"))
        if raw_min is None or raw_max is None:
            raise ParameterError("%s needs min and max corners" % label)
        minimum = _require_vec3(raw_min, "%s.min" % label)
        maximum = _require_vec3(raw_max, "%s.max" % label)
        if np.any(minimum > maximum):
            raise ParameterError("%s.min must not exceed max" % label)
        attenuation = _require_float(
            item.get("attenuation", 0.5), "%s.attenuation" % label
        )
        if not 0.0 <= attenuation <= 1.0:
            raise ParameterError("%s.attenuation must be in [0, 1]" % label)
        cutoff_hz = _require_float(
            item.get("cutoff_hz", 6000.0), "%s.cutoff_hz" % label
        )
        if cutoff_hz <= 0.0:
            raise ParameterError("%s.cutoff_hz must be positive" % label)
        result.append(
            OccluderSpec(
                name=name,
                minimum=minimum,
                maximum=maximum,
                attenuation=attenuation,
                cutoff_hz=cutoff_hz,
            )
        )
    return tuple(result)


def parse_scene(data: Any, scene_path: Union[str, Path]) -> SceneSpec:
    if not isinstance(data, dict):
        raise ParameterError("scene JSON root must be an object")
    scene_path = Path(scene_path).resolve()
    sample_rate = _require_int(
        data.get("sample_rate", data.get("sampleRate")),
        "sample_rate",
        8000,
    )
    if sample_rate > 384000:
        raise ParameterError("sample_rate must be <= 384000")
    frame_length = _require_int(
        _key_value(
            data, ("frame_length", "frameLength", "frame_size", "block_size")
        ),
        "frame_length",
        1,
    )
    if frame_length > 65536:
        raise ParameterError("frame_length must be <= 65536")
    raw_duration = data.get("duration_samples")
    duration_samples: Optional[int] = None
    if raw_duration is not None:
        duration_samples = _require_int(
            raw_duration, "duration_samples", 1
        )
    speed_of_sound = _require_float(
        data.get("speed_of_sound", 343.0), "speed_of_sound"
    )
    if speed_of_sound <= 0.0:
        raise ParameterError("speed_of_sound must be positive")
    reference_distance = _require_float(
        data.get("reference_distance", 1.0), "reference_distance"
    )
    if reference_distance <= 0.0:
        raise ParameterError("reference_distance must be positive")
    head_radius = _require_float(
        data.get("head_radius", 0.0875), "head_radius"
    )
    if head_radius <= 0.0:
        raise ParameterError("head_radius must be positive")
    base_cutoff = _require_float(
        data.get("occlusion_base_cutoff_hz", 18000.0),
        "occlusion_base_cutoff_hz",
    )
    if base_cutoff <= 0.0:
        raise ParameterError("occlusion_base_cutoff_hz must be positive")

    head = _parse_head(data.get("head"), frame_length, duration_samples)
    sources = _parse_sources(
        data.get("sources", data.get("tracks")),
        scene_path.parent,
        frame_length,
        duration_samples,
    )
    occluders = _parse_occluders(
        data.get("occluders", data.get("aabbs", data.get("obstacles")))
    )
    return SceneSpec(
        path=scene_path,
        sample_rate=sample_rate,
        frame_length=frame_length,
        duration_samples=duration_samples,
        head=head,
        sources=sources,
        occluders=occluders,
        speed_of_sound=speed_of_sound,
        reference_distance=reference_distance,
        head_radius=head_radius,
        occlusion_base_cutoff_hz=base_cutoff,
    )


def load_scene(path: Union[str, Path]) -> SceneSpec:
    scene_path = Path(path)
    try:
        with scene_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ParameterError("scene file not found: %s" % scene_path) from exc
    except PermissionError as exc:
        raise ParameterError("scene file is not readable: %s" % scene_path) from exc
    except json.JSONDecodeError as exc:
        raise ParameterError(
            "invalid scene JSON at line %d column %d: %s"
            % (exc.lineno, exc.colno, exc.msg)
        ) from exc
    except OSError as exc:
        raise ParameterError("cannot read scene file: %s" % exc) from exc
    return parse_scene(data, scene_path)
