"""Source configuration parsing and spatial/distance models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .errors import (
    BusinessError,
    E_CONFIG_INVALID,
    E_CONFIG_JSON,
    E_CONFIG_MISSING_FIELD,
    E_CONFIG_READ,
    E_HRTF_VERSION,
)
from .hrtf import HRTF_DATASET_VERSION

_DISTANCE_EPSILON = 1.0e-9


def _require(
    mapping: Mapping[str, Any], keys: Sequence[str], where: str
) -> Tuple[str, Any]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    raise BusinessError(
        E_CONFIG_MISSING_FIELD,
        f"Missing required field at {where}: one of {', '.join(keys)}",
        {"path": where, "accepted_keys": list(keys)},
    )


def _as_float(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Expected a finite number at {where}",
            {"path": where, "value": repr(value)},
        ) from exc
    if not math.isfinite(result):
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Expected a finite number at {where}",
            {"path": where, "value": repr(value)},
        )
    return result


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "1", "0"}:
        return value.lower() in {"true", "1"}
    raise BusinessError(
        E_CONFIG_INVALID,
        f"Expected a boolean at {where}",
        {"path": where, "value": repr(value)},
    )


def _vector3(value: Any, where: str) -> np.ndarray:
    if isinstance(value, Mapping):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Expected a three-component vector at {where}",
            {"path": where, "value": repr(value)},
        )
    vector = np.array(
        [_as_float(component, f"{where}[{i}]") for i, component in enumerate(value)],
        dtype=np.float64,
    )
    return vector


def _looks_like_vector3(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(key in value for key in ("x", "y", "z"))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(
        isinstance(component, (int, float)) and not isinstance(component, bool)
        for component in value
    )


def _normalise(vector: np.ndarray, where: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Vector at {where} must not be zero length",
            {"path": where},
        )
    return vector / norm


def right_up_forward_basis(
    forward: np.ndarray, up_hint: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalise(forward, "listener.forward")
    up_hint = _normalise(up_hint, "listener.up")
    right = np.cross(up_hint, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1.0e-8:
        raise BusinessError(
            E_CONFIG_INVALID,
            "Listener forward and up vectors are parallel",
            {"forward": forward.tolist(), "up": up_hint.tolist()},
        )
    right = right / right_norm
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    return right, up, forward


def relative_angles(
    relative_position: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> Tuple[float, float, float]:
    lateral = float(np.dot(relative_position, right))
    vertical = float(np.dot(relative_position, up))
    frontal = float(np.dot(relative_position, forward))
    distance = float(np.linalg.norm(relative_position))
    azimuth = math.degrees(math.atan2(lateral, frontal))
    elevation = math.degrees(math.atan2(vertical, math.hypot(lateral, frontal)))
    return azimuth, elevation, distance


@dataclass(frozen=True)
class DistanceModel:
    kind: str
    reference_distance: float = 1.0
    minimum_distance: float = 0.05
    points: np.ndarray | None = None

    def gain(self, distances: np.ndarray | float) -> np.ndarray:
        values = np.asarray(distances, dtype=np.float64)
        if self.kind == "free_field":
            safe = np.maximum(values, self.minimum_distance)
            return self.reference_distance / safe
        assert self.points is not None
        return np.interp(values, self.points[:, 0], self.points[:, 1])


@dataclass(frozen=True)
class AirAbsorption:
    enabled: bool = True
    threshold: float = 5.0
    coefficient_per_meter: float = 0.15
    min_cutoff_hz: float = 500.0
    base_cutoff_hz: float | None = None

    def alpha(self, distance: float, sample_rate: int) -> float | None:
        if not self.enabled or distance <= self.threshold:
            return None
        base = self.base_cutoff_hz
        if base is None:
            base = min(19000.0, 0.48 * sample_rate)
        base = min(max(base, self.min_cutoff_hz), 0.48 * sample_rate)
        exponent = -self.coefficient_per_meter * (distance - self.threshold)
        cutoff = self.min_cutoff_hz + (base - self.min_cutoff_hz) * math.exp(exponent)
        cutoff = min(max(cutoff, self.min_cutoff_hz), 0.49 * sample_rate)
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sample_rate)
        return min(1.0, max(1.0e-6, alpha))


@dataclass(frozen=True)
class MotionTrack:
    times: np.ndarray
    positions: np.ndarray

    @property
    def moving(self) -> bool:
        return self.times.size > 1 and not np.allclose(
            self.positions, self.positions[0], rtol=0.0, atol=0.0
        )

    @property
    def start_time(self) -> float:
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        return float(self.times[-1])

    def positions_at(self, times: np.ndarray | float) -> np.ndarray:
        query = np.asarray(times, dtype=np.float64)
        if self.times.size == 1:
            return np.broadcast_to(self.positions[0], query.shape + (3,)).copy()
        result = np.empty(query.shape + (3,), dtype=np.float64)
        for axis in range(3):
            result[..., axis] = np.interp(query, self.times, self.positions[:, axis])
        return result

    def velocities_at(self, times: np.ndarray | float) -> np.ndarray:
        query = np.asarray(times, dtype=np.float64)
        result = np.zeros(query.shape + (3,), dtype=np.float64)
        if self.times.size == 1:
            return result
        intervals = np.diff(self.times)
        velocities = np.diff(self.positions, axis=0) / intervals[:, None]
        indices = np.searchsorted(self.times, query, side="right") - 1
        inside = (query >= self.times[0]) & (query <= self.times[-1])
        clipped = np.clip(indices, 0, velocities.shape[0] - 1)
        result[inside] = velocities[clipped[inside]]
        return result

    def distances_at(
        self, times: np.ndarray | float, listener: np.ndarray
    ) -> np.ndarray:
        positions = self.positions_at(times)
        return np.linalg.norm(positions - listener, axis=-1)

    def radial_velocity_at(
        self, times: np.ndarray | float, listener: np.ndarray
    ) -> np.ndarray:
        query = np.asarray(times, dtype=np.float64)
        positions = self.positions_at(query)
        relative = positions - listener
        distances = np.linalg.norm(relative, axis=-1)
        safe = np.maximum(distances, _DISTANCE_EPSILON)
        unit = relative / safe[..., None]
        velocity = self.velocities_at(query)
        return np.sum(velocity * unit, axis=-1)

    def representative_distance(self, listener: np.ndarray) -> float:
        if self.times.size == 1:
            return float(np.linalg.norm(self.positions[0] - listener))
        sample_times = np.linspace(
            self.start_time,
            self.end_time,
            min(257, max(3, self.times.size * 4)),
            dtype=np.float64,
        )
        return float(np.mean(self.distances_at(sample_times, listener)))


@dataclass(frozen=True)
class SourceSpec:
    index: int
    source_id: str
    motion: MotionTrack
    channels: Tuple[int, ...] | None
    gain: float
    delay_frames: int
    spatialize: bool
    hrtf_gain: float
    azimuth_offset_deg: float
    elevation_offset_deg: float
    distance_model: DistanceModel
    air_absorption: AirAbsorption
    reference_position: np.ndarray


@dataclass(frozen=True)
class SourceConfig:
    sources: Tuple[SourceSpec, ...]
    sha256: str
    raw: Dict[str, Any] = field(default_factory=dict)
    hrtf_version: str = HRTF_DATASET_VERSION


def _parse_points(value: Any, where: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) < 2:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Custom distance curve at {where} must have at least two points",
            {"path": where},
        )
    points: List[Tuple[float, float]] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            _, distance = _require(item, ("distance", "r", "x"), f"{where}[{index}]")
            _, gain = _require(item, ("gain", "value", "y"), f"{where}[{index}]")
            pair = (
                _as_float(distance, f"{where}[{index}].distance"),
                _as_float(gain, f"{where}[{index}].gain"),
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pair = (
                _as_float(item[0], f"{where}[{index}][0]"),
                _as_float(item[1], f"{where}[{index}][1]"),
            )
        else:
            raise BusinessError(
                E_CONFIG_INVALID,
                f"Invalid custom curve point at {where}[{index}]",
                {"path": f"{where}[{index}]", "value": repr(item)},
            )
        if pair[0] < 0.0 or pair[1] < 0.0:
            raise BusinessError(
                E_CONFIG_INVALID,
                "Custom curve distance and gain must be non-negative at "
                f"{where}[{index}]",
                {"path": f"{where}[{index}]"},
            )
        points.append(pair)
    array = np.asarray(points, dtype=np.float64)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    if np.any(np.diff(array[:, 0]) <= 0.0):
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Custom curve distances must be strictly increasing at {where}",
            {"path": where},
        )
    return array


def _parse_distance_model(value: Any, where: str) -> DistanceModel:
    if value is None:
        return DistanceModel(kind="free_field")
    if isinstance(value, str):
        kind = value.strip().lower()
        if kind in {"free_field", "free-field", "inverse", "1/r"}:
            return DistanceModel(kind="free_field")
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Unsupported distance model at {where}",
            {"path": where, "value": value},
        )
    if not isinstance(value, Mapping):
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Distance model at {where} must be an object or string",
            {"path": where, "value": repr(value)},
        )
    kind_value = value.get("type", value.get("kind", "free_field"))
    kind = str(kind_value).strip().lower().replace("-", "_").replace(" ", "_")
    reference = _as_float(
        value.get("reference_distance", 1.0), f"{where}.reference_distance"
    )
    minimum = _as_float(
        value.get("minimum_distance", 0.05), f"{where}.minimum_distance"
    )
    if reference < 0.0 or minimum <= 0.0:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Invalid distance model distances at {where}",
            {"path": where},
        )
    if kind in {"free_field", "freefield", "inverse", "1_over_r", "1/r"}:
        return DistanceModel(
            kind="free_field",
            reference_distance=reference,
            minimum_distance=minimum,
        )
    if kind in {"custom", "curve", "custom_curve"}:
        points_value = value.get("points", value.get("curve", value.get("knots")))
        if points_value is None:
            raise BusinessError(
                E_CONFIG_MISSING_FIELD,
                f"Custom distance model at {where} requires points",
                {"path": where, "accepted_keys": ["points", "curve", "knots"]},
            )
        return DistanceModel(
            kind="custom",
            reference_distance=reference,
            minimum_distance=minimum,
            points=_parse_points(points_value, f"{where}.points"),
        )
    raise BusinessError(
        E_CONFIG_INVALID,
        f"Unsupported distance model at {where}",
        {"path": where, "value": kind_value},
    )


def _parse_air(value: Any, where: str) -> AirAbsorption:
    if value is None:
        return AirAbsorption()
    if not isinstance(value, Mapping):
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Air absorption at {where} must be an object",
            {"path": where, "value": repr(value)},
        )
    enabled = _as_bool(value.get("enabled", True), f"{where}.enabled")
    threshold = _as_float(
        value.get("threshold", value.get("distance_threshold", 5.0)),
        f"{where}.threshold",
    )
    coefficient = _as_float(
        value.get(
            "coefficient_per_meter",
            value.get("coefficient", value.get("decay_per_meter", 0.15)),
        ),
        f"{where}.coefficient_per_meter",
    )
    min_cutoff = _as_float(value.get("min_cutoff_hz", 500.0), f"{where}.min_cutoff_hz")
    base_raw = value.get("base_cutoff_hz")
    base_cutoff = (
        None if base_raw is None else _as_float(base_raw, f"{where}.base_cutoff_hz")
    )
    if threshold < 0.0 or coefficient < 0.0 or min_cutoff <= 0.0:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Air absorption parameters at {where} must be non-negative",
            {"path": where},
        )
    return AirAbsorption(
        enabled=enabled,
        threshold=threshold,
        coefficient_per_meter=coefficient,
        min_cutoff_hz=min_cutoff,
        base_cutoff_hz=base_cutoff,
    )


def _parse_trajectory(value: Any, where: str) -> MotionTrack:
    if isinstance(value, Mapping):
        times_value = value.get("times", value.get("t"))
        positions_value = value.get(
            "positions", value.get("values", value.get("points"))
        )
        if isinstance(times_value, list) and isinstance(positions_value, list):
            if len(times_value) != len(positions_value):
                raise BusinessError(
                    E_CONFIG_INVALID,
                    f"Trajectory times/positions lengths differ at {where}",
                    {"path": where},
                )
            value = [
                [time_value, *list(position_value)]
                if isinstance(position_value, (list, tuple))
                else {"time": time_value, "position": position_value}
                for time_value, position_value in zip(times_value, positions_value)
            ]
        else:
            raise BusinessError(
                E_CONFIG_INVALID,
                f"Trajectory object at {where} requires times and positions arrays",
                {"path": where},
            )
    if not isinstance(value, list) or not value:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Trajectory at {where} must be a non-empty array",
            {"path": where},
        )
    times: List[float] = []
    positions: List[np.ndarray] = []
    for index, item in enumerate(value):
        item_where = f"{where}[{index}]"
        if isinstance(item, Mapping):
            _, time_value = _require(item, ("time", "t", "time_seconds"), item_where)
            if all(key in item for key in ("x", "y", "z")):
                position_value = item
            else:
                _, position_value = _require(
                    item, ("position", "pos", "coordinates"), item_where
                )
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            time_value = item[0]
            position_value = item[1:]
        else:
            raise BusinessError(
                E_CONFIG_INVALID,
                f"Trajectory point at {item_where} must be [time,x,y,z] or an object",
                {"path": item_where, "value": repr(item)},
            )
        times.append(_as_float(time_value, f"{item_where}.time"))
        positions.append(_vector3(position_value, f"{item_where}.position"))
    time_array = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(time_array) <= 0.0):
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Trajectory times must be strictly increasing at {where}",
            {"path": where},
        )
    return MotionTrack(time_array, np.stack(positions, axis=0))


def _parse_channels(value: Any, where: str) -> Tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, list) and value:
        values = value
    else:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Channel selection at {where} must be an integer or non-empty list",
            {"path": where, "value": repr(value)},
        )
    result: List[int] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, int):
            raise BusinessError(
                E_CONFIG_INVALID,
                f"Channel index at {where}[{index}] must be an integer",
                {"path": f"{where}[{index}]", "value": repr(item)},
            )
        if item < 0:
            raise BusinessError(
                E_CONFIG_INVALID,
                f"Channel index at {where}[{index}] must be non-negative",
                {"path": f"{where}[{index}]"},
            )
        result.append(item)
    return tuple(result)


def _parse_source(
    value: Any,
    index: int,
    global_distance: DistanceModel,
    global_air: AirAbsorption,
    sample_rate: int,
) -> SourceSpec:
    where = f"sources[{index}]"
    if not isinstance(value, Mapping):
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Source at {where} must be an object",
            {"path": where, "value": repr(value)},
        )

    trajectory_value = None
    for key in ("positions", "position_series", "trajectory", "motion"):
        if key in value and value[key] is not None:
            trajectory_value = value[key]
            break

    position_value = None
    for key in ("position", "pos", "coordinates", "location"):
        if key in value and value[key] is not None:
            position_value = value[key]
            break

    if position_value is None and trajectory_value is None:
        raise BusinessError(
            E_CONFIG_MISSING_FIELD,
            f"Source at {where} requires a position or a position time series",
            {
                "path": where,
                "accepted_keys": [
                    "position",
                    "positions",
                    "position_series",
                    "trajectory",
                ],
            },
        )

    if trajectory_value is not None and _looks_like_vector3(trajectory_value):
        if position_value is None:
            position_value = trajectory_value
        trajectory_value = None

    if trajectory_value is not None:
        motion = _parse_trajectory(trajectory_value, f"{where}.positions")
    else:
        position = _vector3(position_value, f"{where}.position")
        motion = MotionTrack(np.array([0.0]), position[None, :])

    reference_position = (
        _vector3(position_value, f"{where}.position")
        if position_value is not None
        else motion.positions[0].copy()
    )

    source_id_value = value.get("id", value.get("name", f"source-{index}"))
    source_id = str(source_id_value)
    gain = _as_float(value.get("gain", value.get("volume", 1.0)), f"{where}.gain")
    if gain < 0.0:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Source gain at {where} must be non-negative",
            {"path": where},
        )

    delay_seconds = value.get("delay_seconds")
    delay_ms = value.get("delay_ms")
    delay_samples = value.get("delay_samples")
    if delay_seconds is None and delay_samples is None and delay_ms is None:
        delay_seconds = value.get("delay", 0.0)
    specified_delays = sum(
        item is not None for item in (delay_seconds, delay_samples, delay_ms)
    )
    if specified_delays > 1:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Source at {where} must specify only one delay field",
            {"path": where},
        )
    if delay_samples is not None:
        delay_frames_float = _as_float(delay_samples, f"{where}.delay_samples")
    elif delay_ms is not None:
        delay_frames_float = (
            _as_float(delay_ms, f"{where}.delay_ms") * sample_rate / 1000.0
        )
    else:
        delay_frames_float = (
            _as_float(delay_seconds, f"{where}.delay_seconds") * sample_rate
        )
    if delay_frames_float < 0.0:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"Source delay at {where} must be non-negative",
            {"path": where},
        )
    delay_frames = int(round(delay_frames_float))

    channels_value = value.get(
        "channels", value.get("channel", value.get("input_channels"))
    )
    channels = _parse_channels(channels_value, f"{where}.channels")

    spatialize_value = value.get(
        "spatialize",
        value.get(
            "hrtf_enabled",
            value.get("spatialization", value.get("hrtf", True)),
        ),
    )
    spatialize = _as_bool(spatialize_value, f"{where}.spatialize")
    hrtf_gain = _as_float(
        value.get("hrtf_gain", value.get("spatial_gain", 1.0)),
        f"{where}.hrtf_gain",
    )
    if hrtf_gain < 0.0:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"HRTF gain at {where} must be non-negative",
            {"path": where},
        )
    azimuth_offset = _as_float(
        value.get("azimuth_offset_deg", value.get("azimuth_offset", 0.0)),
        f"{where}.azimuth_offset_deg",
    )
    elevation_offset = _as_float(
        value.get("elevation_offset_deg", value.get("elevation_offset", 0.0)),
        f"{where}.elevation_offset_deg",
    )

    if "custom_curve" in value:
        source_distance = DistanceModel(
            kind="custom",
            points=_parse_points(value["custom_curve"], f"{where}.custom_curve"),
        )
    elif "distance_model" in value:
        model_value = value.get("distance_model")
        if isinstance(model_value, str) and model_value.strip().lower() in {
            "custom",
            "curve",
            "custom_curve",
        }:
            raise BusinessError(
                E_CONFIG_MISSING_FIELD,
                f"Custom distance model at {where} requires a custom_curve field",
                {"path": f"{where}.distance_model"},
            )
        source_distance = _parse_distance_model(model_value, f"{where}.distance_model")
    else:
        source_distance = global_distance
    source_air = (
        _parse_air(value.get("air_absorption"), f"{where}.air_absorption")
        if "air_absorption" in value
        else global_air
    )

    return SourceSpec(
        index=index,
        source_id=source_id,
        motion=motion,
        channels=channels,
        gain=gain,
        delay_frames=delay_frames,
        spatialize=spatialize,
        hrtf_gain=hrtf_gain,
        azimuth_offset_deg=azimuth_offset,
        elevation_offset_deg=elevation_offset,
        distance_model=source_distance,
        air_absorption=source_air,
        reference_position=reference_position,
    )


def load_source_config(path: str | Path, sample_rate: int) -> SourceConfig:
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as exc:
        raise BusinessError(
            E_CONFIG_READ,
            f"Unable to read source configuration: {config_path}",
            {"path": str(config_path), "reason": str(exc)},
        ) from exc

    try:
        document = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BusinessError(
            E_CONFIG_JSON,
            f"Invalid JSON source configuration: {config_path}",
            {"path": str(config_path), "reason": str(exc)},
        ) from exc

    if isinstance(document, list):
        root: Dict[str, Any] = {"sources": document}
    elif isinstance(document, dict):
        root = document
    else:
        raise BusinessError(
            E_CONFIG_INVALID,
            "Source configuration must be a JSON object or array",
            {"path": str(config_path)},
        )

    hrtf_version_value = root.get(
        "hrtf_version", root.get("hrtf_dataset_version", HRTF_DATASET_VERSION)
    )
    hrtf_version = str(hrtf_version_value)
    if hrtf_version != HRTF_DATASET_VERSION:
        raise BusinessError(
            E_HRTF_VERSION,
            "HRTF dataset version mismatch",
            {
                "config_version": hrtf_version,
                "available_version": HRTF_DATASET_VERSION,
            },
        )

    raw_sources = root.get("sources", root.get("source_positions", root.get("objects")))
    if raw_sources is None:
        raise BusinessError(
            E_CONFIG_MISSING_FIELD,
            "Source configuration is missing the sources array",
            {"path": "sources"},
        )
    if isinstance(raw_sources, Mapping):
        raw_sources = [
            {**dict(source_value), "id": source_id}
            if isinstance(source_value, Mapping)
            else source_value
            for source_id, source_value in raw_sources.items()
        ]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise BusinessError(
            E_CONFIG_INVALID,
            "sources must be a non-empty array",
            {"path": "sources"},
        )

    if "custom_curve" in root:
        global_distance = DistanceModel(
            kind="custom",
            points=_parse_points(root["custom_curve"], "custom_curve"),
        )
    else:
        model_value = root.get("distance_model")
        if isinstance(model_value, str) and model_value.strip().lower() in {
            "custom",
            "curve",
            "custom_curve",
        }:
            raise BusinessError(
                E_CONFIG_MISSING_FIELD,
                "Custom distance model requires a custom_curve field",
                {"path": "distance_model"},
            )
        global_distance = _parse_distance_model(model_value, "distance_model")
    global_air = _parse_air(root.get("air_absorption"), "air_absorption")

    sources = tuple(
        _parse_source(
            value,
            index,
            global_distance,
            global_air,
            sample_rate,
        )
        for index, value in enumerate(raw_sources)
    )
    return SourceConfig(
        sources=sources,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw=root,
        hrtf_version=hrtf_version,
    )
