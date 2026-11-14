"""Parameter and telemetry parsing with business-level validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import CHANNELS, FRAME_INTERVAL_MS, SAMPLE_RATE
from .errors import SynthError
from .util import finite_number, round_half_up


def _error(code: str, message: str) -> None:
    raise SynthError(code, message)


def _mapping(value: Any, path: str, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _error(code, f"{path} must be an object")
    return value


def _required(mapping: dict[str, Any], key: str, path: str, code: str) -> Any:
    if key not in mapping:
        _error(code, f"missing required field {path}.{key}")
    return mapping[key]


def _string(value: Any, path: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(code, f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _error(code, f"{path} must be an integer")
    return value


def _number(
    value: Any,
    path: str,
    code: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_minimum: bool = True,
    inclusive_maximum: bool = True,
) -> float:
    if not finite_number(value):
        _error(code, f"{path} must be a finite number")
    number = float(value)
    if minimum is not None:
        bad_minimum = number < minimum if inclusive_minimum else number <= minimum
        if bad_minimum:
            _error(
                "E_PARAMS_RANGE"
                if code.startswith("E_PARAMS")
                else "E_TELEMETRY_RANGE",
                f"{path}={number} is below minimum {minimum}",
            )
    if maximum is not None:
        bad_maximum = number > maximum if inclusive_maximum else number >= maximum
        if bad_maximum:
            _error(
                "E_PARAMS_RANGE"
                if code.startswith("E_PARAMS")
                else "E_TELEMETRY_RANGE",
                f"{path}={number} is above maximum {maximum}",
            )
    return number


def _load_json(path: Path, code_io: str, code_json: str) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _error(code_io, f"cannot read {path}: {exc}")
    except UnicodeError as exc:
        _error(code_json, f"{path} is not valid UTF-8: {exc}")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        _error(code_json, f"invalid JSON in {path}: {exc}")


@dataclass(frozen=True)
class Harmonic:
    order: float
    gain: float
    phase_rad: float


@dataclass(frozen=True)
class EngineModel:
    rpm_min: float
    rpm_max: float
    firing_order: float
    harmonics: tuple[Harmonic, ...]
    base_amplitude: float
    throttle_amplitude: float
    shift_rpm_threshold: float
    shift_crossfade_ms: float
    stereo_phase_spread_deg: float

    @property
    def orders(self) -> np.ndarray:
        return np.array([item.order for item in self.harmonics], dtype=np.float64)

    @property
    def gains(self) -> np.ndarray:
        return np.array([item.gain for item in self.harmonics], dtype=np.float64)

    @property
    def phases_deg(self) -> np.ndarray:
        return np.array(
            [math.degrees(item.phase_rad) for item in self.harmonics],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SurfaceProfile:
    grip_cutoff_hz: float
    grip_gain: float
    slip_cutoff_hz: float
    slip_gain: float


@dataclass(frozen=True)
class TireModel:
    level: float
    max_slip: float
    slip_threshold: float
    release_ratio: float
    attack_ms: float
    release_ms: float
    slip_full_scale: float
    slip_load_floor: float
    profiles: dict[str, SurfaceProfile]


@dataclass(frozen=True)
class EventTypeModel:
    gain: float
    duration_ms: float
    attack_ms: float
    decay_ms: float
    waveform: str
    frequency_hz: float
    frequency_end_hz: float
    noise_mix: float


@dataclass(frozen=True)
class EventModel:
    overlap_window_ms: float
    overlap_attenuation: float
    types: dict[str, EventTypeModel]


@dataclass(frozen=True)
class MixModel:
    engine_gain: float
    tire_gain: float
    event_gain: float
    master_gain: float
    soft_limit_threshold: float


@dataclass(frozen=True)
class ParamsModel:
    model_id: str
    engine: EngineModel
    tires: TireModel
    events: EventModel
    mix: MixModel


@dataclass(frozen=True)
class TelemetryEvent:
    frame_index: int
    event_type: str
    intensity: float
    pan: float
    sample_start: int


@dataclass(frozen=True)
class TelemetryModel:
    frame_interval_ms: float
    frame_interval_s: float
    frame_count: int
    samples_per_frame: float
    total_samples: int
    rpm: np.ndarray
    throttle: np.ndarray
    slip: np.ndarray
    surface_indices: np.ndarray
    surfaces: tuple[str, ...]
    events: tuple[TelemetryEvent, ...]


def load_params(path: Path) -> ParamsModel:
    document = _mapping(
        _load_json(path, "E_PARAMS_IO", "E_PARAMS_JSON"),
        "parameters",
        "E_PARAMS_SCHEMA",
    )
    schema_version = _integer(
        _required(document, "schema_version", "parameters", "E_PARAMS_SCHEMA"),
        "parameters.schema_version",
        "E_PARAMS_SCHEMA",
    )
    if schema_version != 1:
        _error("E_PARAMS_SCHEMA", f"unsupported parameters schema_version={schema_version}")

    model_id = _string(
        _required(document, "model_id", "parameters", "E_PARAMS_SCHEMA"),
        "parameters.model_id",
        "E_PARAMS_SCHEMA",
    )

    engine_raw = _mapping(
        _required(document, "engine", "parameters", "E_PARAMS_SCHEMA"),
        "parameters.engine",
        "E_PARAMS_SCHEMA",
    )
    rpm_min = _number(
        _required(engine_raw, "rpm_min", "parameters.engine", "E_PARAMS_SCHEMA"),
        "parameters.engine.rpm_min",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
    )
    rpm_max = _number(
        _required(engine_raw, "rpm_max", "parameters.engine", "E_PARAMS_SCHEMA"),
        "parameters.engine.rpm_max",
        "E_PARAMS_SCHEMA",
        minimum=rpm_min,
        inclusive_minimum=False,
    )
    firing_order = _number(
        _required(engine_raw, "firing_order", "parameters.engine", "E_PARAMS_SCHEMA"),
        "parameters.engine.firing_order",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=16.0,
        inclusive_minimum=False,
    )
    harmonics_raw = _required(
        engine_raw, "harmonics", "parameters.engine", "E_PARAMS_SCHEMA"
    )
    if not isinstance(harmonics_raw, list) or not harmonics_raw:
        _error("E_PARAMS_SCHEMA", "parameters.engine.harmonics must be a non-empty array")
    if len(harmonics_raw) > 64:
        _error("E_PARAMS_RANGE", "parameters.engine.harmonics supports at most 64 entries")
    harmonics: list[Harmonic] = []
    for index, item_raw in enumerate(harmonics_raw):
        path_prefix = f"parameters.engine.harmonics[{index}]"
        item = _mapping(item_raw, path_prefix, "E_PARAMS_SCHEMA")
        order = _number(
            _required(item, "order", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.order",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=16.0,
            inclusive_minimum=False,
        )
        gain = _number(
            _required(item, "gain", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.gain",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=8.0,
        )
        phase_deg = _number(
            _required(item, "phase_deg", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.phase_deg",
            "E_PARAMS_SCHEMA",
            minimum=-720.0,
            maximum=720.0,
        )
        harmonics.append(Harmonic(order, gain, math.radians(phase_deg)))

    base_amplitude = _number(
        _required(engine_raw, "base_amplitude", "parameters.engine", "E_PARAMS_SCHEMA"),
        "parameters.engine.base_amplitude",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=4.0,
    )
    throttle_amplitude = _number(
        _required(
            engine_raw, "throttle_amplitude", "parameters.engine", "E_PARAMS_SCHEMA"
        ),
        "parameters.engine.throttle_amplitude",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=4.0,
    )
    shift_rpm_threshold = _number(
        _required(
            engine_raw, "shift_rpm_threshold", "parameters.engine", "E_PARAMS_SCHEMA"
        ),
        "parameters.engine.shift_rpm_threshold",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        inclusive_minimum=False,
    )
    shift_crossfade_ms = _number(
        _required(
            engine_raw, "shift_crossfade_ms", "parameters.engine", "E_PARAMS_SCHEMA"
        ),
        "parameters.engine.shift_crossfade_ms",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=2000.0,
        inclusive_minimum=False,
    )
    stereo_phase_spread_deg = _number(
        _required(
            engine_raw,
            "stereo_phase_spread_deg",
            "parameters.engine",
            "E_PARAMS_SCHEMA",
        ),
        "parameters.engine.stereo_phase_spread_deg",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=180.0,
    )
    engine = EngineModel(
        rpm_min=rpm_min,
        rpm_max=rpm_max,
        firing_order=firing_order,
        harmonics=tuple(harmonics),
        base_amplitude=base_amplitude,
        throttle_amplitude=throttle_amplitude,
        shift_rpm_threshold=shift_rpm_threshold,
        shift_crossfade_ms=shift_crossfade_ms,
        stereo_phase_spread_deg=stereo_phase_spread_deg,
    )

    tires_raw = _mapping(
        _required(document, "tires", "parameters", "E_PARAMS_SCHEMA"),
        "parameters.tires",
        "E_PARAMS_SCHEMA",
    )
    tire_level = _number(
        _required(tires_raw, "level", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.level",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=8.0,
    )
    max_slip = _number(
        _required(tires_raw, "max_slip", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.max_slip",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=10.0,
        inclusive_minimum=False,
    )
    slip_threshold = _number(
        _required(
            tires_raw, "slip_threshold", "parameters.tires", "E_PARAMS_SCHEMA"
        ),
        "parameters.tires.slip_threshold",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=max_slip,
        inclusive_minimum=False,
    )
    release_ratio = _number(
        _required(tires_raw, "release_ratio", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.release_ratio",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=1.0,
        inclusive_maximum=False,
    )
    attack_ms = _number(
        _required(tires_raw, "attack_ms", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.attack_ms",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=1000.0,
        inclusive_minimum=False,
    )
    release_ms = _number(
        _required(tires_raw, "release_ms", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.release_ms",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=1000.0,
        inclusive_minimum=False,
    )
    slip_full_scale = _number(
        _required(tires_raw, "slip_full_scale", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.slip_full_scale",
        "E_PARAMS_SCHEMA",
        minimum=slip_threshold,
        maximum=max_slip,
    )
    slip_load_floor = _number(
        _required(
            tires_raw, "slip_load_floor", "parameters.tires", "E_PARAMS_SCHEMA"
        ),
        "parameters.tires.slip_load_floor",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=1.0,
    )
    surfaces_raw = _mapping(
        _required(tires_raw, "surfaces", "parameters.tires", "E_PARAMS_SCHEMA"),
        "parameters.tires.surfaces",
        "E_PARAMS_SCHEMA",
    )
    if not surfaces_raw:
        _error("E_PARAMS_SCHEMA", "parameters.tires.surfaces must not be empty")
    profiles: dict[str, SurfaceProfile] = {}
    for surface_name, profile_raw in surfaces_raw.items():
        if not isinstance(surface_name, str) or not surface_name.strip():
            _error("E_PARAMS_SCHEMA", "surface names must be non-empty strings")
        path_prefix = f"parameters.tires.surfaces.{surface_name}"
        profile = _mapping(profile_raw, path_prefix, "E_PARAMS_SCHEMA")
        grip_raw = _mapping(
            _required(profile, "grip", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.grip",
            "E_PARAMS_SCHEMA",
        )
        slip_raw = _mapping(
            _required(profile, "slip", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.slip",
            "E_PARAMS_SCHEMA",
        )
        grip_cutoff = _number(
            _required(grip_raw, "cutoff_hz", f"{path_prefix}.grip", "E_PARAMS_SCHEMA"),
            f"{path_prefix}.grip.cutoff_hz",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=SAMPLE_RATE / 2.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        )
        grip_gain = _number(
            _required(grip_raw, "gain", f"{path_prefix}.grip", "E_PARAMS_SCHEMA"),
            f"{path_prefix}.grip.gain",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=8.0,
        )
        slip_cutoff = _number(
            _required(slip_raw, "cutoff_hz", f"{path_prefix}.slip", "E_PARAMS_SCHEMA"),
            f"{path_prefix}.slip.cutoff_hz",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=SAMPLE_RATE / 2.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        )
        slip_gain = _number(
            _required(slip_raw, "gain", f"{path_prefix}.slip", "E_PARAMS_SCHEMA"),
            f"{path_prefix}.slip.gain",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=8.0,
        )
        profiles[surface_name] = SurfaceProfile(
            grip_cutoff_hz=grip_cutoff,
            grip_gain=grip_gain,
            slip_cutoff_hz=slip_cutoff,
            slip_gain=slip_gain,
        )
    tires = TireModel(
        level=tire_level,
        max_slip=max_slip,
        slip_threshold=slip_threshold,
        release_ratio=release_ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        slip_full_scale=slip_full_scale,
        slip_load_floor=slip_load_floor,
        profiles=profiles,
    )

    events_raw = _mapping(
        _required(document, "events", "parameters", "E_PARAMS_SCHEMA"),
        "parameters.events",
        "E_PARAMS_SCHEMA",
    )
    overlap_window_ms = _number(
        _required(
            events_raw, "overlap_window_ms", "parameters.events", "E_PARAMS_SCHEMA"
        ),
        "parameters.events.overlap_window_ms",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=10000.0,
    )
    overlap_attenuation = _number(
        _required(
            events_raw,
            "overlap_attenuation",
            "parameters.events",
            "E_PARAMS_SCHEMA",
        ),
        "parameters.events.overlap_attenuation",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=1.0,
    )
    event_types_raw = _mapping(
        _required(events_raw, "types", "parameters.events", "E_PARAMS_SCHEMA"),
        "parameters.events.types",
        "E_PARAMS_SCHEMA",
    )
    if not event_types_raw:
        _error("E_PARAMS_SCHEMA", "parameters.events.types must not be empty")
    event_types: dict[str, EventTypeModel] = {}
    for event_name, type_raw in event_types_raw.items():
        if not isinstance(event_name, str) or not event_name.strip():
            _error("E_PARAMS_SCHEMA", "event type names must be non-empty strings")
        path_prefix = f"parameters.events.types.{event_name}"
        item = _mapping(type_raw, path_prefix, "E_PARAMS_SCHEMA")
        gain = _number(
            _required(item, "gain", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.gain",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=8.0,
        )
        duration_ms = _number(
            _required(item, "duration_ms", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.duration_ms",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=5000.0,
            inclusive_minimum=False,
        )
        attack_ms = _number(
            _required(item, "attack_ms", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.attack_ms",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=duration_ms,
            inclusive_minimum=False,
        )
        decay_ms = _number(
            _required(item, "decay_ms", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.decay_ms",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=10000.0,
            inclusive_minimum=False,
        )
        waveform = _string(
            _required(item, "waveform", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.waveform",
            "E_PARAMS_SCHEMA",
        )
        if waveform not in {"impact", "tone"}:
            _error(
                "E_PARAMS_SCHEMA",
                f"{path_prefix}.waveform must be 'impact' or 'tone'",
            )
        frequency_hz = _number(
            _required(item, "frequency_hz", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.frequency_hz",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=SAMPLE_RATE / 2.0,
            inclusive_minimum=False,
        )
        frequency_end_hz = _number(
            item.get("frequency_end_hz", frequency_hz),
            f"{path_prefix}.frequency_end_hz",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=SAMPLE_RATE / 2.0,
            inclusive_minimum=False,
        )
        noise_mix = _number(
            _required(item, "noise_mix", path_prefix, "E_PARAMS_SCHEMA"),
            f"{path_prefix}.noise_mix",
            "E_PARAMS_SCHEMA",
            minimum=0.0,
            maximum=1.0,
        )
        event_types[event_name] = EventTypeModel(
            gain=gain,
            duration_ms=duration_ms,
            attack_ms=attack_ms,
            decay_ms=decay_ms,
            waveform=waveform,
            frequency_hz=frequency_hz,
            frequency_end_hz=frequency_end_hz,
            noise_mix=noise_mix,
        )
    events = EventModel(
        overlap_window_ms=overlap_window_ms,
        overlap_attenuation=overlap_attenuation,
        types=event_types,
    )

    mix_raw = _mapping(
        _required(document, "mix", "parameters", "E_PARAMS_SCHEMA"),
        "parameters.mix",
        "E_PARAMS_SCHEMA",
    )
    engine_gain = _number(
        _required(mix_raw, "engine_gain", "parameters.mix", "E_PARAMS_SCHEMA"),
        "parameters.mix.engine_gain",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=16.0,
    )
    tire_gain = _number(
        _required(mix_raw, "tire_gain", "parameters.mix", "E_PARAMS_SCHEMA"),
        "parameters.mix.tire_gain",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=16.0,
    )
    event_gain = _number(
        _required(mix_raw, "event_gain", "parameters.mix", "E_PARAMS_SCHEMA"),
        "parameters.mix.event_gain",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=16.0,
    )
    master_gain = _number(
        _required(mix_raw, "master_gain", "parameters.mix", "E_PARAMS_SCHEMA"),
        "parameters.mix.master_gain",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=8.0,
    )
    soft_limit_threshold = _number(
        _required(
            mix_raw, "soft_limit_threshold", "parameters.mix", "E_PARAMS_SCHEMA"
        ),
        "parameters.mix.soft_limit_threshold",
        "E_PARAMS_SCHEMA",
        minimum=0.0,
        maximum=1.0,
        inclusive_minimum=False,
        inclusive_maximum=False,
    )
    mix = MixModel(
        engine_gain=engine_gain,
        tire_gain=tire_gain,
        event_gain=event_gain,
        master_gain=master_gain,
        soft_limit_threshold=soft_limit_threshold,
    )

    return ParamsModel(
        model_id=model_id,
        engine=engine,
        tires=tires,
        events=events,
        mix=mix,
    )


def load_telemetry(path: Path, params: ParamsModel) -> TelemetryModel:
    document = _mapping(
        _load_json(path, "E_TELEMETRY_IO", "E_TELEMETRY_JSON"),
        "telemetry",
        "E_TELEMETRY_SCHEMA",
    )
    schema_version = _integer(
        _required(document, "schema_version", "telemetry", "E_TELEMETRY_SCHEMA"),
        "telemetry.schema_version",
        "E_TELEMETRY_SCHEMA",
    )
    if schema_version != 1:
        _error("E_TELEMETRY_SCHEMA", f"unsupported telemetry schema_version={schema_version}")

    format_raw = _mapping(
        _required(document, "format", "telemetry", "E_TELEMETRY_SCHEMA"),
        "telemetry.format",
        "E_TELEMETRY_SCHEMA",
    )
    output_sr = _integer(
        _required(format_raw, "sample_rate", "telemetry.format", "E_TELEMETRY_SCHEMA"),
        "telemetry.format.sample_rate",
        "E_TELEMETRY_SCHEMA",
    )
    output_channels = _integer(
        _required(format_raw, "channels", "telemetry.format", "E_TELEMETRY_SCHEMA"),
        "telemetry.format.channels",
        "E_TELEMETRY_SCHEMA",
    )
    output_bits = _integer(
        _required(format_raw, "bit_depth", "telemetry.format", "E_TELEMETRY_SCHEMA"),
        "telemetry.format.bit_depth",
        "E_TELEMETRY_SCHEMA",
    )
    if output_sr != SAMPLE_RATE:
        _error("E_TELEMETRY_SCHEMA", f"sample_rate must be {SAMPLE_RATE}, got {output_sr}")
    if output_channels != CHANNELS:
        _error("E_TELEMETRY_SCHEMA", f"channels must be {CHANNELS}, got {output_channels}")
    if output_bits != 16:
        _error("E_TELEMETRY_SCHEMA", f"bit_depth must be 16, got {output_bits}")

    frame_interval_ms = _number(
        _required(
            document, "frame_interval_ms", "telemetry", "E_TELEMETRY_SCHEMA"
        ),
        "telemetry.frame_interval_ms",
        "E_TELEMETRY_SCHEMA",
        minimum=0.0,
        maximum=100.0,
        inclusive_minimum=False,
    )
    if not math.isclose(frame_interval_ms, FRAME_INTERVAL_MS, abs_tol=1e-9):
        _error(
            "E_TELEMETRY_SCHEMA",
            f"frame_interval_ms must be {FRAME_INTERVAL_MS}, got {frame_interval_ms}",
        )

    frame_count = _integer(
        _required(document, "frame_count", "telemetry", "E_TELEMETRY_SCHEMA"),
        "telemetry.frame_count",
        "E_TELEMETRY_SCHEMA",
    )
    if frame_count <= 0:
        _error("E_TELEMETRY_RANGE", "telemetry.frame_count must be greater than zero")
    if frame_count > 20_000_000:
        _error("E_TELEMETRY_RANGE", "telemetry.frame_count exceeds the supported limit")
    frames_raw = _required(document, "frames", "telemetry", "E_TELEMETRY_SCHEMA")
    if not isinstance(frames_raw, list):
        _error("E_TELEMETRY_SCHEMA", "telemetry.frames must be an array")
    if len(frames_raw) != frame_count:
        _error(
            "E_TELEMETRY_MISSING_FRAME",
            f"frame_count declares {frame_count}, but frames contains {len(frames_raw)} entries",
        )

    rpm_values = np.empty(frame_count, dtype=np.float64)
    throttle_values = np.empty(frame_count, dtype=np.float64)
    slip_values = np.empty(frame_count, dtype=np.float64)
    surface_names: list[str] = []
    previous_time = -math.inf
    interval_tolerance_ms = 1e-6

    for index, frame_raw in enumerate(frames_raw):
        path_prefix = f"telemetry.frames[{index}]"
        frame = _mapping(frame_raw, path_prefix, "E_TELEMETRY_FIELD")
        frame_index = _integer(
            _required(frame, "frame_index", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.frame_index",
            "E_TELEMETRY_FIELD",
        )
        if frame_index > index:
            _error(
                "E_TELEMETRY_MISSING_FRAME",
                f"expected frame_index {index}, got {frame_index}; one or more frames are missing",
            )
        if frame_index < index:
            _error(
                "E_TELEMETRY_TIMESTAMP",
                f"duplicate or out-of-order frame_index {frame_index} at position {index}",
            )

        time_ms = _number(
            _required(frame, "time_ms", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.time_ms",
            "E_TELEMETRY_FIELD",
        )
        if time_ms <= previous_time:
            _error(
                "E_TELEMETRY_TIMESTAMP",
                f"{path_prefix}.time_ms={time_ms} is not strictly increasing",
            )
        expected_time = index * frame_interval_ms
        if abs(time_ms - expected_time) > interval_tolerance_ms:
            if time_ms - previous_time > frame_interval_ms + interval_tolerance_ms:
                _error(
                    "E_TELEMETRY_MISSING_FRAME",
                    f"frame {index} leaves a time gap: expected {expected_time} ms, got {time_ms} ms",
                )
            _error(
                "E_TELEMETRY_TIMESTAMP",
                f"frame {index} timestamp {time_ms} ms does not match {expected_time} ms",
            )
        previous_time = time_ms

        rpm = _number(
            _required(frame, "rpm", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.rpm",
            "E_TELEMETRY_FIELD",
            minimum=params.engine.rpm_min,
            maximum=params.engine.rpm_max,
        )
        throttle = _number(
            _required(frame, "throttle", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.throttle",
            "E_TELEMETRY_FIELD",
            minimum=0.0,
            maximum=1.0,
        )
        slip = _number(
            _required(frame, "slip", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.slip",
            "E_TELEMETRY_FIELD",
            minimum=0.0,
            maximum=params.tires.max_slip,
        )
        surface = _string(
            _required(frame, "surface", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.surface",
            "E_TELEMETRY_FIELD",
        )
        if surface not in params.tires.profiles:
            _error(
                "E_TELEMETRY_RANGE",
                f"{path_prefix}.surface={surface!r} is not defined in parameters.tires.surfaces",
            )
        rpm_values[index] = rpm
        throttle_values[index] = throttle
        slip_values[index] = slip
        surface_names.append(surface)

    surface_set = tuple(sorted(set(surface_names)))
    surface_lookup = {name: idx for idx, name in enumerate(surface_set)}
    surface_indices = np.array(
        [surface_lookup[name] for name in surface_names], dtype=np.int16
    )

    events_raw = document.get("events", [])
    if not isinstance(events_raw, list):
        _error("E_TELEMETRY_SCHEMA", "telemetry.events must be an array when present")
    events: list[TelemetryEvent] = []
    frames_to_samples = frame_interval_ms / 1000.0 * SAMPLE_RATE
    for index, event_raw in enumerate(events_raw):
        path_prefix = f"telemetry.events[{index}]"
        event = _mapping(event_raw, path_prefix, "E_TELEMETRY_FIELD")
        event_frame_index = _integer(
            _required(event, "frame_index", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.frame_index",
            "E_TELEMETRY_FIELD",
        )
        if event_frame_index < 0 or event_frame_index >= frame_count:
            _error(
                "E_TELEMETRY_RANGE",
                f"{path_prefix}.frame_index must be in [0, {frame_count - 1}]",
            )
        event_type = _string(
            _required(event, "type", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.type",
            "E_TELEMETRY_FIELD",
        )
        if event_type not in params.events.types:
            _error(
                "E_TELEMETRY_RANGE",
                f"{path_prefix}.type={event_type!r} is not defined in parameters.events.types",
            )
        intensity = _number(
            _required(event, "intensity", path_prefix, "E_TELEMETRY_FIELD"),
            f"{path_prefix}.intensity",
            "E_TELEMETRY_FIELD",
            minimum=0.0,
            maximum=1.0,
        )
        pan = _number(
            event.get("pan", 0.0),
            f"{path_prefix}.pan",
            "E_TELEMETRY_FIELD",
            minimum=-1.0,
            maximum=1.0,
        )
        # A marker denotes the center of its 5 ms telemetry frame. Mapping to
        # an audio sample is therefore the half-frame interpolation below.
        sample_start = round_half_up(
            (event_frame_index + 0.5) * frames_to_samples
        )
        events.append(
            TelemetryEvent(
                frame_index=event_frame_index,
                event_type=event_type,
                intensity=intensity,
                pan=pan,
                sample_start=sample_start,
            )
        )

    frame_interval_s = frame_interval_ms / 1000.0
    samples_per_frame = frame_interval_s * SAMPLE_RATE
    total_samples = round_half_up(frame_count * samples_per_frame)
    if abs(total_samples - frame_count * samples_per_frame) > 1e-9:
        _error(
            "E_ALIGNMENT",
            "frame count and sample rate do not produce a whole sample count",
        )

    return TelemetryModel(
        frame_interval_ms=frame_interval_ms,
        frame_interval_s=frame_interval_s,
        frame_count=frame_count,
        samples_per_frame=samples_per_frame,
        total_samples=total_samples,
        rpm=rpm_values,
        throttle=throttle_values,
        slip=slip_values,
        surface_indices=surface_indices,
        surfaces=surface_set,
        events=tuple(events),
    )

