"""Scene JSON parsing, validation, and immutable scene models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import BusinessError, asset_missing, invalid_scene
from .paths import PathLike, normalize_path


SUPPORTED_BIT_DEPTHS = {8, 16, 24, 32}


def _first(data: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _require(data: Mapping[str, Any], names: Sequence[str], where: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise invalid_scene("missing required field", field=where, accepted_names=list(names))


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise invalid_scene("expected an object", field=where, actual_type=type(value).__name__)
    return value


def _as_list(value: Any, where: str) -> List[Any]:
    if not isinstance(value, list):
        raise invalid_scene("expected an array", field=where, actual_type=type(value).__name__)
    return value


def _as_number(value: Any, where: str, *, minimum: Optional[float] = None,
               maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid_scene("expected a finite number", field=where)
    number = float(value)
    if not math.isfinite(number):
        raise invalid_scene("number must be finite", field=where)
    if minimum is not None and number < minimum:
        raise invalid_scene("number is below the allowed minimum",
                            field=where, value=number, minimum=minimum)
    if maximum is not None and number > maximum:
        raise invalid_scene("number is above the allowed maximum",
                            field=where, value=number, maximum=maximum)
    return number


def _as_int(value: Any, where: str, *, minimum: Optional[int] = None,
            maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise invalid_scene("expected an integer", field=where)
    if minimum is not None and value < minimum:
        raise invalid_scene("integer is below the allowed minimum",
                            field=where, value=value, minimum=minimum)
    if maximum is not None and value > maximum:
        raise invalid_scene("integer is above the allowed maximum",
                            field=where, value=value, maximum=maximum)
    return value


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise invalid_scene("expected a boolean", field=where)


def _as_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise invalid_scene("expected a non-empty string", field=where)
    return value


def _vec3(value: Any, where: str) -> Tuple[float, float, float]:
    if isinstance(value, dict):
        value = _first(value, ("xyz", "value"))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise invalid_scene("expected a 3-element position", field=where)
    result = tuple(float(_as_number(item, "%s[%d]" % (where, index)))
                   for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def _vec_length(vector: Sequence[float]) -> float:
    return float(math.sqrt(sum(float(component) ** 2 for component in vector)))


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    kind: str
    frequency_hz: float
    amplitude: float
    phase_radians: float
    seed: int
    duration: Optional[float]
    loop: bool


@dataclass(frozen=True)
class AttenuationConfig:
    model: str
    reference_distance: float = 1.0
    rolloff: float = 1.0
    min_distance: float = 0.0
    max_distance: Optional[float] = 1000.0
    max_gain: float = 1.0


@dataclass(frozen=True)
class OcclusionConfig:
    open_cutoff_hz: float = 20000.0
    closed_cutoff_hz: float = 700.0


@dataclass(frozen=True)
class ReverbConfig:
    rt60_seconds: float
    pre_delay_seconds: float
    wet: float
    dry: float
    seed: int
    damping_hz: Optional[float]

    def canonical(self) -> Dict[str, Any]:
        return {
            "rt60_seconds": self.rt60_seconds,
            "pre_delay_seconds": self.pre_delay_seconds,
            "wet": self.wet,
            "dry": self.dry,
            "seed": self.seed,
            "damping_hz": self.damping_hz,
        }


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    shape: str
    center: Tuple[float, float, float]
    radius: float
    minimum: Tuple[float, float, float]
    maximum: Tuple[float, float, float]
    reverb: ReverbConfig
    lowpass_hz: Optional[float]

    def contains(self, point: Sequence[float]) -> bool:
        if self.shape in {"global", "everywhere"}:
            return True
        if self.shape == "sphere":
            distance = _vec_length(np.asarray(point, dtype=np.float64) - np.asarray(self.center))
            return distance <= self.radius
        if self.shape == "box":
            return all(self.minimum[axis] <= point[axis] <= self.maximum[axis] for axis in range(3))
        raise AssertionError("unsupported region shape reached runtime")


@dataclass(frozen=True)
class EmitterSpec:
    emitter_id: str
    asset_id: str
    position: Tuple[float, float, float]
    gain_db: float
    pan: Optional[float]
    occlusion_amount: float
    occlusion_closed_cutoff_hz: Optional[float]
    region_id: Optional[str]


@dataclass(frozen=True)
class Scene:
    source_path: Path
    sample_rate: int
    duration_seconds: float
    total_samples: int
    bit_depth: int
    listener_position: Tuple[float, float, float]
    listener_forward: Tuple[float, float, float]
    listener_up: Tuple[float, float, float]
    master_gain_db: float
    frame_size: int
    attenuation: AttenuationConfig
    occlusion: OcclusionConfig
    reverb: ReverbConfig
    assets: Mapping[str, AssetSpec]
    emitters: Tuple[EmitterSpec, ...]
    regions: Tuple[RegionSpec, ...]
    fingerprint: str
    raw_sha256: str

    def asset_for_emitter(self, emitter: EmitterSpec) -> AssetSpec:
        try:
            return self.assets[emitter.asset_id]
        except KeyError as exc:
            raise asset_missing(
                "emitter references an undefined audio asset",
                emitter_id=emitter.emitter_id,
                asset_id=emitter.asset_id,
            ) from exc

    def region_for_emitter(self, emitter: EmitterSpec) -> Optional[RegionSpec]:
        if emitter.region_id is not None:
            for region in self.regions:
                if region.region_id == emitter.region_id:
                    return region
            raise invalid_scene(
                "emitter references an undefined audio region",
                emitter_id=emitter.emitter_id,
                region_id=emitter.region_id,
            )
        for region in self.regions:
            if region.contains(emitter.position):
                return region
        return None

    def with_frame_size(self, frame_size: int) -> "Scene":
        """Return a scene with a CLI frame-size override.

        The original scene fingerprint is extended with the effective frame size
        so snapshots produced with a different block layout are not resumed.
        """

        if frame_size <= 0:
            raise invalid_scene("frame_size must be positive", frame_size=frame_size)
        effective = hashlib.sha256(
            ("%s:frame_size=%d" % (self.fingerprint, frame_size)).encode("utf-8")
        ).hexdigest()
        return replace(self, frame_size=frame_size, fingerprint=effective)


def _parse_reverb(value: Any, where: str, defaults: Optional[ReverbConfig] = None,
                  sample_rate: int = 48000) -> ReverbConfig:
    data = _as_mapping(value, where)
    rt60_raw = _first(
        data,
        ("rt60_seconds", "rt60", "reverb_time", "reverb_time_seconds", "rt60_ms"),
        defaults.rt60_seconds if defaults else None,
    )
    if rt60_raw is None:
        raise invalid_scene("missing reverb time", field=where + ".rt60_seconds")
    if any(key in data for key in ("rt60_seconds", "rt60", "reverb_time",
                                   "reverb_time_seconds")):
        rt60 = _as_number(rt60_raw, where + ".rt60_seconds", minimum=0.01, maximum=60.0)
    elif "rt60_ms" in data:
        rt60 = _as_number(rt60_raw, where + ".rt60_ms", minimum=10.0, maximum=60000.0) / 1000.0
    else:
        rt60 = defaults.rt60_seconds if defaults else 0.0

    pre_raw = _first(
        data,
        ("pre_delay_seconds", "pre_delay", "predelay", "predelay_seconds",
         "pre_delay_ms", "predelay_ms"),
        defaults.pre_delay_seconds if defaults else 0.0,
    )
    if any(key in data for key in ("pre_delay_seconds", "pre_delay", "predelay",
                                   "predelay_seconds")):
        pre_delay = _as_number(pre_raw, where + ".pre_delay_seconds",
                               minimum=0.0, maximum=2.0)
    elif "pre_delay_ms" in data or "predelay_ms" in data:
        pre_delay = _as_number(pre_raw, where + ".pre_delay_ms",
                               minimum=0.0, maximum=2000.0) / 1000.0
    else:
        pre_delay = defaults.pre_delay_seconds if defaults else 0.0

    wet_raw = _first(data, ("wet", "wet_level", "dry_wet", "wet_ratio",
                            "dry_wet_ratio", "wet_dry_ratio"))
    dry_raw = _first(data, ("dry", "dry_level", "dry_ratio"))
    if wet_raw is None and dry_raw is None:
        if defaults is None:
            raise invalid_scene("missing reverb wet/dry ratio", field=where)
        wet, dry = defaults.wet, defaults.dry
    elif wet_raw is None:
        dry = _as_number(dry_raw, where + ".dry", minimum=0.0, maximum=1.0)
        wet = 1.0 - dry
    else:
        wet = _as_number(wet_raw, where + ".wet", minimum=0.0, maximum=1.0)
        dry = (1.0 - wet) if dry_raw is None else _as_number(
            dry_raw, where + ".dry", minimum=0.0, maximum=1.0
        )

    seed = _as_int(_first(data, ("seed", "noise_seed"), defaults.seed if defaults else 0),
                   where + ".seed", minimum=0)

    damping_raw = _first(data, ("damping_hz", "lowpass_hz", "cutoff_hz"),
                         defaults.damping_hz if defaults else None)
    damping = None
    if damping_raw is not None:
        damping = _as_number(damping_raw, where + ".damping_hz", minimum=20.0)
        damping = min(damping, sample_rate * 0.45)
    return ReverbConfig(rt60, pre_delay, wet, dry, seed, damping)


def _parse_asset(asset_id: str, value: Any, where: str) -> AssetSpec:
    data = _as_mapping(value, where)
    kind = str(_first(data, ("type", "kind"), "sine")).strip().lower()
    if kind in {"tone", "oscillator"}:
        kind = "sine"
    if kind == "white":
        kind = "noise"
    if kind not in {"sine", "noise"}:
        if kind in {"file", "external", "wav"}:
            raise asset_missing(
                "external audio assets are not supported by this offline renderer",
                asset_id=asset_id,
                asset_type=kind,
            )
        raise invalid_scene("unsupported generated asset type", field=where, asset_type=kind)

    frequency = _as_number(_first(data, ("frequency_hz", "frequency", "freq"), 440.0),
                           where + ".frequency_hz", minimum=0.0)
    amplitude = _as_number(_first(data, ("amplitude", "gain"), 0.5),
                          where + ".amplitude", minimum=0.0, maximum=1.0)
    phase = _as_number(_first(data, ("phase_radians", "phase"), 0.0),
                       where + ".phase", minimum=-1000.0, maximum=1000.0)
    seed = _as_int(_first(data, ("seed", "noise_seed"), 0), where + ".seed", minimum=0)
    duration_raw = _first(data, ("duration_seconds", "duration"))
    duration = None if duration_raw is None else _as_number(
        duration_raw, where + ".duration_seconds", minimum=0.001
    )
    loop = _as_bool(_first(data, ("loop",), False), where + ".loop")
    return AssetSpec(asset_id, kind, frequency, amplitude, phase, seed, duration, loop)


def _parse_attenuation(value: Any) -> AttenuationConfig:
    where = "attenuation"
    data = _as_mapping(value, where)
    model = str(_first(data, ("model", "distance_model"), "inverse_distance")).strip().lower()
    aliases = {
        "inverse": "inverse_distance",
        "inverse-square": "inverse_distance",
        "inverse_square": "inverse_distance",
        "none": "none",
        "constant": "none",
        "linear": "linear",
    }
    model = aliases.get(model, model)
    if model not in {"inverse_distance", "linear", "none"}:
        raise invalid_scene("unsupported distance attenuation model",
                            field=where + ".model", model=model)
    reference = _as_number(_first(data, ("reference_distance", "reference_distance_m"), 1.0),
                           where + ".reference_distance", minimum=0.0001)
    rolloff = _as_number(_first(data, ("rolloff", "rolloff_factor"), 1.0),
                         where + ".rolloff", minimum=0.0, maximum=16.0)
    min_distance = _as_number(_first(data, ("min_distance", "min_distance_m"), 0.0),
                              where + ".min_distance", minimum=0.0)
    max_raw = _first(data, ("max_distance", "max_distance_m"), 1000.0)
    max_distance = None if max_raw is None else _as_number(
        max_raw, where + ".max_distance", minimum=0.0001
    )
    max_gain = _as_number(_first(data, ("max_gain", "gain_cap"), 1.0),
                          where + ".max_gain", minimum=0.0)
    return AttenuationConfig(model, reference, rolloff, min_distance, max_distance, max_gain)


def _parse_occlusion_config(value: Any) -> OcclusionConfig:
    if value is None:
        return OcclusionConfig()
    data = _as_mapping(value, "occlusion")
    open_cutoff = _as_number(
        _first(data, ("open_cutoff_hz", "max_cutoff_hz", "cutoff_open_hz"), 20000.0),
        "occlusion.open_cutoff_hz",
        minimum=20.0,
    )
    closed_cutoff = _as_number(
        _first(data, ("closed_cutoff_hz", "min_cutoff_hz", "cutoff_closed_hz"), 700.0),
        "occlusion.closed_cutoff_hz",
        minimum=20.0,
    )
    if closed_cutoff > open_cutoff:
        raise invalid_scene("occlusion closed cutoff must not exceed open cutoff",
                            field="occlusion", closed_cutoff_hz=closed_cutoff,
                            open_cutoff_hz=open_cutoff)
    return OcclusionConfig(open_cutoff, closed_cutoff)


def _parse_occlusion_amount(emitter: Mapping[str, Any], emitter_id: str) -> Tuple[float, Optional[float]]:
    where = "emitters[%s].occlusion" % emitter_id
    value = _first(emitter, ("occlusion", "occlusion_amount", "occlusion_polarity"))
    if value is None and "occluded" in emitter:
        value = emitter["occluded"]
    if value is None:
        return 0.0, None

    cutoff_override = None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"none", "open", "clear", "unoccluded"}:
            return 0.0, cutoff_override
        if normalized in {"blocked", "occluded", "positive", "full"}:
            return 1.0, cutoff_override
        raise invalid_scene("occlusion string is not recognized", field=where, value=value)
    if isinstance(value, dict):
        data = _as_mapping(value, where)
        if not _first(data, ("enabled",), True):
            return 0.0, None
        amount_raw = _first(data, ("amount", "occlusion", "polarity"))
        if amount_raw is None:
            amount = 1.0
        elif isinstance(amount_raw, bool):
            amount = 1.0 if amount_raw else 0.0
        elif isinstance(amount_raw, (int, float)):
            signed = float(amount_raw)
            amount = max(0.0, min(1.0, abs(signed)))
        else:
            raise invalid_scene("occlusion amount must be numeric", field=where + ".amount")
        cutoff_raw = _first(data, ("closed_cutoff_hz", "min_cutoff_hz", "cutoff_hz"))
        if cutoff_raw is not None:
            cutoff_override = _as_number(cutoff_raw, where + ".closed_cutoff_hz", minimum=20.0)
        return amount, cutoff_override

    if isinstance(value, bool):
        return (1.0 if value else 0.0), cutoff_override
    if isinstance(value, (int, float)):
        amount = float(value)
        if amount < 0.0:
            amount = min(1.0, -amount)
        else:
            amount = min(1.0, amount)
        return amount, cutoff_override
    raise invalid_scene("occlusion must be a number, boolean, or object", field=where)


def _parse_pan(value: Any, where: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"auto", "automatic", "direction"}:
            return None
        raise invalid_scene("pan string must be 'auto'", field=where)
    if isinstance(value, dict):
        data = _as_mapping(value, where)
        mode = str(_first(data, ("mode",), "value")).lower()
        if mode in {"auto", "direction"}:
            return None
        value = _first(data, ("value", "pan"))
        if value is None:
            raise invalid_scene("pan object is missing value", field=where)
    return _as_number(value, where, minimum=-1.0, maximum=1.0)


def _parse_region(region_id: str, value: Any, global_reverb: ReverbConfig,
                  sample_rate: int) -> RegionSpec:
    where = "regions[%s]" % region_id
    data = _as_mapping(value, where)
    shape = str(_first(data, ("shape", "type"), "sphere")).strip().lower()
    if shape in {"everywhere", "world"}:
        shape = "global"
    if shape in {"aabb", "cuboid"}:
        shape = "box"
    if shape not in {"sphere", "box", "global"}:
        raise invalid_scene("unsupported audio region shape", field=where + ".shape", shape=shape)

    center = _vec3(_first(data, ("center", "position"), [0.0, 0.0, 0.0]), where + ".center")
    radius = _as_number(_first(data, ("radius",), 1.0), where + ".radius", minimum=0.0)
    minimum = center
    maximum = center
    if shape == "box":
        minimum = _vec3(_first(data, ("min", "minimum"), [-1.0, -1.0, -1.0]),
                        where + ".minimum")
        maximum = _vec3(_first(data, ("max", "maximum"), [1.0, 1.0, 1.0]),
                        where + ".maximum")
        if any(minimum[axis] > maximum[axis] for axis in range(3)):
            raise invalid_scene("box region minimum exceeds maximum", field=where)

    reverb_value = _first(data, ("reverb", "reverb_params"))
    reverb = global_reverb
    if reverb_value is not None:
        if isinstance(reverb_value, bool):
            if not reverb_value:
                reverb = ReverbConfig(
                    global_reverb.rt60_seconds,
                    global_reverb.pre_delay_seconds,
                    0.0,
                    global_reverb.dry,
                    global_reverb.seed,
                    global_reverb.damping_hz,
                )
        else:
            reverb = _parse_reverb(reverb_value, where + ".reverb", global_reverb, sample_rate)

    lowpass_value = _first(data, ("lowpass", "lowpass_hz", "send_lowpass_hz"))
    lowpass_hz = None
    if lowpass_value is not None:
        if isinstance(lowpass_value, dict):
            raw = _first(lowpass_value, ("cutoff_hz", "frequency_hz", "hz"))
            if raw is None:
                raise invalid_scene("region lowpass is missing cutoff_hz", field=where + ".lowpass")
        else:
            raw = lowpass_value
        lowpass_hz = _as_number(raw, where + ".lowpass.cutoff_hz", minimum=20.0)
        lowpass_hz = min(lowpass_hz, sample_rate * 0.45)

    return RegionSpec(region_id, shape, center, radius, minimum, maximum, reverb, lowpass_hz)


def _parse_scene_data(data: Any, source_path: Path) -> Scene:
    root = _as_mapping(data, "$")
    sample_rate = _as_int(_first(root, ("sample_rate", "sampleRate", "sample_rate_hz")),
                          "sample_rate", minimum=8000, maximum=192000)

    duration_raw = _first(root, ("duration_seconds", "duration", "total_duration"))
    total_samples_raw = _first(root, ("total_samples", "sample_frames"))
    if duration_raw is None and total_samples_raw is None:
        raise invalid_scene("scene must define duration or total_samples", field="duration")
    if total_samples_raw is not None:
        total_samples = _as_int(total_samples_raw, "total_samples", minimum=1)
        duration = total_samples / sample_rate
        if duration_raw is not None:
            supplied_duration = _as_number(duration_raw, "duration_seconds", minimum=0.0001)
            expected = int(math.ceil(supplied_duration * sample_rate))
            if expected != total_samples:
                raise invalid_scene("duration and total_samples are inconsistent",
                                    duration_seconds=supplied_duration,
                                    total_samples=total_samples,
                                    sample_rate=sample_rate)
    else:
        duration = _as_number(duration_raw, "duration_seconds", minimum=0.0001, maximum=3600.0)
        total_samples = int(math.ceil(duration * sample_rate))

    if total_samples <= 0:
        raise invalid_scene("scene duration produces no samples", field="duration")

    bit_depth = _as_int(_first(root, ("bit_depth", "bitDepth", "bits_per_sample"), 16),
                        "bit_depth")
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise invalid_scene("unsupported WAV bit depth", field="bit_depth",
                            bit_depth=bit_depth, supported=sorted(SUPPORTED_BIT_DEPTHS))

    listener = _as_mapping(_require(root, ("listener",), "listener"), "listener")
    listener_position = _vec3(_require(listener, ("position", "pos"), "listener.position"),
                              "listener.position")
    listener_forward = _vec3(_first(listener, ("forward", "front"), [0.0, 0.0, -1.0]),
                             "listener.forward")
    listener_up = _vec3(_first(listener, ("up",), [0.0, 1.0, 0.0]), "listener.up")
    if _vec_length(listener_forward) <= 1e-12 or _vec_length(listener_up) <= 1e-12:
        raise invalid_scene("listener orientation vectors must be non-zero")

    master_gain_db = _as_number(_first(root, ("master_gain_db", "master_gain_db"), 0.0),
                                "master_gain_db")
    frame_size = _as_int(_first(root, ("frame_size", "audio_frame_size"), 1024),
                         "frame_size", minimum=1, maximum=65536)

    attenuation = _parse_attenuation(_first(root, ("attenuation", "distance_attenuation"),
                                           {"model": "inverse_distance"}))
    occlusion = _parse_occlusion_config(_first(root, ("occlusion", "occlusion_filter")))

    global_reverb_value = _first(root, ("reverb", "global_reverb", "reverb_bus"))
    if global_reverb_value is None:
        raise invalid_scene("scene must define global reverb parameters", field="reverb")
    global_reverb = _parse_reverb(global_reverb_value, "reverb", None, sample_rate)

    assets_value = _require(root, ("assets", "audio_assets"), "assets")
    assets: Dict[str, AssetSpec] = {}
    if isinstance(assets_value, dict):
        asset_items = list(assets_value.items())
        for asset_id, asset_value in asset_items:
            if not isinstance(asset_id, str) or not asset_id:
                raise invalid_scene("asset keys must be non-empty strings", field="assets")
            assets[asset_id] = _parse_asset(asset_id, asset_value, "assets[%s]" % asset_id)
    else:
        for index, asset_value in enumerate(_as_list(assets_value, "assets")):
            asset_data = _as_mapping(asset_value, "assets[%d]" % index)
            asset_id = _as_string(
                _require(asset_data, ("id", "asset_id", "name"), "assets[%d].id" % index),
                "assets[%d].id" % index,
            )
            if asset_id in assets:
                raise invalid_scene("duplicate asset id", field="assets", asset_id=asset_id)
            assets[asset_id] = _parse_asset(asset_id, asset_data, "assets[%d]" % index)
    if not assets:
        raise invalid_scene("scene has no audio assets", field="assets")

    region_values = _first(root, ("regions", "audio_regions", "zones"), [])
    regions: List[RegionSpec] = []
    region_ids: set[str] = set()
    for index, region_value in enumerate(_as_list(region_values, "regions")):
        region_data = _as_mapping(region_value, "regions[%d]" % index)
        region_id = _as_string(
            _require(region_data, ("id", "region_id", "name"), "regions[%d].id" % index),
            "regions[%d].id" % index,
        )
        if region_id in region_ids:
            raise invalid_scene("duplicate region id", field="regions", region_id=region_id)
        region_ids.add(region_id)
        regions.append(_parse_region(region_id, region_data, global_reverb, sample_rate))

    emitter_values = _require(root, ("emitters", "sources", "audio_emitters"), "emitters")
    emitters: List[EmitterSpec] = []
    emitter_ids: set[str] = set()
    for index, emitter_value in enumerate(_as_list(emitter_values, "emitters")):
        where = "emitters[%d]" % index
        emitter_data = _as_mapping(emitter_value, where)
        emitter_id = _as_string(
            _require(emitter_data, ("id", "emitter_id", "name"), where + ".id"),
            where + ".id",
        )
        if emitter_id in emitter_ids:
            raise invalid_scene("duplicate emitter id", field="emitters", emitter_id=emitter_id)
        emitter_ids.add(emitter_id)
        asset_id = _as_string(
            _require(emitter_data, ("asset", "asset_id", "audio_asset"), where + ".asset"),
            where + ".asset",
        )
        if asset_id not in assets:
            raise asset_missing(
                "emitter references an undefined audio asset",
                emitter_id=emitter_id,
                asset_id=asset_id,
            )
        position = _vec3(
            _require(emitter_data, ("position", "source_position", "pos"), where + ".position"),
            where + ".position",
        )
        gain_db = _as_number(_first(emitter_data, ("gain_db", "gainDb"), 0.0),
                             where + ".gain_db")
        if "gain" in emitter_data and "gain_db" not in emitter_data:
            gain = _as_number(emitter_data["gain"], where + ".gain", minimum=0.0)
            gain_db = 20.0 * math.log10(gain) if gain > 0.0 else -120.0
        pan = _parse_pan(_first(emitter_data, ("pan",), None), where + ".pan")
        occlusion_amount, cutoff_override = _parse_occlusion_amount(emitter_data, emitter_id)
        region_id_raw = _first(emitter_data, ("region", "region_id", "audio_region"))
        region_id = None
        if region_id_raw is not None:
            region_id = _as_string(region_id_raw, where + ".region_id")
            if region_id not in region_ids:
                raise invalid_scene("emitter references an undefined audio region",
                                    field=where + ".region_id", region_id=region_id)
        emitters.append(EmitterSpec(
            emitter_id, asset_id, position, gain_db, pan, occlusion_amount,
            cutoff_override, region_id
        ))

    normalized = {
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "total_samples": total_samples,
        "bit_depth": bit_depth,
        "listener": {
            "position": list(listener_position),
            "forward": list(listener_forward),
            "up": list(listener_up),
        },
        "master_gain_db": master_gain_db,
        "frame_size": frame_size,
        "attenuation": attenuation.__dict__,
        "occlusion": occlusion.__dict__,
        "reverb": global_reverb.canonical(),
        "assets": {asset_id: asset.__dict__ for asset_id, asset in assets.items()},
        "regions": [
            {
                "id": region.region_id,
                "shape": region.shape,
                "center": list(region.center),
                "radius": region.radius,
                "minimum": list(region.minimum),
                "maximum": list(region.maximum),
                "reverb": region.reverb.canonical(),
                "lowpass_hz": region.lowpass_hz,
            }
            for region in regions
        ],
        "emitters": [emitter.__dict__ for emitter in emitters],
    }
    fingerprint = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw_sha256 = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return Scene(
        source_path=source_path,
        sample_rate=sample_rate,
        duration_seconds=duration,
        total_samples=total_samples,
        bit_depth=bit_depth,
        listener_position=listener_position,
        listener_forward=listener_forward,
        listener_up=listener_up,
        master_gain_db=master_gain_db,
        frame_size=frame_size,
        attenuation=attenuation,
        occlusion=occlusion,
        reverb=global_reverb,
        assets=assets,
        emitters=tuple(emitters),
        regions=tuple(regions),
        fingerprint=fingerprint,
        raw_sha256=raw_sha256,
    )


def load_scene(path: PathLike) -> Scene:
    source_path = normalize_path(path)
    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise invalid_scene(
            "scene file cannot be read",
            path=str(source_path),
            reason=exc.strerror or str(exc),
        ) from exc
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise invalid_scene(
            "scene file is not valid JSON",
            path=str(source_path),
            reason=getattr(exc, "msg", str(exc)),
            line=getattr(exc, "lineno", None),
            column=getattr(exc, "colno", None),
        ) from exc
    return _parse_scene_data(data, source_path)


def safe_track_filename(emitter_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", emitter_id).strip("._")
    if not cleaned:
        cleaned = "emitter"
    return "dry_%s.wav" % cleaned
