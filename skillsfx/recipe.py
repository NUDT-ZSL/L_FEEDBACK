"""JSON recipe parsing and strict business validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .errors import SkillSFXError
from .wavio import sha256_file


SUPPORTED_SAMPLE_RATES = (44100, 48000)
SUPPORTED_BIT_DEPTHS = (16, 24, 32)


def _where(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SkillSFXError(
            "SFX-E1002",
            "expected a JSON object",
            {"field": where, "actual_type": type(value).__name__},
        )
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    where: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    if missing:
        raise SkillSFXError(
            "SFX-E1002",
            "recipe is missing required field",
            {"field": _where(where, missing[0]), "missing_fields": missing},
        )
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise SkillSFXError(
            "SFX-E1002",
            "recipe contains an unknown field",
            {"field": _where(where, unknown[0]), "unknown_fields": unknown},
        )


def _finite_float(
    value: Any,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillSFXError(
            "SFX-E1002",
            "expected a numeric value",
            {"field": where, "actual_type": type(value).__name__},
        )
    result = float(value)
    if not math.isfinite(result):
        raise SkillSFXError(
            "SFX-E1003",
            "numeric value must be finite",
            {"field": where, "value": value},
        )
    if minimum is not None and result < minimum:
        raise SkillSFXError(
            "SFX-E1003",
            "numeric value is below the accepted minimum",
            {"field": where, "value": result, "minimum": minimum},
        )
    if maximum is not None and result > maximum:
        raise SkillSFXError(
            "SFX-E1003",
            "numeric value is above the accepted maximum",
            {"field": where, "value": result, "maximum": maximum},
        )
    return result


def _integer(
    value: Any,
    where: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkillSFXError(
            "SFX-E1002",
            "expected an integer",
            {"field": where, "actual_type": type(value).__name__},
        )
    if minimum is not None and value < minimum:
        raise SkillSFXError(
            "SFX-E1003",
            "integer value is below the accepted minimum",
            {"field": where, "value": value, "minimum": minimum},
        )
    if maximum is not None and value > maximum:
        raise SkillSFXError(
            "SFX-E1003",
            "integer value is above the accepted maximum",
            {"field": where, "value": value, "maximum": maximum},
        )
    return value


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SkillSFXError(
            "SFX-E1002",
            "expected a string",
            {"field": where, "actual_type": type(value).__name__},
        )
    if not allow_empty and not value.strip():
        raise SkillSFXError(
            "SFX-E1003",
            "string value must not be empty",
            {"field": where},
        )
    return value


@dataclass(frozen=True)
class OutputSpec:
    sample_rate: int
    bit_depth: int | str
    channels: int
    duration_ms: float

    @property
    def sample_format(self) -> str:
        return "float" if self.bit_depth == "float32" else "pcm"

    @property
    def duration_frames(self) -> int:
        return int(math.floor(self.duration_ms * self.sample_rate / 1000.0 + 0.5))

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / float(self.sample_rate)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sample_format"] = self.sample_format
        result["duration_frames"] = self.duration_frames
        return result


@dataclass(frozen=True)
class CompressorParams:
    threshold_db: float
    ratio: float
    knee_db: float = 6.0
    attack_ms: float = 5.0
    release_ms: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LimiterParams:
    ceiling_dbtp: float
    lookahead_ms: float = 5.0
    attack_ms: float = 0.8
    release_ms: float = 80.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ceiling_linear(self) -> float:
        return 10.0 ** (self.ceiling_dbtp / 20.0)


@dataclass(frozen=True)
class BusParams:
    compressor: CompressorParams
    tail_fade_ms: float
    limiter: LimiterParams

    def to_dict(self) -> dict[str, Any]:
        return {
            "compressor": self.compressor.to_dict(),
            "tail_fade_ms": self.tail_fade_ms,
            "limiter": self.limiter.to_dict(),
        }


@dataclass(frozen=True)
class NormalizationParams:
    target_lufs: float | None
    target_rms_dbfs: float | None
    min_gain_db: float = -24.0
    max_gain_db: float = 24.0

    @property
    def metric(self) -> str:
        return "lufs" if self.target_lufs is not None else "rms_dbfs"

    @property
    def target(self) -> float:
        if self.target_lufs is not None:
            return self.target_lufs
        assert self.target_rms_dbfs is not None
        return self.target_rms_dbfs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TailHarmonicConfig:
    enabled: bool
    fundamental_hz: float
    ratios: tuple[float, ...]
    gain_db_min: float
    gain_db_max: float
    decay_ms: float
    attack_ms: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ratios"] = list(self.ratios)
        return result


@dataclass(frozen=True)
class RandomizationParams:
    gain_jitter_db: float
    time_jitter_ms: float
    tail_harmonics: TailHarmonicConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain_jitter_db": self.gain_jitter_db,
            "time_jitter_ms": self.time_jitter_ms,
            "tail_harmonics": self.tail_harmonics.to_dict(),
        }


@dataclass(frozen=True)
class EventSpec:
    index: int
    role: str
    source_path: Path
    source_ref: str
    offset_ms: float
    duration_ms: float | None
    gain: float
    channel_map: Any
    channel_map_descriptor: Any

    def to_manifest_base(self) -> dict[str, Any]:
        return {
            "event_index": self.index,
            "role": self.role,
            "source": {
                "path": self.source_ref,
                "resolved_path": str(self.source_path),
            },
            "offset_ms": self.offset_ms,
            "duration_ms": self.duration_ms,
            "gain_linear": self.gain,
            "channel_map": self.channel_map_descriptor,
        }


@dataclass(frozen=True)
class Recipe:
    path: Path
    sha256: str
    schema_version: int
    skill_name: str
    variant_count: int
    seed: int
    output: OutputSpec
    events: tuple[EventSpec, ...]
    randomization: RandomizationParams
    bus_by_level: Mapping[str, BusParams]
    normalization: NormalizationParams
    metadata: dict[str, Any] = field(default_factory=dict)

    def bus_for_level(self, level: int) -> BusParams:
        key = str(level)
        if key not in self.bus_by_level:
            raise SkillSFXError(
                "SFX-E1003",
                "requested skill level has no bus processing chain",
                {
                    "level": level,
                    "available_levels": sorted(
                        int(item) for item in self.bus_by_level
                    ),
                },
            )
        return self.bus_by_level[key]

    def to_manifest_summary(self, level: int) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "skill_name": self.skill_name,
            "variant_count": self.variant_count,
            "seed": self.seed,
            "level": level,
            "output": self.output.to_dict(),
            "randomization": self.randomization.to_dict(),
            "normalization": self.normalization.to_dict(),
            "bus": self.bus_for_level(level).to_dict(),
        }


def _parse_output(value: Any) -> OutputSpec:
    obj = _require_mapping(value, "output")
    _check_keys(
        obj,
        where="output",
        required={"sample_rate", "bit_depth", "channels", "duration_ms"},
        optional={"sample_format"},
    )
    sample_rate = _integer(obj["sample_rate"], "output.sample_rate")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise SkillSFXError(
            "SFX-E1003",
            "unsupported output sample rate",
            {
                "field": "output.sample_rate",
                "value": sample_rate,
                "supported": list(SUPPORTED_SAMPLE_RATES),
            },
        )
    channels = _integer(obj["channels"], "output.channels", minimum=1, maximum=2)
    duration_ms = _finite_float(
        obj["duration_ms"],
        "output.duration_ms",
        minimum=1.0,
        maximum=3_600_000.0,
    )

    raw_depth = obj["bit_depth"]
    sample_format = obj.get("sample_format")
    if sample_format is not None:
        sample_format = _string(sample_format, "output.sample_format")
        if sample_format not in ("pcm", "float"):
            raise SkillSFXError(
                "SFX-E1003",
                "sample_format must be 'pcm' or 'float'",
                {"field": "output.sample_format", "value": sample_format},
            )

    if raw_depth == "float32":
        if sample_format not in (None, "float"):
            raise SkillSFXError(
                "SFX-E1003",
                "bit_depth float32 conflicts with sample_format",
                {
                    "field": "output.sample_format",
                    "bit_depth": raw_depth,
                    "sample_format": sample_format,
                },
            )
        bit_depth: int | str = "float32"
    else:
        bit_depth_int = _integer(raw_depth, "output.bit_depth")
        if bit_depth_int not in SUPPORTED_BIT_DEPTHS:
            raise SkillSFXError(
                "SFX-E1003",
                "unsupported output bit depth",
                {
                    "field": "output.bit_depth",
                    "value": bit_depth_int,
                    "supported": list(SUPPORTED_BIT_DEPTHS) + ["float32"],
                },
            )
        if sample_format == "float":
            if bit_depth_int != 32:
                raise SkillSFXError(
                    "SFX-E1003",
                    "float output requires 32-bit samples",
                    {
                        "field": "output.bit_depth",
                        "value": bit_depth_int,
                    },
                )
            bit_depth = "float32"
        else:
            bit_depth = bit_depth_int

    if duration_ms * sample_rate / 1000.0 < 1.0:
        raise SkillSFXError(
            "SFX-E1003",
            "output duration is shorter than one sample",
            {"field": "output.duration_ms", "value": duration_ms},
        )
    return OutputSpec(sample_rate, bit_depth, channels, duration_ms)


def _parse_compressor(value: Any) -> CompressorParams:
    obj = _require_mapping(value, "bus.compressor")
    _check_keys(
        obj,
        where="bus.compressor",
        required={"threshold_db", "ratio"},
        optional={"knee_db", "attack_ms", "release_ms"},
    )
    threshold = _finite_float(
        obj["threshold_db"],
        "bus.compressor.threshold_db",
        minimum=-80.0,
        maximum=0.0,
    )
    ratio = _finite_float(
        obj["ratio"],
        "bus.compressor.ratio",
        minimum=1.0,
        maximum=40.0,
    )
    knee = _finite_float(
        obj.get("knee_db", 6.0),
        "bus.compressor.knee_db",
        minimum=0.0,
        maximum=40.0,
    )
    attack = _finite_float(
        obj.get("attack_ms", 5.0),
        "bus.compressor.attack_ms",
        minimum=0.05,
        maximum=1000.0,
    )
    release = _finite_float(
        obj.get("release_ms", 100.0),
        "bus.compressor.release_ms",
        minimum=1.0,
        maximum=10000.0,
    )
    return CompressorParams(threshold, ratio, knee, attack, release)


def _parse_limiter(value: Any) -> LimiterParams:
    obj = _require_mapping(value, "bus.limiter")
    _check_keys(
        obj,
        where="bus.limiter",
        required={"ceiling_dbtp"},
        optional={"lookahead_ms", "attack_ms", "release_ms"},
    )
    ceiling = _finite_float(
        obj["ceiling_dbtp"],
        "bus.limiter.ceiling_dbtp",
        minimum=-20.0,
        maximum=-0.1,
    )
    lookahead = _finite_float(
        obj.get("lookahead_ms", 5.0),
        "bus.limiter.lookahead_ms",
        minimum=0.2,
        maximum=50.0,
    )
    attack = _finite_float(
        obj.get("attack_ms", 0.8),
        "bus.limiter.attack_ms",
        minimum=0.05,
        maximum=lookahead,
    )
    release = _finite_float(
        obj.get("release_ms", 80.0),
        "bus.limiter.release_ms",
        minimum=5.0,
        maximum=5000.0,
    )
    return LimiterParams(ceiling, lookahead, attack, release)


def _parse_bus(value: Any, where: str) -> BusParams:
    obj = _require_mapping(value, where)
    _check_keys(
        obj,
        where=where,
        required={"compressor", "tail_fade_ms", "limiter"},
    )
    compressor = _parse_compressor(obj["compressor"])
    fade_ms = _finite_float(
        obj["tail_fade_ms"],
        f"{where}.tail_fade_ms",
        minimum=0.0,
        maximum=60_000.0,
    )
    limiter = _parse_limiter(obj["limiter"])
    return BusParams(compressor, fade_ms, limiter)


def _parse_normalization(value: Any) -> NormalizationParams:
    obj = _require_mapping(value, "normalization")
    _check_keys(
        obj,
        where="normalization",
        required=set(),
        optional={
            "target_lufs",
            "target_rms_dbfs",
            "min_gain_db",
            "max_gain_db",
        },
    )
    has_lufs = "target_lufs" in obj
    has_rms = "target_rms_dbfs" in obj
    if has_lufs == has_rms:
        raise SkillSFXError(
            "SFX-E1003",
            "normalization must specify exactly one loudness target",
            {
                "field": "normalization",
                "target_lufs_present": has_lufs,
                "target_rms_dbfs_present": has_rms,
            },
        )
    target_lufs = (
        _finite_float(
            obj["target_lufs"],
            "normalization.target_lufs",
            minimum=-60.0,
            maximum=0.0,
        )
        if has_lufs
        else None
    )
    target_rms = (
        _finite_float(
            obj["target_rms_dbfs"],
            "normalization.target_rms_dbfs",
            minimum=-60.0,
            maximum=0.0,
        )
        if has_rms
        else None
    )
    minimum = _finite_float(
        obj.get("min_gain_db", -24.0),
        "normalization.min_gain_db",
        minimum=-48.0,
        maximum=0.0,
    )
    maximum = _finite_float(
        obj.get("max_gain_db", 24.0),
        "normalization.max_gain_db",
        minimum=0.0,
        maximum=48.0,
    )
    if minimum > maximum:
        raise SkillSFXError(
            "SFX-E1003",
            "normalization minimum gain exceeds maximum gain",
            {
                "field": "normalization",
                "min_gain_db": minimum,
                "max_gain_db": maximum,
            },
        )
    return NormalizationParams(target_lufs, target_rms, minimum, maximum)


def _parse_tail_harmonics(value: Any) -> TailHarmonicConfig:
    obj = _require_mapping(value, "randomization.tail_harmonics")
    _check_keys(
        obj,
        where="randomization.tail_harmonics",
        required={
            "enabled",
            "fundamental_hz",
            "ratios",
            "gain_db_min",
            "gain_db_max",
            "decay_ms",
        },
        optional={"attack_ms"},
    )
    enabled = obj["enabled"]
    if not isinstance(enabled, bool):
        raise SkillSFXError(
            "SFX-E1002",
            "expected a boolean",
            {
                "field": "randomization.tail_harmonics.enabled",
                "actual_type": type(enabled).__name__,
            },
        )
    fundamental = _finite_float(
        obj["fundamental_hz"],
        "randomization.tail_harmonics.fundamental_hz",
        minimum=20.0,
        maximum=20_000.0,
    )
    raw_ratios = obj["ratios"]
    if not isinstance(raw_ratios, list) or not raw_ratios:
        raise SkillSFXError(
            "SFX-E1002",
            "tail harmonic ratios must be a non-empty array",
            {"field": "randomization.tail_harmonics.ratios"},
        )
    ratios = tuple(
        _finite_float(
            item,
            f"randomization.tail_harmonics.ratios[{index}]",
            minimum=0.01,
            maximum=64.0,
        )
        for index, item in enumerate(raw_ratios)
    )
    gain_min = _finite_float(
        obj["gain_db_min"],
        "randomization.tail_harmonics.gain_db_min",
        minimum=-100.0,
        maximum=0.0,
    )
    gain_max = _finite_float(
        obj["gain_db_max"],
        "randomization.tail_harmonics.gain_db_max",
        minimum=-100.0,
        maximum=0.0,
    )
    if gain_min > gain_max:
        raise SkillSFXError(
            "SFX-E1003",
            "tail harmonic gain minimum exceeds maximum",
            {
                "field": "randomization.tail_harmonics",
                "gain_db_min": gain_min,
                "gain_db_max": gain_max,
            },
        )
    decay = _finite_float(
        obj["decay_ms"],
        "randomization.tail_harmonics.decay_ms",
        minimum=5.0,
        maximum=60_000.0,
    )
    attack = _finite_float(
        obj.get("attack_ms", 2.0),
        "randomization.tail_harmonics.attack_ms",
        minimum=0.0,
        maximum=1000.0,
    )
    return TailHarmonicConfig(
        enabled,
        fundamental,
        ratios,
        gain_min,
        gain_max,
        decay,
        attack,
    )


def _parse_randomization(value: Any) -> RandomizationParams:
    obj = _require_mapping(value, "randomization")
    _check_keys(
        obj,
        where="randomization",
        required={"gain_jitter_db", "time_jitter_ms", "tail_harmonics"},
    )
    gain_jitter = _finite_float(
        obj["gain_jitter_db"],
        "randomization.gain_jitter_db",
        minimum=0.0,
        maximum=24.0,
    )
    time_jitter = _finite_float(
        obj["time_jitter_ms"],
        "randomization.time_jitter_ms",
        minimum=0.0,
        maximum=500.0,
    )
    harmonics = _parse_tail_harmonics(obj["tail_harmonics"])
    return RandomizationParams(gain_jitter, time_jitter, harmonics)


def _parse_event(
    value: Any,
    index: int,
    recipe_dir: Path,
) -> EventSpec:
    where = f"events[{index}]"
    obj = _require_mapping(value, where)
    _check_keys(
        obj,
        where=where,
        required={"source", "offset_ms", "gain", "channel_map"},
        optional={"duration_ms", "role"},
    )
    source_ref = _string(obj["source"], f"{where}.source")
    source_path = Path(source_ref)
    if not source_path.is_absolute():
        source_path = recipe_dir / source_path
    source_path = source_path.resolve()
    offset_ms = _finite_float(
        obj["offset_ms"],
        f"{where}.offset_ms",
        maximum=3_600_000.0,
    )
    if offset_ms < 0.0:
        raise SkillSFXError(
            "SFX-E2003",
            "event offset must not be negative",
            {
                "event_index": index,
                "source_path": source_ref,
                "resolved_source_path": str(source_path),
                "offset_ms": offset_ms,
            },
        )
    duration_raw = obj.get("duration_ms")
    duration_ms = (
        _finite_float(
            duration_raw,
            f"{where}.duration_ms",
            minimum=1.0,
            maximum=3_600_000.0,
        )
        if duration_raw is not None
        else None
    )
    gain = _finite_float(
        obj["gain"],
        f"{where}.gain",
        minimum=0.0,
        maximum=64.0,
    )
    role = _string(obj.get("role", "event"), f"{where}.role").lower()
    channel_map = obj["channel_map"]
    return EventSpec(
        index=index,
        role=role,
        source_path=source_path,
        source_ref=source_ref,
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        gain=gain,
        channel_map=channel_map,
        channel_map_descriptor=channel_map,
    )


def load_recipe(path: str | Path) -> Recipe:
    recipe_path = Path(path).resolve()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        raw_text = recipe_path.read_text(encoding="utf-8")
        raw = json.loads(
            raw_text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except OSError as exc:
        raise SkillSFXError(
            "SFX-E1001",
            "recipe file could not be read",
            {"recipe_path": str(recipe_path), "reason": str(exc)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise SkillSFXError(
            "SFX-E1001",
            "recipe file is not valid JSON",
            {
                "recipe_path": str(recipe_path),
                "line": exc.lineno,
                "column": exc.colno,
                "reason": exc.msg,
            },
        ) from exc
    except ValueError as exc:
        raise SkillSFXError(
            "SFX-E1001",
            "recipe file is not strict JSON",
            {"recipe_path": str(recipe_path), "reason": str(exc)},
        ) from exc

    obj = _require_mapping(raw, "recipe")
    _check_keys(
        obj,
        where="recipe",
        required={
            "schema_version",
            "skill_name",
            "variant_count",
            "seed",
            "output",
            "events",
            "randomization",
            "bus_by_level",
            "normalization",
        },
        optional={"metadata"},
    )
    schema_version = _integer(
        obj["schema_version"],
        "schema_version",
        minimum=1,
        maximum=1,
    )
    skill_name = _string(obj["skill_name"], "skill_name")
    variant_count = _integer(
        obj["variant_count"],
        "variant_count",
        minimum=1,
        maximum=1000,
    )
    seed = _integer(
        obj["seed"],
        "seed",
        minimum=0,
        maximum=2**63 - 1,
    )
    output = _parse_output(obj["output"])

    raw_events = obj["events"]
    if not isinstance(raw_events, list) or not raw_events:
        raise SkillSFXError(
            "SFX-E1002",
            "events must be a non-empty array",
            {"field": "events"},
        )
    if len(raw_events) > 128:
        raise SkillSFXError(
            "SFX-E1003",
            "recipe contains too many event segments",
            {"field": "events", "count": len(raw_events), "maximum": 128},
        )
    events = tuple(
        _parse_event(item, index, recipe_path.parent)
        for index, item in enumerate(raw_events)
    )
    randomization = _parse_randomization(obj["randomization"])
    normalization = _parse_normalization(obj["normalization"])

    bus_raw = _require_mapping(obj["bus_by_level"], "bus_by_level")
    if not bus_raw:
        raise SkillSFXError(
            "SFX-E1002",
            "bus_by_level must contain at least one level",
            {"field": "bus_by_level"},
        )
    buses: dict[str, BusParams] = {}
    for level_key, bus_value in bus_raw.items():
        if not isinstance(level_key, str) or not level_key.isdigit():
            raise SkillSFXError(
                "SFX-E1002",
                "bus_by_level keys must be non-negative integer strings",
                {"field": "bus_by_level", "key": level_key},
            )
        normalized_key = str(int(level_key))
        if normalized_key != level_key:
            raise SkillSFXError(
                "SFX-E1003",
                "bus_by_level keys must use canonical integer notation",
                {"field": "bus_by_level", "key": level_key},
            )
        buses[level_key] = _parse_bus(
            bus_value,
            f"bus_by_level.{level_key}",
        )
    for event in events:
        if event.offset_ms >= output.duration_ms:
            raise SkillSFXError(
                "SFX-E2003",
                "event offset is outside the output timeline",
                {
                    "event_index": event.index,
                    "source_path": event.source_ref,
                    "offset_ms": event.offset_ms,
                    "output_duration_ms": output.duration_ms,
                },
            )
        worst_left = event.offset_ms - randomization.time_jitter_ms
        worst_right = event.offset_ms + randomization.time_jitter_ms
        if worst_left < 0.0 or worst_right >= output.duration_ms:
            raise SkillSFXError(
                "SFX-E2003",
                "event offset plus randomization can leave the output timeline",
                {
                    "event_index": event.index,
                    "source_path": event.source_ref,
                    "offset_ms": event.offset_ms,
                    "time_jitter_ms": randomization.time_jitter_ms,
                    "output_duration_ms": output.duration_ms,
                },
            )
    has_tail_role = any(event.role == "tail" for event in events)
    randomization_active = (
        randomization.gain_jitter_db > 0.0
        or randomization.time_jitter_ms > 0.0
        or (
            randomization.tail_harmonics.enabled
            and has_tail_role
        )
    )
    if variant_count > 1 and not randomization_active:
        raise SkillSFXError(
            "SFX-E1003",
            "multiple variants require at least one active randomization source",
            {
                "field": "randomization",
                "variant_count": variant_count,
            },
        )

    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SkillSFXError(
            "SFX-E1002",
            "metadata must be a JSON object",
            {"field": "metadata"},
        )
    return Recipe(
        path=recipe_path,
        sha256=sha256_file(recipe_path),
        schema_version=schema_version,
        skill_name=skill_name,
        variant_count=variant_count,
        seed=seed,
        output=output,
        events=events,
        randomization=randomization,
        bus_by_level=buses,
        normalization=normalization,
        metadata=metadata,
    )
