"""YAML configuration parsing and validation.

Invalid configurations raise :class:`ConfigError`; the CLI renders every such
error as a JSON object containing ``"error_code": "ERR_CFG"`` on stderr and
exits with status code 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MIN_SIZE = 256
MAX_SIZE = 2048
MIN_ITERATIONS = 1
MAX_ITERATIONS = 50


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or fails validation."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ErosionConfig:
    hydraulic_weight: float = 0.5
    thermal_weight: float = 0.3
    pipe_factor: float = 0.1
    erodibility: float = 0.3
    deposition: float = 0.3
    evaporation: float = 0.02
    rain: float = 0.1
    talus: float = 0.05
    thermal_carry: float = 0.5


@dataclass(frozen=True)
class NoiseConfig:
    seed: int = 1337
    base_frequency: int = 8
    octaves: int = 5
    lacunarity: float = 2.0
    persistence: float = 0.5


@dataclass(frozen=True)
class TerrainConfig:
    width: int
    height: int
    iterations: int
    output_prefix: str
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)
    sweep_iterations: tuple[int, ...] = ()
    fault_at: int | None = None


def _require_type(key: str, value: Any, expected: tuple[type, ...], typename: str) -> Any:
    # ``bool`` is a subclass of ``int``; treat booleans as invalid for numbers.
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ConfigError(f"{key} must be of type {typename}, got {type(value).__name__}")
    return value


def _as_int(key: str, value: Any) -> int:
    return int(_require_type(key, value, (int,), "integer"))


def _as_float(key: str, value: Any) -> float:
    return float(_require_type(key, value, (int, float), "number"))


def _as_range_int(key: str, value: Any, low: int, high: int) -> int:
    number = _as_int(key, value)
    if not low <= number <= high:
        raise ConfigError(f"{key} must be within [{low}, {high}], got {number}")
    return number


def _as_unit_float(key: str, value: Any) -> float:
    number = _as_float(key, value)
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"{key} must be within [0, 1], got {number}")
    return number


def _as_nonneg_float(key: str, value: Any) -> float:
    number = _as_float(key, value)
    if number < 0.0:
        raise ConfigError(f"{key} must be non-negative, got {number}")
    return number


def _parse_noise(raw: Any) -> NoiseConfig:
    if raw is None:
        return NoiseConfig()
    _require_type("noise", raw, (dict,), "mapping")
    seed = _as_range_int("noise.seed", raw.get("seed", 1337), 0, 2**32 - 1)
    base_frequency = _as_range_int("noise.base_frequency", raw.get("base_frequency", 8), 1, 128)
    octaves = _as_range_int("noise.octaves", raw.get("octaves", 5), 1, 12)
    lacunarity = _as_float("noise.lacunarity", raw.get("lacunarity", 2.0))
    if lacunarity <= 1.0:
        raise ConfigError("noise.lacunarity must be greater than 1.0")
    persistence = _as_unit_float("noise.persistence", raw.get("persistence", 0.5))
    return NoiseConfig(seed, base_frequency, octaves, lacunarity, persistence)


def _parse_erosion(raw: Any) -> ErosionConfig:
    if raw is None:
        return ErosionConfig()
    _require_type("erosion", raw, (dict,), "mapping")
    get = raw.get
    return ErosionConfig(
        hydraulic_weight=_as_unit_float("erosion.hydraulic_weight", get("hydraulic_weight", 0.5)),
        thermal_weight=_as_unit_float("erosion.thermal_weight", get("thermal_weight", 0.3)),
        pipe_factor=_as_nonneg_float("erosion.pipe_factor", get("pipe_factor", 0.1)),
        erodibility=_as_unit_float("erosion.erodibility", get("erodibility", 0.3)),
        deposition=_as_unit_float("erosion.deposition", get("deposition", 0.3)),
        evaporation=_as_unit_float("erosion.evaporation", get("evaporation", 0.02)),
        rain=_as_nonneg_float("erosion.rain", get("rain", 0.1)),
        talus=_as_nonneg_float("erosion.talus", get("talus", 0.05)),
        thermal_carry=_as_unit_float("erosion.thermal_carry", get("thermal_carry", 0.5)),
    )


def _parse_sweep(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    _require_type("sweep_iterations", raw, (list,), "list")
    if not raw:
        raise ConfigError("sweep_iterations must not be empty when provided")
    values = [
        _as_range_int(f"sweep_iterations[{idx}]", item, MIN_ITERATIONS, MAX_ITERATIONS)
        for idx, item in enumerate(raw)
    ]
    return tuple(values)


def parse_config(path: str | Path) -> TerrainConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {config_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML syntax: {exc}") from exc
    if raw is None:
        raise ConfigError("configuration file is empty")
    _require_type("root", raw, (dict,), "mapping")

    required = ("width", "height", "iterations")
    for key in required:
        if key not in raw:
            raise ConfigError(f"missing required key: {key}")
    width = _as_range_int("width", raw["width"], MIN_SIZE, MAX_SIZE)
    height = _as_range_int("height", raw["height"], MIN_SIZE, MAX_SIZE)
    iterations = _as_range_int("iterations", raw["iterations"], MIN_ITERATIONS, MAX_ITERATIONS)
    prefix = raw.get("output_prefix", "terrain")
    _require_type("output_prefix", prefix, (str,), "string")
    if not prefix:
        raise ConfigError("output_prefix must not be empty")

    fault_raw = raw.get("fault_at")
    fault_at = None
    if fault_raw is not None:
        fault_at = _as_range_int("fault_at", fault_raw, 1, MAX_ITERATIONS)

    return TerrainConfig(
        width=width,
        height=height,
        iterations=iterations,
        output_prefix=prefix,
        noise=_parse_noise(raw.get("noise")),
        erosion=_parse_erosion(raw.get("erosion")),
        sweep_iterations=_parse_sweep(raw.get("sweep_iterations")),
        fault_at=fault_at,
    )
