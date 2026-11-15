from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigError


@dataclass(frozen=True)
class VariantConfig:
    index: int
    bitrate_kbps: int
    input_path: Path
    offset_ms: Decimal

    @property
    def media_dir(self) -> str:
        return f"{self.bitrate_kbps}k"


@dataclass(frozen=True)
class BuildConfig:
    config_path: Path
    output_dir: Path
    segment_duration: Decimal
    overwrite: bool
    variants: Tuple[VariantConfig, ...]


def _find_one(
    obj: Dict[str, Any],
    names: Tuple[str, ...],
    context: str,
    required: bool = True,
) -> Tuple[Optional[str], Any]:
    found = [name for name in names if name in obj]
    if len(found) > 1:
        raise ConfigError(
            f"{context} contains conflicting fields: {', '.join(found)}"
        )
    if not found:
        if required:
            raise ConfigError(
                f"{context} is missing required field '{names[0]}'"
            )
        return None, None
    return found[0], obj[found[0]]


def _number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{field} must be finite")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigError(f"{field} is not a valid number") from exc


def _resolve_path(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(config_path: Path) -> BuildConfig:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")

    base = config_path.parent
    _, output_value = _find_one(
        raw, ("output_dir", "output_directory"), "config"
    )
    _, duration_value = _find_one(
        raw, ("segment_duration", "segment_duration_seconds"), "config"
    )
    _, variants_value = _find_one(raw, ("variants", "bitrates"), "config")

    output_dir = _resolve_path(base, output_value, "output_dir")
    segment_duration = _number(duration_value, "segment_duration")
    if segment_duration < Decimal("1") or segment_duration > Decimal("30"):
        raise ConfigError("segment_duration must be between 1 and 30 seconds")

    overwrite = raw.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ConfigError("overwrite must be a boolean")

    if not isinstance(variants_value, list) or not variants_value:
        raise ConfigError("variants must be a non-empty array")

    variants: List[VariantConfig] = []
    seen_bitrates = set()
    for index, item in enumerate(variants_value):
        context = f"variants[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{context} must be an object")

        _, bitrate_value = _find_one(
            item, ("bitrate_kbps", "bitrate"), context
        )
        _, input_value = _find_one(item, ("input", "input_wav"), context)
        _, offset_value = _find_one(
            item, ("offset_ms",), context, required=False
        )

        if isinstance(bitrate_value, bool) or not isinstance(bitrate_value, int):
            raise ConfigError(f"{context}.bitrate_kbps must be an integer")
        if bitrate_value <= 0:
            raise ConfigError(f"{context}.bitrate_kbps must be greater than zero")
        if bitrate_value in seen_bitrates:
            raise ConfigError(f"duplicate bitrate: {bitrate_value} kbps")
        seen_bitrates.add(bitrate_value)

        input_path = _resolve_path(base, input_value, f"{context}.input")
        if not input_path.is_file():
            raise ConfigError(f"input WAV does not exist: {input_path}")

        offset_ms = (
            Decimal("0")
            if offset_value is None
            else _number(offset_value, f"{context}.offset_ms")
        )

        variants.append(
            VariantConfig(
                index=index,
                bitrate_kbps=bitrate_value,
                input_path=input_path,
                offset_ms=offset_ms,
            )
        )

    for previous, current in zip(variants, variants[1:]):
        if current.bitrate_kbps <= previous.bitrate_kbps:
            raise ConfigError(
                "variants must be in strictly ascending bitrate order: "
                f"{previous.bitrate_kbps} kbps appears before "
                f"{current.bitrate_kbps} kbps"
            )

    return BuildConfig(
        config_path=config_path,
        output_dir=output_dir,
        segment_duration=segment_duration,
        overwrite=overwrite,
        variants=tuple(variants),
    )

