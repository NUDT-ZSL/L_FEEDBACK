from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ConfigError(ValueError):
    """Raised when the JSON configuration is missing or invalid."""


@dataclass(frozen=True)
class SoundOverride:
    variants: int
    seed: Optional[int]


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    global_seed: int
    target_lufs: float
    output_dir: Path
    default_variants: int
    overrides: Tuple[Tuple[str, str, SoundOverride], ...]

    def override_for(self, filename: str, stem: str) -> Optional[SoundOverride]:
        matches: List[SoundOverride] = []
        for kind, target, override in self.overrides:
            if kind == "filename" and filename == target:
                matches.append(override)
            elif kind == "stem" and stem == target:
                matches.append(override)
        if len(matches) > 1:
            raise ConfigError(f"multiple sound overrides match input file {filename!r}")
        return matches[0] if matches else None

    def unmatched_overrides(self, filenames: List[str], stems: List[str]) -> List[str]:
        names = set(filenames)
        stem_set = set(stems)
        missing: List[str] = []
        for kind, target, _ in self.overrides:
            if kind == "filename" and target not in names:
                missing.append(target)
            elif kind == "stem" and target not in stem_set:
                missing.append(target)
        return missing

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self.config_path),
            "global_seed": self.global_seed,
            "target_lufs": self.target_lufs,
            "output_dir": str(self.output_dir),
            "default_variants": self.default_variants,
            "sound_overrides": {
                target: {"variants": override.variants, "seed": override.seed}
                for _, target, override in self.overrides
            },
        }


_TOP_LEVEL_KEYS = {
    "global_seed",
    "seed",
    "target_lufs",
    "output_dir",
    "default_variants",
    "variants",
    "sounds",
    "effects",
    "files",
    "overrides",
    "sound_overrides",
}

_SOUND_KEYS = {"variants", "variant_count", "count", "seed"}
_SOUND_LIST_KEYS = {"file", "path", "name", "source"} | _SOUND_KEYS


def _reject_constant(value: str) -> Any:
    raise ConfigError(f"invalid non-finite JSON number: {value}")


def _object_without_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate configuration key: {key!r}")
        result[key] = value
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, field: str, minimum: int, maximum: int = 2**64 - 1) -> int:
    if not _is_int(value):
        raise ConfigError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ConfigError(f"{field} must be between {minimum} and {maximum}")
    return value


def _require_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite")
    if result < minimum or result > maximum:
        raise ConfigError(f"{field} must be between {minimum} and {maximum}")
    return result


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _select_alias(
    data: Dict[str, Any], canonical: str, aliases: Tuple[str, ...]
) -> Optional[Any]:
    present = [key for key in (canonical,) + aliases if key in data]
    if len(present) > 1:
        raise ConfigError(f"configuration fields are mutually exclusive: {present}")
    return data[present[0]] if present else None


def _parse_override(value: Any, field: str, default_variants: int) -> SoundOverride:
    if _is_int(value):
        return SoundOverride(
            variants=_require_int(value, f"{field}.variants", 1, 1000), seed=None
        )
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object or an integer")
    unknown = sorted(set(value) - _SOUND_KEYS)
    if unknown:
        raise ConfigError(f"{field} contains unknown fields: {unknown}")
    variants_value = _select_alias(value, "variants", ("variant_count", "count"))
    variants = default_variants
    if variants_value is not None:
        variants = _require_int(variants_value, f"{field}.variants", 1, 1000)
    seed = None
    if "seed" in value:
        seed = _require_int(value["seed"], f"{field}.seed", 0)
    return SoundOverride(variants=variants, seed=seed)


def _override_target(key: str, field: str) -> Tuple[str, str]:
    cleaned = key.strip()
    if not cleaned:
        raise ConfigError(f"{field} contains an empty sound name")
    name = Path(cleaned).name
    suffix = Path(name).suffix.lower()
    if suffix == ".wav":
        return "filename", name
    if suffix:
        raise ConfigError(f"{field} key {key!r} must be a .wav filename or a file stem")
    return "stem", name


def _parse_sound_overrides(
    data: Dict[str, Any], default_variants: int
) -> Tuple[Tuple[str, str, SoundOverride], ...]:
    value = _select_alias(
        data,
        "sounds",
        ("effects", "files", "overrides", "sound_overrides"),
    )
    if value is None and isinstance(data.get("variants"), (dict, list)):
        value = data["variants"]
    if value is None:
        return ()
    overrides: List[Tuple[str, str, SoundOverride]] = []
    seen: Dict[Tuple[str, str], str] = {}
    target_names: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError("sounds keys must be strings")
            kind, target = _override_target(key, "sounds")
            if (kind, target) in seen:
                raise ConfigError(f"sounds contains duplicate override for {target!r}")
            if target in target_names:
                raise ConfigError(f"sounds contains ambiguous overrides for {target!r}")
            seen[(kind, target)] = key
            target_names[target] = key
            overrides.append(
                (
                    kind,
                    target,
                    _parse_override(item, f"sounds.{key}", default_variants),
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            field = f"sounds[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{field} must be an object")
            unknown = sorted(set(item) - _SOUND_LIST_KEYS)
            if unknown:
                raise ConfigError(f"{field} contains unknown fields: {unknown}")
            target_value = _select_alias(item, "file", ("path", "name", "source"))
            if target_value is None:
                raise ConfigError(f"{field}.file is required")
            target_text = _require_string(target_value, f"{field}.file")
            kind, target = _override_target(target_text, field)
            if (kind, target) in seen:
                raise ConfigError(f"sounds contains duplicate override for {target!r}")
            if target in target_names:
                raise ConfigError(f"sounds contains ambiguous overrides for {target!r}")
            seen[(kind, target)] = target_text
            target_names[target] = target_text
            override_data = {key: item[key] for key in _SOUND_KEYS if key in item}
            overrides.append(
                (
                    kind,
                    target,
                    _parse_override(override_data, field, default_variants),
                )
            )
    else:
        raise ConfigError("sounds must be an object or an array")
    return tuple(overrides)


def load_config(path_value: str) -> AppConfig:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise ConfigError(f"configuration file does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"configuration path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ConfigError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a JSON object")
    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"configuration contains unknown fields: {unknown}")

    seed_value = _select_alias(data, "global_seed", ("seed",))
    if seed_value is None:
        raise ConfigError("global_seed is required")
    global_seed = _require_int(seed_value, "global_seed", 0)

    target_lufs = -23.0
    if "target_lufs" in data:
        target_lufs = _require_number(data["target_lufs"], "target_lufs", -70.0, 0.0)

    if "output_dir" not in data:
        raise ConfigError("output_dir is required")
    output_text = _require_string(data["output_dir"], "output_dir")
    output_dir = Path(output_text).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.resolve().parent / output_dir
    output_dir = output_dir.resolve()

    default_variants = 3
    if "default_variants" in data:
        default_variants = _require_int(
            data["default_variants"], "default_variants", 1, 1000
        )
    elif "variants" in data and _is_int(data["variants"]):
        default_variants = _require_int(data["variants"], "variants", 1, 1000)
    elif "variants" in data and not isinstance(data["variants"], (dict, list)):
        raise ConfigError("variants must be an integer, an object, or an array")

    if "sounds" in data and isinstance(data.get("variants"), (dict, list)):
        raise ConfigError(
            "sounds and a variants mapping cannot both define sound overrides"
        )
    if "default_variants" in data and _is_int(data.get("variants")):
        raise ConfigError(
            "default_variants and an integer variants field are mutually exclusive"
        )
    overrides = _parse_sound_overrides(data, default_variants)
    return AppConfig(
        config_path=path.resolve(),
        global_seed=global_seed,
        target_lufs=target_lufs,
        output_dir=output_dir,
        default_variants=default_variants,
        overrides=overrides,
    )
