"""Configuration model and tolerant config-file loading.

A missing config file or any invalid / out-of-range field falls back to the
default value, and every fallback is recorded in ``notes`` so the processing
report can explain what happened.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    # WSOLA time-stretch factor bounds (input duration / output duration)
    "stretch_min": 0.75,
    "stretch_max": 1.25,
    # maximum energy-search offset allowed for lip events, milliseconds
    "align_tolerance_ms": 50.0,
    # weight of the event-overlap term in the alignment cost function
    "overlap_penalty_weight": 1.0,
    # silence detection
    "silence_threshold_db": -45.0,
    "silence_min_ms": 500.0,
    # minimum fraction of the timeline that lip events must cover
    "coverage_min_ratio": 0.5,
    # test hook: force every MP4 mux attempt to fail so fallback can be tested
    "simulate_mux_failure": False,
}


@dataclass
class Config:
    stretch_min: float = 0.75
    stretch_max: float = 1.25
    align_tolerance_ms: float = 50.0
    overlap_penalty_weight: float = 1.0
    silence_threshold_db: float = -45.0
    silence_min_ms: float = 500.0
    coverage_min_ratio: float = 0.5
    simulate_mux_failure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_field(name: str, value: Any) -> Tuple[bool, Any]:
    """Return (ok, normalized_value) for a single config field."""
    if name == "simulate_mux_failure":
        return isinstance(value, bool), bool(value)
    if not _is_number(value):
        return False, DEFAULT_CONFIG[name]
    value = float(value)
    if name in ("stretch_min", "stretch_max") and not (0.1 <= value <= 4.0):
        return False, DEFAULT_CONFIG[name]
    if name == "align_tolerance_ms" and not (0.0 < value <= 500.0):
        return False, DEFAULT_CONFIG[name]
    if name == "overlap_penalty_weight" and value < 0.0:
        return False, DEFAULT_CONFIG[name]
    if name == "silence_threshold_db" and not (-120.0 <= value <= 0.0):
        return False, DEFAULT_CONFIG[name]
    if name == "silence_min_ms" and not (0.0 < value <= 60_000.0):
        return False, DEFAULT_CONFIG[name]
    if name == "coverage_min_ratio" and not (0.0 <= value <= 1.0):
        return False, DEFAULT_CONFIG[name]
    return True, value


def load_config(path: str | Path | None) -> Tuple[Config, List[str]]:
    """Load a JSON config file, falling back to defaults on any problem."""
    notes: List[str] = []
    raw: Dict[str, Any] = {}

    if path is None:
        notes.append("no config file supplied; using built-in defaults")
        return Config(), notes

    config_path = Path(path)
    if not config_path.is_file():
        notes.append(f"config file '{config_path}' not found; using built-in defaults")
        return Config(), notes

    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("config root is not a JSON object")
        raw = loaded
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        notes.append(f"config file '{config_path}' is unreadable/invalid ({exc}); using defaults")
        return Config(), notes

    values: Dict[str, Any] = {}
    for name, default in DEFAULT_CONFIG.items():
        if name not in raw:
            notes.append(f"config field '{name}' missing; using default {default!r}")
            values[name] = default
            continue
        ok, normalized = _validate_field(name, raw[name])
        if ok:
            values[name] = normalized
        else:
            notes.append(
                f"config field '{name}' has invalid value {raw[name]!r}; "
                f"using default {default!r}"
            )
            values[name] = default

    if values["stretch_min"] > values["stretch_max"]:
        notes.append(
            "stretch_min is greater than stretch_max; resetting both to defaults "
            f"({DEFAULT_CONFIG['stretch_min']}, {DEFAULT_CONFIG['stretch_max']})"
        )
        values["stretch_min"] = DEFAULT_CONFIG["stretch_min"]
        values["stretch_max"] = DEFAULT_CONFIG["stretch_max"]

    return Config(**values), notes
