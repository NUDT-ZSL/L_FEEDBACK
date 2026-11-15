"""Bake configuration validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constants import CHANNELS, SAMPLE_FORMAT, SAMPLE_RATE
from .errors import InputFatal


@dataclass(frozen=True)
class BakeConfig:
    sample_rate: int
    channels: int
    sample_format: str
    stretch_min_ratio: float
    stretch_max_ratio: float
    crossfade_ms: float
    attack_ms: float
    release_ms: float
    target_peak_dbfs: float
    chunk_alignment: int

    @property
    def crossfade_samples(self) -> int:
        return max(1, round(self.crossfade_ms * self.sample_rate / 1000.0))

    @property
    def attack_samples(self) -> int:
        return max(1, round(self.attack_ms * self.sample_rate / 1000.0))

    @property
    def release_samples(self) -> int:
        return max(1, round(self.release_ms * self.sample_rate / 1000.0))

    @property
    def target_peak(self) -> float:
        return 10.0 ** (self.target_peak_dbfs / 20.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_format": self.sample_format,
            "stretch_min_ratio": self.stretch_min_ratio,
            "stretch_max_ratio": self.stretch_max_ratio,
            "crossfade_ms": self.crossfade_ms,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "target_peak_dbfs": self.target_peak_dbfs,
            "chunk_alignment": self.chunk_alignment,
        }


def _number(
    value: Any,
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputFatal("CONFIG_INVALID", f"{field} must be a number", {"field": field})
    result = float(value)
    if not math.isfinite(result):
        raise InputFatal("CONFIG_INVALID", f"{field} must be finite", {"field": field})
    if positive and result <= 0:
        raise InputFatal("CONFIG_INVALID", f"{field} must be > 0", {"field": field})
    if nonnegative and result < 0:
        raise InputFatal("CONFIG_INVALID", f"{field} must be >= 0", {"field": field})
    return result


def parse_config(raw: Any) -> BakeConfig:
    if not isinstance(raw, dict):
        raise InputFatal("CONFIG_INVALID", "config root must be an object")
    if raw.get("schema_version") != 1:
        raise InputFatal(
            "CONFIG_INVALID",
            "unsupported config schema_version",
            {"schema_version": raw.get("schema_version")},
        )

    sample_rate = raw.get("sample_rate")
    channels = raw.get("channels")
    sample_format = raw.get("sample_format")
    if sample_rate != SAMPLE_RATE:
        raise InputFatal(
            "CONFIG_INVALID",
            f"sample_rate must be {SAMPLE_RATE}",
            {"sample_rate": sample_rate},
        )
    if channels != CHANNELS:
        raise InputFatal(
            "CONFIG_INVALID",
            f"channels must be {CHANNELS}",
            {"channels": channels},
        )
    if sample_format != SAMPLE_FORMAT:
        raise InputFatal(
            "CONFIG_INVALID",
            f"sample_format must be {SAMPLE_FORMAT}",
            {"sample_format": sample_format},
        )

    stretch = raw.get("stretch")
    synthesis = raw.get("synthesis")
    mix = raw.get("mix")
    pack = raw.get("pack")
    if not all(isinstance(section, dict) for section in (stretch, synthesis, mix, pack)):
        raise InputFatal(
            "CONFIG_INVALID",
            "stretch, synthesis, mix, and pack must be objects",
        )

    min_ratio = _number(stretch.get("min_ratio"), "stretch.min_ratio", positive=True)
    max_ratio = _number(stretch.get("max_ratio"), "stretch.max_ratio", positive=True)
    if min_ratio > 1.0 or max_ratio < 1.0:
        raise InputFatal(
            "CONFIG_INVALID",
            "stretch range must contain ratio 1.0",
            {"min_ratio": min_ratio, "max_ratio": max_ratio},
        )
    if min_ratio > max_ratio:
        raise InputFatal(
            "CONFIG_INVALID",
            "stretch.min_ratio must be <= stretch.max_ratio",
        )

    crossfade_ms = _number(
        synthesis.get("crossfade_ms"), "synthesis.crossfade_ms", nonnegative=True
    )
    attack_ms = _number(
        synthesis.get("attack_ms"), "synthesis.attack_ms", nonnegative=True
    )
    release_ms = _number(
        synthesis.get("release_ms"), "synthesis.release_ms", nonnegative=True
    )
    target_peak_dbfs = _number(
        mix.get("target_peak_dbfs"), "mix.target_peak_dbfs"
    )
    if target_peak_dbfs >= 0:
        raise InputFatal(
            "CONFIG_INVALID",
            "mix.target_peak_dbfs must be below 0 dBFS",
            {"target_peak_dbfs": target_peak_dbfs},
        )

    alignment = pack.get("chunk_alignment")
    if isinstance(alignment, bool) or not isinstance(alignment, int):
        raise InputFatal(
            "CONFIG_INVALID",
            "pack.chunk_alignment must be an integer",
        )
    if alignment < 1 or alignment > 65536 or alignment & (alignment - 1):
        raise InputFatal(
            "CONFIG_INVALID",
            "pack.chunk_alignment must be a power of two between 1 and 65536",
        )

    return BakeConfig(
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        stretch_min_ratio=min_ratio,
        stretch_max_ratio=max_ratio,
        crossfade_ms=crossfade_ms,
        attack_ms=attack_ms,
        release_ms=release_ms,
        target_peak_dbfs=target_peak_dbfs,
        chunk_alignment=alignment,
    )
