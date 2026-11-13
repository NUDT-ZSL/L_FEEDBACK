from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ExportConfig:
    sample_rate: int = 48_000
    chunk_frames: int = 4_800
    event_categories: tuple[str, ...] = (
        "kill",
        "teamfight",
        "crate",
        "击杀",
        "团战",
        "开箱",
    )
    clip_threshold: int = 32_767
    clip_min_run: int = 2
    silence_threshold_dbfs: float = -45.0
    silence_window_ms: float = 20.0
    silence_hop_ms: float = 10.0
    silence_min_ms: float = 40.0
    silence_merge_ms: float = 60.0
    silence_fade_ms: float = 8.0
    target_lufs: float = -16.0
    game_target_lufs: float | None = None
    mic_target_lufs: float | None = None
    game_mix_db: float = -3.0
    mic_mix_db: float = 0.0
    max_boost_db: float = 24.0
    max_cut_db: float = 24.0
    peak_limit_dbfs: float = -1.0
    sinc_taps: int = 32
    sinc_phases: int = 1_024
    varispeed_min_rate: float = 0.25
    varispeed_max_rate: float = 4.0

    def __post_init__(self) -> None:
        if self.sample_rate != 48_000:
            raise ValueError("only the required 48 kHz sample rate is supported")
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if self.clip_min_run < 1:
            raise ValueError("clip_min_run must be >= 1")
        if not (0 < self.clip_threshold <= 32_768):
            raise ValueError("clip_threshold must be in 1..32768")
        if self.silence_window_ms <= 0 or self.silence_hop_ms <= 0:
            raise ValueError("silence window and hop must be positive")
        if self.silence_fade_ms < 0:
            raise ValueError("silence_fade_ms must be non-negative")
        if self.varispeed_min_rate <= 0:
            raise ValueError("varispeed_min_rate must be positive")
        if self.varispeed_max_rate < self.varispeed_min_rate:
            raise ValueError("invalid varispeed range")
        if self.sinc_taps < 8 or self.sinc_taps % 2:
            raise ValueError("sinc_taps must be an even integer >= 8")
        if self.sinc_phases < 64:
            raise ValueError("sinc_phases must be >= 64")
        if not (-100.0 < self.peak_limit_dbfs <= 0.0):
            raise ValueError("peak_limit_dbfs must be in (-100, 0]")
        numeric_fields = (
            self.silence_threshold_dbfs,
            self.silence_window_ms,
            self.silence_hop_ms,
            self.silence_min_ms,
            self.silence_merge_ms,
            self.silence_fade_ms,
            self.target_lufs,
            self.game_mix_db,
            self.mic_mix_db,
            self.max_boost_db,
            self.max_cut_db,
            self.peak_limit_dbfs,
            self.varispeed_min_rate,
            self.varispeed_max_rate,
        )
        if not all(math.isfinite(value) for value in numeric_fields):
            raise ValueError("all numeric configuration values must be finite")
        if self.game_target_lufs is not None and not math.isfinite(
            self.game_target_lufs
        ):
            raise ValueError("game_target_lufs must be finite")
        if self.mic_target_lufs is not None and not math.isfinite(
            self.mic_target_lufs
        ):
            raise ValueError("mic_target_lufs must be finite")

    def effective_game_target(self) -> float:
        return self.target_lufs if self.game_target_lufs is None else self.game_target_lufs

    def effective_mic_target(self) -> float:
        return self.target_lufs if self.mic_target_lufs is None else self.mic_target_lufs

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_categories"] = list(self.event_categories)
        return value

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExportConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        values = dict(values)
        if "event_categories" in values:
            values["event_categories"] = tuple(values["event_categories"])
        return cls(**values)

    def overlaid(self, values: dict[str, Any] | None) -> "ExportConfig":
        if not values:
            return self
        merged = self.to_dict()
        merged.update(values)
        return ExportConfig.from_dict(merged)

