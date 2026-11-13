"""Configuration for the packaging pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from typing import Any

from .hashing import canonical_json_hash

#: Snapshot layout version.  Bumping this forces every language block to be
#: recomputed on resume instead of reusing stale offsets.
SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class Config:
    sample_rate: int = 48_000
    max_workers: int = 4
    # Target whole-track loudness in dBFS (RMS based, equal-power).
    target_loudness_db: float = -23.0
    gain_min_db: float = -20.0
    gain_max_db: float = 12.0
    rms_window_ms: float = 100.0
    crossfade_ms: float = 5.0
    # Windows at or below this RMS level count as silence and are excluded
    # from loudness averaging and out-of-target segment reporting.
    silence_floor_db: float = -80.0
    # Tolerance for marking post-normalization segments as out of target.
    loudness_tolerance_db: float = 1.5
    default_max_delay_ms: int = 500
    max_retries: int = 2
    allowed_langs: tuple[str, ...] = ("zh", "en", "ja")
    min_time_ms: int = 0
    max_time_ms: int = 4 * 60 * 60 * 1000
    snapshot_version: int = SNAPSHOT_VERSION

    @property
    def ms_frame(self) -> int:
        """Samples per millisecond (exact for 48kHz)."""
        return self.sample_rate // 1000

    @property
    def rms_window_frames(self) -> int:
        return max(1, int(round(self.sample_rate * self.rms_window_ms / 1000.0)))

    @property
    def crossfade_frames(self) -> int:
        return max(0, int(round(self.sample_rate * self.crossfade_ms / 1000.0)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_langs"] = list(self.allowed_langs)
        return data

    def content_hash(self) -> str:
        """Hash every behaviour-defining setting (used by snapshots)."""
        return canonical_json_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        if "allowed_langs" in filtered:
            filtered["allowed_langs"] = tuple(filtered["allowed_langs"])
        return cls(**filtered)
