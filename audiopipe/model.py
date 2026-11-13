"""Shared data structures and run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Manifest:
    dialogue_path: str
    music_path: str
    ambience_path: str
    target_lufs: float
    music_fade_start_sec: float
    music_fade_duration_sec: float
    music_gain_db: float
    ambience_gain_db: float
    max_attenuation_db: float
    max_boost_db: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dialogue_path": self.dialogue_path,
            "music_path": self.music_path,
            "ambience_path": self.ambience_path,
            "target_lufs": self.target_lufs,
            "music_fade_start_sec": self.music_fade_start_sec,
            "music_fade_duration_sec": self.music_fade_duration_sec,
            "music_gain_db": self.music_gain_db,
            "ambience_gain_db": self.ambience_gain_db,
            "max_attenuation_db": self.max_attenuation_db,
            "max_boost_db": self.max_boost_db,
        }


@dataclass(frozen=True)
class Job:
    job_id: str
    job_dir: Path
    manifest_path: Path
    manifest_rel: str


@dataclass(frozen=True)
class TrackInfo:
    role: str
    path: Path
    sample_rate: int
    channels: int
    duration_sec: Optional[float]
    codec_name: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_sec": self.duration_sec,
            "codec_name": self.codec_name,
        }


@dataclass
class JobFailure(Exception):
    code: str
    message: str
    fatal: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            value["details"] = self.details
        return value


@dataclass
class PipelineConfig:
    root: Path
    ffmpeg: Path
    ffprobe: Path
    jobs: int = 2
    work_dir: Optional[Path] = None
    job_temp_limit_mb: int = 256
    min_free_mb: int = 32
    journal_path: Optional[Path] = None
    result_path: Optional[Path] = None
    ffmpeg_timeout_sec: int = 600


@dataclass
class JobOutcome:
    job: Job
    status: str
    fingerprint: str = ""
    output_rel: Optional[str] = None
    output_size: Optional[int] = None
    measured_dialogue_lufs: Optional[float] = None
    requested_gain_db: Optional[float] = None
    applied_gain_db: Optional[float] = None
    clamped: bool = False
    clamp: str = "none"
    error: Optional[dict[str, Any]] = None
    fatal: bool = False
    track_info: list[dict[str, Any]] = field(default_factory=list)
    manifest_values: Optional[dict[str, Any]] = None
    output_validated: bool = False
    from_cache: bool = False

    @property
    def job_exit_code(self) -> int:
        if self.status in {"completed", "skipped"}:
            return 0
        if self.status == "interrupted":
            return 130
        if self.fatal:
            return 2
        return 1

    def as_result(self) -> dict[str, Any]:
        return {
            "id": self.job.job_id,
            "manifest": self.job.manifest_rel,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "job_exit_code": self.job_exit_code,
            "output": self.output_rel,
            "output_size": self.output_size,
            "output_validated": self.output_validated,
            "from_cache": self.from_cache,
            "measured_dialogue_lufs": self.measured_dialogue_lufs,
            "requested_gain_db": self.requested_gain_db,
            "applied_gain_db": self.applied_gain_db,
            "clamped": self.clamped,
            "clamp": self.clamp,
            "tracks": self.track_info,
            "manifest_values": self.manifest_values,
            "error": self.error,
        }

    def journal_entry(self, updated_at: str) -> dict[str, Any]:
        return {
            "id": self.job.job_id,
            "manifest": self.job.manifest_rel,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "output": self.output_rel,
            "output_size": self.output_size,
            "measured_dialogue_lufs": self.measured_dialogue_lufs,
            "requested_gain_db": self.requested_gain_db,
            "applied_gain_db": self.applied_gain_db,
            "clamped": self.clamped,
            "clamp": self.clamp,
            "error": self.error,
            "updated_at": updated_at,
        }
