from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int
    channels: int
    bit_depth: int


@dataclass(frozen=True)
class LanguageSpec:
    language_id: str
    path: Path


@dataclass(frozen=True)
class SwitchPoint:
    original_index: int
    frame: int
    language_id: str
    implicit: bool = False


@dataclass
class Manifest:
    manifest_path: Path
    frame_rate: float
    audio: AudioSpec
    languages: list[LanguageSpec]
    switches: list[SwitchPoint]
    video_path: Path
    reference_stream_index: int | None
    duration_frames: int | None
    crossfade_ms: float = 15.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def language_by_id(self) -> dict[str, LanguageSpec]:
        return {item.language_id: item for item in self.languages}


@dataclass(frozen=True)
class WavInfo:
    path: Path
    audio_format: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    block_align: int
    data_offset: int
    data_size: int
    data_frames: int
    block_aligned: bool
    file_size: int
    truncated: bool


@dataclass
class VideoInfo:
    path: Path
    duration_seconds: float | None
    audio_streams: list[dict[str, Any]]
    selected_reference_index: int | None


@dataclass
class MediaInfo:
    wavs: dict[str, WavInfo]
    video: VideoInfo

    def duration_frames(self, frame_rate: float) -> int:
        if not self.wavs:
            return 0
        return min(
            int(info.data_frames * frame_rate / info.sample_rate)
            for info in self.wavs.values()
        )

