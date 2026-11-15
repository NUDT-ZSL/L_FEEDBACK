from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrackConfig:
    id: str
    name: str
    initial_mute: bool = False
    initial_solo: bool = False
    gain_db: float = 0.0


@dataclass
class TrackState:
    id: str
    name: str
    initial_mute: bool
    initial_solo: bool
    effective_mute: bool
    effective_solo: bool
    audible: bool
    mute_source: str
    solo_source: str
    gain_db: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "gain_db": self.gain_db,
            "initial_mute": self.initial_mute,
            "initial_solo": self.initial_solo,
            "mute": self.effective_mute,
            "solo": self.effective_solo,
            "audible": self.audible,
            "mute_source": self.mute_source,
            "solo_source": self.solo_source,
        }


@dataclass(frozen=True)
class ClipSpec:
    id: str
    path: Path
    offset_ms: float = 0.0
    duration_ms: Optional[float] = None


@dataclass
class EventSpec:
    id: str
    event_type: str
    time_ms: float
    track_id: str
    asset_id: str
    gain_db: float = 0.0
    offset_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    source_index: int = 0


@dataclass
class ParsedTimeline:
    track_configs: List[TrackConfig]
    clip_specs: Dict[str, ClipSpec]
    events: List[EventSpec]
    timeline_duration_ms: Optional[float]
    title: str
    parse_errors: List[Dict[str, Any]] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)
    asset_manifest: Optional[str] = None
    timeline_path: Optional[Path] = None


@dataclass
class CachedAsset:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    audio: Any
    resampling: str
    ratio_up: int
    ratio_down: int
    referenced_by: List[str] = field(default_factory=list)
    load_count: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "original_sample_rate": self.sample_rate,
            "channels": self.channels,
            "source_frames": self.frames,
            "duration_s": round(self.frames / self.sample_rate, 6),
            "output_sample_rate": 48000,
            "resampling": self.resampling,
            "resampling_ratio": [self.ratio_up, self.ratio_down],
            "referenced_by": self.referenced_by,
            "load_count": self.load_count,
        }


@dataclass
class RenderedEvent:
    event: EventSpec
    start_frame: int
    end_frame: int
    audio: Any
    start_s: float
    end_s: float
    audible: bool


@dataclass
class AssetFailure:
    path: Path
    error: str
