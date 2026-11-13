"""Plain data structures shared by the planner, renderer and validator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class SegmentInput:
    """One event segment as described by the timeline + segment metadata."""

    index: int
    event_type: str
    voice_wav: str
    ambience_wav: str
    duration_ms: float
    dialogue_start_ms: float
    dialogue_end_ms: float
    target_peak_dbfs: float
    fade_ms: float
    gap_after_ms: float = 0.0


@dataclass
class CrossfadeRegion:
    """Boundary descriptor between two adjacent planned segments."""

    boundary_index: int
    left_segment: int
    right_segment: int
    start_frame: int
    end_frame: int
    fade_frames: int
    gap_frames: int
    start_ms: float
    end_ms: float


@dataclass
class PlannedSegment:
    index: int
    event_type: str
    voice_wav: str
    ambience_wav: str
    start_frame: int
    end_frame: int  # exclusive
    fade_in_frames: int
    fade_out_frames: int
    duck_start_frame: int
    duck_end_frame: int
    dialogue_start_ms: float
    dialogue_end_ms: float
    target_peak_dbfs: float
    fade_ms: float
    start_ms: float
    end_ms: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class TimelinePlan:
    sample_rate: int
    block_frames: int
    total_frames: int
    total_ms: float
    loudness_target_dbfs: float
    duck_gain_db: float
    duck_attack_ms: float
    duck_release_ms: float
    gain_smoothing_ms: float
    limiter_ceiling_dbfs: float
    segments: List[PlannedSegment]
    crossfades: List[CrossfadeRegion]
    inputs: List[SegmentInput]

    def block_count(self) -> int:
        if self.total_frames == 0:
            return 0
        return (self.total_frames + self.block_frames - 1) // self.block_frames

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockDescriptor:
    index: int
    start_frame: int
    frame_count: int  # last block may be shorter
    start_ms: float
    end_ms: float
    input_hash: str
    output_hash: Optional[str] = None


@dataclass
class Snapshot:
    version: str
    created_at: str
    source_timeline_path: str
    plan: dict
    blocks: List[dict] = field(default_factory=list)
    deprecated_blocks: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentGainReport:
    segment_index: int
    measured_peak_dbfs: float
    measured_rms_dbfs: float
    target_peak_dbfs: float
    loudness_gain_db: float
    peak_gain_db: float
    applied_gain_db: float
    smoothed: bool
    limiter_active: bool
    limiter_ceiling_dbfs: float

    def to_dict(self) -> dict:
        return asdict(self)
