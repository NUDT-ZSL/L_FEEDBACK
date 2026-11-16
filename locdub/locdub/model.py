"""Typed data structures shared by the pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Segment:
    """A single dialogue line on the timeline.

    Times are stored as integer milliseconds.
    """

    segment_id: str
    start_ms: int
    end_ms: int
    audio_path: str
    min_ratio: float
    max_ratio: float
    group: Optional[str] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "group": self.group,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "target_duration_ms": self.duration_ms,
            "audio_path": self.audio_path,
            "min_ratio": round(self.min_ratio, 6),
            "max_ratio": round(self.max_ratio, 6),
        }


@dataclass
class SegmentResult:
    """Outcome of adapting one segment for one language."""

    segment_id: str
    status: str
    original_duration_ms: Optional[float] = None
    adapted_duration_ms: Optional[float] = None
    requested_ratio: Optional[float] = None
    applied_ratio: Optional[float] = None
    boundary_clamped: bool = False
    silence_placeholder: bool = False
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "status": self.status,
            "original_duration_ms": _round(self.original_duration_ms),
            "adapted_duration_ms": _round(self.adapted_duration_ms),
            "requested_ratio": _round(self.requested_ratio),
            "applied_ratio": _round(self.applied_ratio),
            "boundary_clamped": self.boundary_clamped,
            "silence_placeholder": self.silence_placeholder,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class LanguageResult:
    """Aggregate result for one language."""

    language: str
    status: str
    output_file: Optional[str]
    segments: List[SegmentResult]
    total_segments: int
    ok_segments: int
    skipped_segments: int

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "overall_status": self.status,
            "output_file": self.output_file,
            "total_segments": self.total_segments,
            "ok_segments": self.ok_segments,
            "skipped_segments": self.skipped_segments,
            "segments": [item.to_report_dict() for item in self.segments],
        }


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)
