"""Timeline segmentation for active, silence, gap, and roll regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .timeline import ms_to_frames


Interval = tuple[float, float]


@dataclass(frozen=True)
class SegmentSpec:
    kind: str
    start_ms: float
    end_ms: float
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def clip_intervals(
    intervals: Iterable[Interval], lower: float, upper: float
) -> list[Interval]:
    clipped: list[Interval] = []
    for start, end in intervals:
        start = max(start, lower)
        end = min(end, upper)
        if end > start:
            clipped.append((start, end))
    return merge_intervals(clipped)


def _inside(intervals: list[Interval], point: float) -> bool:
    return any(start <= point < end for start, end in intervals)


def _adjacent(left: SegmentSpec, right: SegmentSpec) -> bool:
    return left.kind == right.kind and left.end_frame == right.start_frame


def _append_merged(segments: list[SegmentSpec], item: SegmentSpec) -> None:
    if item.frames <= 0:
        return
    if segments and _adjacent(segments[-1], item):
        previous = segments[-1]
        segments[-1] = SegmentSpec(
            kind=previous.kind,
            start_ms=previous.start_ms,
            end_ms=item.end_ms,
            start_frame=previous.start_frame,
            end_frame=item.end_frame,
        )
    else:
        segments.append(item)


def _segment(kind: str, start_ms: float, end_ms: float) -> SegmentSpec:
    return SegmentSpec(
        kind=kind,
        start_ms=float(start_ms),
        end_ms=float(end_ms),
        start_frame=ms_to_frames(start_ms),
        end_frame=ms_to_frames(end_ms),
    )


def build_segments(
    *,
    session_ms: float,
    write_start_ms: float,
    write_end_ms: float,
    silence_ms: Iterable[Interval],
    gaps_ms: Iterable[Interval],
) -> list[SegmentSpec]:
    """Partition one track's session span into explicit segment types.

    Silence and gap intervals are session-absolute millisecond intervals.
    Gaps take precedence over silence when both overlap. Pre-roll and post-roll
    outside the track's write interval are represented as explicit zero segments.
    """

    start = max(0.0, float(write_start_ms))
    end = min(float(session_ms), float(write_end_ms))
    if end <= start:
        if float(session_ms) <= 0:
            return []
        return [
            SegmentSpec("pre_roll", 0.0, float(session_ms), 0, ms_to_frames(session_ms))
        ]

    gaps = clip_intervals(gaps_ms, start, end)
    silences = clip_intervals(silence_ms, start, end)

    boundaries = {start, end}
    for interval_start, interval_end in gaps + silences:
        boundaries.add(interval_start)
        boundaries.add(interval_end)
    ordered = sorted(boundaries)

    core: list[SegmentSpec] = []
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        if _inside(gaps, midpoint):
            kind = "gap"
        elif _inside(silences, midpoint):
            kind = "silence"
        else:
            kind = "active"
        _append_merged(core, _segment(kind, left, right))

    segments: list[SegmentSpec] = []
    if start > 0:
        _append_merged(segments, _segment("pre_roll", 0.0, start))
    for item in core:
        _append_merged(segments, item)
    if end < float(session_ms):
        _append_merged(segments, _segment("post_roll", end, float(session_ms)))
    return segments
