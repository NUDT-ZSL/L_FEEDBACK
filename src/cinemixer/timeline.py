"""Timeline sample math and event placement.

All time values are milliseconds in the project description and integer
sample positions on the project sample-rate clock. Placement tolerates
negative start times, events extending past the timeline, sources shorter
than the requested region, and millisecond-level overlaps between events.
Out-of-range regions are filled with silence and reported as warnings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logging_util import StructuredLogger
from .project import EventSpec, Project


@dataclass(frozen=True)
class IndexEntry:
    track_id: str
    event_id: str
    start_sample: int
    end_sample: int
    length_samples: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "track_id": self.track_id,
            "event_id": self.event_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "length_samples": self.length_samples,
        }


def ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    """Round a millisecond offset to the nearest sample."""
    return int(round(milliseconds * sample_rate / 1000.0))


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    return samples / float(sample_rate)


@dataclass(frozen=True)
class Placement:
    """Mapping between one timeline region and one source region.

    All indices are clamped to valid ranges; ``copy_length`` may be smaller
    than the event length when the source ends early. The remaining timeline
    samples stay at zero (silence fill).
    """

    timeline_start: int
    timeline_end: int
    source_start: int
    copy_length: int
    event_length: int
    truncated_left: bool
    truncated_right: bool
    source_padded: bool


def plan_placement(
    event: EventSpec,
    project: Project,
    source_length: int,
    timeline_length: int,
) -> Placement:
    rate = project.sample_rate
    raw_start = ms_to_samples(event.start_ms, rate)
    event_length = ms_to_samples(event.duration_ms, rate)
    source_offset = ms_to_samples(event.source_offset_ms, rate)

    timeline_start = raw_start
    truncated_left = timeline_start < 0
    timeline_start = max(timeline_start, 0)

    raw_end = raw_start + event_length
    truncated_right = raw_end > timeline_length
    timeline_end = min(max(raw_end, 0), timeline_length)
    timeline_end = max(timeline_end, timeline_start)

    # Source read region: shift by the same amount when the event was clipped
    # on the left so audio content stays time-aligned.
    left_clip = max(0, -raw_start)
    source_start = source_offset + left_clip
    wanted_length = timeline_end - timeline_start

    available = max(0, source_length - source_start)
    if source_start >= source_length:
        copy_length = 0
    else:
        copy_length = min(wanted_length, max(available, 0))

    source_padded = copy_length < wanted_length
    return Placement(
        timeline_start=timeline_start,
        timeline_end=timeline_end,
        source_start=max(source_start, 0),
        copy_length=copy_length,
        event_length=timeline_end - timeline_start,
        truncated_left=truncated_left,
        truncated_right=truncated_right,
        source_padded=source_padded,
    )


def place_event(
    track_buffer: np.ndarray,
    normalized_source: np.ndarray,
    track_id: str,
    event: EventSpec,
    project: Project,
    timeline_length: int,
    log: StructuredLogger,
) -> IndexEntry:
    """Copy (and silence-pad) one event into the track timeline buffer."""
    source_length = normalized_source.shape[0]
    placement = plan_placement(event, project, source_length, timeline_length)

    if placement.copy_length > 0:
        region = normalized_source[
            placement.source_start : placement.source_start + placement.copy_length
        ]
        track_buffer[
            placement.timeline_start : placement.timeline_start + placement.copy_length
        ] += region

    if placement.truncated_left:
        log.warning(
            "W_EVENT_NEGATIVE_START",
            "event starts before timeline zero; clipped and silence-filled",
            track_id=track_id,
            event_id=event.event_id,
        )
    if placement.truncated_right:
        log.warning(
            "W_EVENT_PAST_TIMELINE",
            "event extends beyond project duration; truncated to timeline",
            track_id=track_id,
            event_id=event.event_id,
        )
    if placement.source_padded:
        log.warning(
            "W_SOURCE_PAD_SILENCE",
            "source shorter than event region; padding with silence",
            track_id=track_id,
            event_id=event.event_id,
        )

    return IndexEntry(
        track_id=track_id,
        event_id=event.event_id,
        start_sample=placement.timeline_start,
        end_sample=placement.timeline_end,
        length_samples=placement.timeline_end - placement.timeline_start,
    )


def index_entry_for(
    track_id: str,
    event: EventSpec,
    project: Project,
    source_length: int,
    timeline_length: int,
) -> IndexEntry:
    """Compute the index entry without touching audio buffers or emitting logs.

    Used for muted, failed and checkpoint-resumed tracks whose sample
    positions must stay identical to the live rendering path.
    """
    placement = plan_placement(event, project, source_length, timeline_length)
    return IndexEntry(
        track_id=track_id,
        event_id=event.event_id,
        start_sample=placement.timeline_start,
        end_sample=placement.timeline_end,
        length_samples=placement.timeline_end - placement.timeline_start,
    )
