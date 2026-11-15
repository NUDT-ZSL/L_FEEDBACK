"""Sample-accurate event scheduling and conflict resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import EventSpec
from .spatial import SpatialGains


def seconds_to_samples(seconds: float, sample_rate: int = 48000) -> int:
    """Round a non-negative time in seconds to the nearest sample."""

    return int(math.floor(float(seconds) * int(sample_rate) + 0.5))


def samples_to_milliseconds(samples: int, sample_rate: int = 48000) -> int:
    """Round a sample position to the nearest millisecond (half up)."""

    return int(math.floor(float(samples) * 1000.0 / float(sample_rate) + 0.5))


@dataclass
class PreparedEvent:
    spec: EventSpec
    mono: np.ndarray
    spatial: SpatialGains
    source_sample_rate: int
    source_frames: int
    source_channels: int
    fit_kind: str
    requested_start_sample: int
    requested_length: int


@dataclass
class ScheduledEvent:
    prepared: PreparedEvent
    start_sample: Optional[int]
    dropped: bool = False
    reason: str = ""
    delay_samples: int = 0
    conflict_indices: List[int] = field(default_factory=list)

    @property
    def end_sample(self) -> Optional[int]:
        if self.start_sample is None:
            return None
        return self.start_sample + int(self.prepared.mono.shape[0])

    @property
    def spec(self) -> EventSpec:
        return self.prepared.spec


def _overlaps(start: int, end: int, other: ScheduledEvent) -> bool:
    if other.dropped or other.start_sample is None:
        return False
    other_end = other.end_sample
    if other_end is None:
        return False
    return start < other_end and other.start_sample < end


def schedule_events(prepared_events: List[PreparedEvent]) -> List[ScheduledEvent]:
    """Resolve overlaps in input order according to each event policy.

    ``mix`` events opt into overlap and are never blocking events.  ``drop``
    removes the current event if any blocking event has an equal or higher
    priority; if the current event has higher priority, all blocking events are
    removed.  ``delay`` moves the current event to the first gap after all
    blocking events it overlaps.
    """

    scheduled: List[ScheduledEvent] = []
    accepted: List[ScheduledEvent] = []

    for prepared in prepared_events:
        requested_start = prepared.requested_start_sample
        requested_end = requested_start + int(prepared.mono.shape[0])
        entry = ScheduledEvent(
            prepared=prepared,
            start_sample=requested_start,
        )
        scheduled.append(entry)

        if prepared.spec.conflict_policy == "mix":
            overlapping = [
                item.spec.index
                for item in accepted
                if _overlaps(requested_start, requested_end, item)
            ]
            if overlapping:
                entry.reason = (
                    "overlap allowed by mix policy; mixed with event(s) {}".format(
                        ", ".join(str(index) for index in overlapping)
                    )
                )
            else:
                entry.reason = "no timeline adjustment"
            accepted.append(entry)
            continue

        blockers = [
            item
            for item in accepted
            if item.spec.conflict_policy != "mix"
            and _overlaps(requested_start, requested_end, item)
        ]

        if prepared.spec.conflict_policy == "drop":
            if not blockers:
                entry.reason = "no overlap conflict"
                accepted.append(entry)
                continue

            highest_blocker = max(blockers, key=lambda item: item.spec.priority)
            if prepared.spec.priority <= highest_blocker.spec.priority:
                entry.dropped = True
                entry.start_sample = None
                entry.conflict_indices = [item.spec.index for item in blockers]
                entry.reason = (
                    "dropped by drop policy: overlaps event {} with priority {} "
                    ">= current priority {}".format(
                        highest_blocker.spec.index,
                        _format_priority(highest_blocker.spec.priority),
                        _format_priority(prepared.spec.priority),
                    )
                )
                continue

            removed_indices = []
            for blocker in blockers:
                blocker.dropped = True
                blocker.start_sample = None
                blocker.conflict_indices = [prepared.spec.index]
                blocker.reason = (
                    "dropped by drop policy: lower priority {} replaced by "
                    "event {} with priority {}".format(
                        _format_priority(blocker.spec.priority),
                        prepared.spec.index,
                        _format_priority(prepared.spec.priority),
                    )
                )
                removed_indices.append(blocker.spec.index)
            accepted = [item for item in accepted if item not in blockers]
            entry.conflict_indices = removed_indices
            entry.reason = (
                "replaced lower-priority overlapping event(s) {}".format(
                    ", ".join(str(index) for index in removed_indices)
                )
            )
            accepted.append(entry)
            continue

        if prepared.spec.conflict_policy == "delay":
            if not blockers:
                entry.reason = "no overlap conflict"
                accepted.append(entry)
                continue

            original_start = requested_start
            candidate = requested_start
            involved = set()
            while True:
                current_blockers = [
                    item
                    for item in accepted
                    if item.spec.conflict_policy != "mix"
                    and _overlaps(
                        candidate,
                        candidate + int(prepared.mono.shape[0]),
                        item,
                    )
                ]
                if not current_blockers:
                    break
                for item in current_blockers:
                    involved.add(item.spec.index)
                candidate = max(item.end_sample for item in current_blockers)

            entry.start_sample = candidate
            entry.delay_samples = candidate - original_start
            entry.conflict_indices = sorted(involved)
            if entry.delay_samples:
                entry.reason = (
                    "delayed from {} ms to {} ms to avoid overlap with "
                    "event(s) {}".format(
                        samples_to_milliseconds(original_start),
                        samples_to_milliseconds(candidate),
                        ", ".join(str(index) for index in entry.conflict_indices),
                    )
                )
            else:
                entry.reason = "no timeline adjustment"
            accepted.append(entry)
            continue

        raise AssertionError("unhandled conflict policy: {}".format(prepared.spec.conflict_policy))

    return scheduled


def _format_priority(priority: float) -> str:
    if float(priority).is_integer():
        return str(int(priority))
    return "{:.12g}".format(priority)
