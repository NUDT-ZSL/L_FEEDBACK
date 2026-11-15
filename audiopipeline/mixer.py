"""Timeline placement and overlap priority resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CHANNELS, Config
from .manifest import Event, ms_to_samples, samples_to_ms


@dataclass
class PlacedEvent:
    event: Event
    start: int          # global sample index of segment start (after clipping)
    end: int            # exclusive
    segment: np.ndarray
    mix_mode: bool      # True = additive, False = replace/ownership


def _resolves_overlap(a: PlacedEvent, b: PlacedEvent) -> bool:
    """Return True when the two intervals overlap on the timeline."""
    return a.start < b.end and b.start < a.end


def place_events(
    placements: list[tuple[Event, np.ndarray]],
    total_samples: int,
    config: Config,
) -> tuple[np.ndarray, list[PlacedEvent], list[str]]:
    """Render all built segments onto the global stereo timeline.

    Replace-mode events arbitrate sample ownership by ``priority`` (higher
    wins); equal priority keeps the earlier event (first-come-first-served)
    and records a warning.  Mix-mode events always add into the sum buffer.
    """
    timeline = np.zeros((total_samples, CHANNELS), dtype=np.float64)
    mix_buffer = np.zeros_like(timeline)
    owner_priority = np.full(total_samples, -1_000_000_000, dtype=np.int64)
    owner_index = np.full(total_samples, -1, dtype=np.int64)

    placed: list[PlacedEvent] = []
    warnings: list[str] = []

    for event, segment in placements:
        start = ms_to_samples(event.start_ms)
        end = start + segment.shape[0]
        if start >= total_samples:
            event.warnings.append("E_WARN_OUT_OF_TIMELINE")
            warnings.append(f"{event.event_id}: 事件起点超出总时长，未放置")
            continue

        clip_left, clip_right = start, min(end, total_samples)
        clipped_seg = segment[: clip_right - clip_left]
        mode = event.overlap or config.overlap_strategy
        is_mix = mode == "mix"
        placed_event = PlacedEvent(
            event=event,
            start=clip_left,
            end=clip_right,
            segment=clipped_seg,
            mix_mode=is_mix,
        )

        for other in placed:
            if _resolves_overlap(placed_event, other):
                if event.priority == other.event.priority:
                    msg = (
                        f"{event.event_id} 与 {other.event.event_id} 重叠且优先级相同"
                        f"({event.priority})，采用先到先得策略"
                    )
                    event.warnings.append("E_WARN_EQUAL_PRIORITY")
                    other.event.warnings.append("E_WARN_EQUAL_PRIORITY")
                    warnings.append(msg)

        if is_mix:
            mix_buffer[clip_left:clip_right] += clipped_seg
        else:
            region = slice(clip_left, clip_right)
            take = (owner_priority[region] < event.priority) | (
                owner_index[region] == -1
            )
            if not take.any():
                event.warnings.append("E_WARN_REPLACED")
                warnings.append(
                    f"{event.event_id}: 重叠区全部被更高优先级事件占据，事件被替换"
                )
            else:
                if not take.all():
                    event.warnings.append("E_WARN_PARTIAL_REPLACED")
                target_idx = np.where(take)[0] + clip_left
                timeline[target_idx] = clipped_seg[take]
                owner_priority[target_idx] = event.priority
                owner_index[target_idx] = event.index
        placed.append(placed_event)

    # Mix events add into every sample; replace owners occupy their own
    # samples and also receive the additive mix underneath.
    timeline += mix_buffer

    for pe in placed:
        pe.event.actual_start_ms = samples_to_ms(pe.start)
        pe.event.actual_end_ms = samples_to_ms(pe.end)
    return timeline.astype(np.float32), placed, warnings
