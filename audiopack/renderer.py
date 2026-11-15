"""Timeline rendering: per-event slicing, fades, overlap and mixing."""

from __future__ import annotations

import os

import numpy as np

from .audioio import load_clip, to_stereo
from .errors import E_CORRUPT_CLIP, W_SHORT_CLIP, PipelineError
from .loudness import linear_gain
from .memoryutil import MemoryGuard
from .timeline import Event
from .timeutils import ms_to_samples


def render_events(
    events: list[Event],
    input_dir: str,
    sample_rate: int,
    guard: MemoryGuard,
) -> tuple[np.ndarray, list[dict], list[dict], list[str], list[dict]]:
    """Render all events into one float32 stereo mix.

    Returns ``(mix, event_reports, overlap_records, warnings, errors)``.
    """
    boundaries = [
        (
            event,
            ms_to_samples(event.start_ms, sample_rate),
            ms_to_samples(event.end_ms, sample_rate),
        )
        for event in events
    ]
    total_length = max(end for _, _, end in boundaries)

    # Main stereo mix + int32 ownership map + one scratch event segment.
    guard.check_frames(total_length, channels=2)
    guard.check(total_length * 4)
    max_segment = max(end - start for _, start, end in boundaries)
    guard.check_frames(max_segment, channels=2)

    mix = np.zeros((total_length, 2), dtype=np.float32)
    owner = np.full(total_length, -1, dtype=np.int32)
    priority_map = np.zeros(total_length, dtype=np.int32)

    clip_cache: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    warning_details: list[dict] = []
    errors: list[dict] = []
    event_reports: list[dict] = []

    for event, start, end in boundaries:
        length = end - start
        base_report = {
            "event_index": event.index,
            "clip": event.clip,
            "start_sample": int(start),
            "end_sample": int(end),
            "duration_samples": int(length),
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
            "gain_db": event.gain_db,
            "gain_linear": round(linear_gain(event.gain_db), 9),
            "fade_in_samples": ms_to_samples(event.fade_in_ms, sample_rate),
            "fade_out_samples": ms_to_samples(event.fade_out_ms, sample_rate),
            "loop": event.loop,
            "priority": event.priority,
        }

        try:
            segment, used_source_frames, short = _render_segment(
                event, length, sample_rate, input_dir, clip_cache, guard
            )
        except PipelineError as exc:
            errors.append({"code": exc.code, "event_index": event.index, "details": exc.details})
            event_reports.append({**base_report, "status": "error", "error_code": exc.code})
            if exc.code == E_CORRUPT_CLIP:
                continue
            raise

        if short:
            warnings.append(W_SHORT_CLIP)
            warning_details.append(
                {
                    "code": W_SHORT_CLIP,
                    "event_index": event.index,
                    "clip": event.clip,
                    "source_frames": int(used_source_frames),
                    "required_frames": int(length),
                    "padded_silence_frames": int(length - used_source_frames),
                }
            )

        _composite(mix, owner, priority_map, segment, start, end, event)
        event_reports.append(
            {
                **base_report,
                "status": "rendered",
                "source_frames_used": int(used_source_frames),
                "short_clip": bool(short),
            }
        )

    overlap_records = _overlap_records(boundaries)
    return mix, event_reports, overlap_records, _dedupe(warning_details), errors


def _render_segment(
    event: Event,
    length: int,
    sample_rate: int,
    input_dir: str,
    clip_cache: dict[str, np.ndarray],
    guard: MemoryGuard,
) -> tuple[np.ndarray, int, bool]:
    """Build the length-``length`` stereo segment for one event."""
    clip_path = os.path.join(input_dir, event.clip)
    if clip_path not in clip_cache:
        loaded = load_clip(clip_path, sample_rate)
        guard.check_frames(loaded.shape[0], channels=loaded.shape[1])
        clip_cache[clip_path] = loaded
    source = to_stereo(clip_cache[clip_path])
    source_frames = source.shape[0]

    segment = np.zeros((length, 2), dtype=np.float32)
    short = False
    if event.loop:
        if source_frames >= length:
            segment[:] = source[:length]
            used = length
        else:
            reps = int(np.ceil(length / source_frames))
            tiled = np.tile(source, (reps, 1))[:length]
            segment[:] = tiled
            used = source_frames
    else:
        used = min(source_frames, length)
        segment[:used] = source[:used]
        if source_frames < length:
            short = True

    _apply_fades(segment, event, sample_rate)
    segment *= np.float32(linear_gain(event.gain_db))
    return segment, used, short


def _apply_fades(segment: np.ndarray, event: Event, sample_rate: int) -> None:
    """Apply linear head/tail fades; loop junctions are intentionally untouched."""
    length = segment.shape[0]
    fade_in = min(ms_to_samples(event.fade_in_ms, sample_rate), length)
    fade_out = min(ms_to_samples(event.fade_out_ms, sample_rate), length)

    if fade_in > 0:
        ramp_in = np.linspace(0.0, 1.0, fade_in, endpoint=True, dtype=np.float32)
        segment[:fade_in] *= ramp_in[:, np.newaxis]
    if fade_out > 0:
        ramp_out = np.linspace(1.0, 0.0, fade_out, endpoint=True, dtype=np.float32)
        segment[length - fade_out :] *= ramp_out[:, np.newaxis]


def _composite(
    mix: np.ndarray,
    owner: np.ndarray,
    priority_map: np.ndarray,
    segment: np.ndarray,
    start: int,
    end: int,
    event: Event,
) -> None:
    """Overlay ``segment`` following priority / later-index override rules."""
    region_owner = owner[start:end]
    region_priority = priority_map[start:end]
    free = region_owner == -1
    stronger = event.priority > region_priority
    same_priority_later = (event.priority == region_priority) & (region_owner < event.index)
    wins = free | stronger | same_priority_later

    mix[start:end] = np.where(wins[:, np.newaxis], segment, mix[start:end])
    owner[start:end] = np.where(wins, event.index, region_owner)
    priority_map[start:end] = np.where(wins, event.priority, region_priority)


def _overlap_records(boundaries: list[tuple[Event, int, int]]) -> list[dict]:
    """Record every pairwise overlap with winner/loser event indices."""
    records: list[dict] = []
    for pos_a in range(len(boundaries)):
        event_a, start_a, end_a = boundaries[pos_a]
        for pos_b in range(pos_a + 1, len(boundaries)):
            event_b, start_b, end_b = boundaries[pos_b]
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start >= overlap_end:
                continue

            if event_b.priority > event_a.priority:
                winner, loser = event_b.index, event_a.index
                winner_priority = event_b.priority
            elif event_a.priority > event_b.priority:
                winner, loser = event_a.index, event_b.index
                winner_priority = event_a.priority
            else:
                # Equal priority: the later event in the file wins.
                winner, loser = event_b.index, event_a.index
                winner_priority = event_a.priority

            records.append(
                {
                    "overlap_start_sample": int(overlap_start),
                    "overlap_end_sample": int(overlap_end),
                    "overlap_duration_samples": int(overlap_end - overlap_start),
                    "winner_event_index": int(winner),
                    "loser_event_index": int(loser),
                    "winner_priority": int(winner_priority),
                    "rule": "higher_priority"
                    if event_a.priority != event_b.priority
                    else "later_index_on_tie",
                }
            )
    return records


def _dedupe(details: list[dict]) -> list[dict]:
    """Collapse repeated warnings while preserving first-seen order."""
    seen: set[str] = set()
    unique: list[dict] = []
    for item in details:
        key = repr(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
