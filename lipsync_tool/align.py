"""Lip-event alignment on the stretched audio timeline.

Each lip event is first mapped onto the audio timeline, then an energy based
search limited to a configurable tolerance (default 50 ms) proposes candidate
positions.  A dynamic program jointly minimizes per-event energy mismatch,
absolute shift and overlap between neighbouring events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SilenceSegment:
    start_time: float
    end_time: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "silence",
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
        }


def frame_energy_envelope(
    signal: np.ndarray, sample_rate: int, frame_ms: float = 20.0, hop_ms: float = 5.0
) -> Tuple[np.ndarray, float]:
    """Return an RMS energy envelope sampled every ``hop_ms``.

    Returns the envelope array and the hop size in seconds.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if signal.size == 0:
        return np.zeros(0, dtype=np.float64), hop_length / float(sample_rate)

    window = np.hanning(frame_length)
    starts = np.arange(0, max(1, signal.size - frame_length + 1), hop_length)
    if starts.size == 0:
        starts = np.array([0])
    envelope = np.empty(starts.size, dtype=np.float64)
    for index, start in enumerate(starts):
        frame = signal[start : min(signal.size, start + frame_length)]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        envelope[index] = float(np.sqrt(np.mean((frame * window) ** 2) + 1e-12))
    return envelope, hop_length / float(sample_rate)


def detect_silence(
    signal: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    min_silence_ms: float = 500.0,
) -> List[SilenceSegment]:
    """Detect silence runs longer than ``min_silence_ms`` on the stretched audio."""
    envelope, hop = frame_energy_envelope(signal, sample_rate)
    if envelope.size == 0:
        return []
    threshold = 10.0 ** (threshold_db / 20.0)
    is_silent = envelope < threshold
    segments: List[SilenceSegment] = []
    run_start: int | None = None
    for index, silent in enumerate(is_silent):
        if silent and run_start is None:
            run_start = index
        elif not silent and run_start is not None:
            _append_silence(segments, run_start, index, hop, min_silence_ms)
            run_start = None
    if run_start is not None:
        _append_silence(segments, run_start, is_silent.size, hop, min_silence_ms)
    return segments


def _append_silence(
    segments: List[SilenceSegment], start_idx: int, end_idx: int, hop: float, min_ms: float
) -> None:
    start_time = start_idx * hop
    end_time = end_idx * hop
    duration = end_time - start_time
    if duration * 1000.0 >= min_ms:
        segments.append(SilenceSegment(start_time, end_time, duration))


def map_events_to_audio_time(
    events: Sequence[Dict[str, Any]], frame_rate: float
) -> List[Dict[str, Any]]:
    """Convert frame-numbered lip events into second-based audio timeline events."""
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    mapped: List[Dict[str, Any]] = []
    for event in events:
        start_frame = float(event["start_frame"])
        duration_frames = float(event.get("duration_frames", 1))
        start_time = start_frame / float(frame_rate)
        duration = max(duration_frames / float(frame_rate), 1.0 / float(frame_rate))
        mapped.append(
            {
                "id": event.get("id"),
                "phoneme": event.get("phoneme", "unknown"),
                "start_time": start_time,
                "duration": duration,
                "end_time": start_time + duration,
                "original_start_frame": int(start_frame),
            }
        )
    return mapped


def align_events(
    mapped_events: Sequence[Dict[str, Any]],
    signal: np.ndarray,
    sample_rate: int,
    tolerance_ms: float = 50.0,
    overlap_penalty_weight: float = 1.0,
    hop_ms: float = 5.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Align mapped events to energy peaks via dynamic programming.

    Candidate shifts form a grid from ``-tolerance_ms`` to ``+tolerance_ms``.
    The joint cost combines normalized inverse-energy mismatch, absolute shift
    (in milliseconds) and an overlap penalty for neighbouring events.
    """
    envelope, hop = frame_energy_envelope(signal, sample_rate, hop_ms=hop_ms)
    if envelope.size:
        peak_energy = float(np.max(envelope)) or 1.0
    else:
        peak_energy = 1.0

    tolerance_seconds = tolerance_ms / 1000.0
    shift_grid = np.arange(-tolerance_seconds, tolerance_seconds + hop / 2.0, hop)
    shift_grid = shift_grid[np.abs(shift_grid) <= tolerance_seconds + 1e-9]

    candidate_table: List[np.ndarray] = []
    energy_cost_table: List[np.ndarray] = []
    for event in mapped_events:
        nominal = float(event["start_time"])
        candidates = np.clip(nominal + shift_grid, 0.0, max(0.0, signal.size / sample_rate - hop))
        bins = np.clip(np.round(candidates / hop).astype(int), 0, max(0, envelope.size - 1))
        energy_cost = 1.0 - envelope[bins] / peak_energy if envelope.size else np.ones_like(bins, dtype=float)
        candidate_table.append(candidates)
        energy_cost_table.append(energy_cost)

    # Dynamic programming over (event, candidate shift).
    number_events = len(mapped_events)
    number_candidates = shift_grid.size
    cost = np.full((number_events, number_candidates), np.inf, dtype=np.float64)
    back_pointer = np.zeros((number_events, number_candidates), dtype=np.int64)
    shift_cost_scale = 1.0 / max(tolerance_ms, 1e-9)

    for event_index in range(number_events):
        candidates = candidate_table[event_index]
        durations = float(mapped_events[event_index]["duration"])
        for candidate_index in range(number_candidates):
            local_cost = float(
                energy_cost_table[event_index][candidate_index]
                + 0.5 * abs(shift_grid[candidate_index]) * 1000.0 * shift_cost_scale
            )
            if event_index == 0:
                cost[event_index, candidate_index] = local_cost
                continue
            previous_candidates = candidate_table[event_index - 1]
            overlap = np.maximum(
                0.0, previous_candidates + float(mapped_events[event_index - 1]["duration"]) - candidates[candidate_index]
            )
            transition = overlap_penalty_weight * overlap / max(durations, 1e-6)
            previous_cost = cost[event_index - 1] + transition
            best_previous = int(np.argmin(previous_cost))
            back_pointer[event_index, candidate_index] = best_previous
            cost[event_index, candidate_index] = local_cost + float(previous_cost[best_previous])

    aligned: List[Dict[str, Any]] = []
    if number_events:
        chosen = [0] * number_events
        chosen[-1] = int(np.argmin(cost[-1]))
        for event_index in range(number_events - 1, 0, -1):
            chosen[event_index - 1] = int(back_pointer[event_index, chosen[event_index]])

        for event_index, event in enumerate(mapped_events):
            candidate_index = chosen[event_index]
            aligned_time = float(candidate_table[event_index][candidate_index])
            shift_ms = float(shift_grid[candidate_index] * 1000.0)
            aligned.append(
                {
                    "id": event["id"],
                    "phoneme": event["phoneme"],
                    "start_time": round(aligned_time, 4),
                    "duration": round(float(event["duration"]), 4),
                    "end_time": round(aligned_time + float(event["duration"]), 4),
                    "nominal_start_time": round(float(event["start_time"]), 4),
                    "offset_ms": round(shift_ms, 3),
                    "original_start_frame": event["original_start_frame"],
                }
            )

    offsets_ms = [item["offset_ms"] for item in aligned]
    stats = {
        "event_count": float(number_events),
        "mean_offset_ms": float(np.mean(np.abs(offsets_ms))) if offsets_ms else 0.0,
        "max_offset_ms": float(np.max(np.abs(offsets_ms))) if offsets_ms else 0.0,
    }
    return aligned, stats


def coverage_ratio(events: Sequence[Dict[str, Any]], timeline_duration: float) -> float:
    """Fraction of the timeline covered by merged event intervals."""
    if not events or timeline_duration <= 0:
        return 0.0
    intervals = sorted(
        (float(event["start_time"]), float(event["end_time"])) for event in events
    )
    covered = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return float(np.clip(covered / timeline_duration, 0.0, 1.0))


def uncovered_segments(
    events: Sequence[Dict[str, Any]], timeline_duration: float, min_gap_ms: float = 100.0
) -> List[Dict[str, Any]]:
    """Return timeline gaps not covered by any lip event."""
    gaps: List[Dict[str, Any]] = []
    cursor = 0.0
    for start, end in sorted(
        (float(event["start_time"]), float(event["end_time"])) for event in events
    ):
        if start - cursor >= min_gap_ms / 1000.0:
            gaps.append(
                {
                    "type": "uncovered_gap",
                    "start_time": round(cursor, 4),
                    "end_time": round(start, 4),
                    "duration": round(start - cursor, 4),
                }
            )
        cursor = max(cursor, end)
    if timeline_duration - cursor >= min_gap_ms / 1000.0:
        gaps.append(
            {
                "type": "uncovered_gap",
                "start_time": round(cursor, 4),
                "end_time": round(timeline_duration, 4),
                "duration": round(timeline_duration - cursor, 4),
            }
        )
    return gaps
