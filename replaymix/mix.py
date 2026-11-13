"""Deterministic arithmetic-sum mixing on the global session timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .track import TrackResult


def _track_sort_key(track_id: Any) -> tuple[int, Any]:
    if isinstance(track_id, int) and not isinstance(track_id, bool):
        return (0, track_id)
    text = str(track_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


@dataclass
class MixResult:
    samples: np.ndarray
    total_frames: int
    peak_before_clip: float
    peak_after_clip: float
    clipped_sample_count: int
    out_of_bounds_frames_discarded: int
    last_active_end_frame: int

    @property
    def tail_padding_frames(self) -> int:
        return max(0, self.total_frames - self.last_active_end_frame)


def mix_tracks(results: Iterable[TrackResult], total_frames: int) -> MixResult:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    mix = np.zeros(total_frames, dtype=np.float64)
    last_active_end = 0
    discarded = 0

    ordered = sorted(results, key=lambda item: _track_sort_key(item.track_id))
    for result in ordered:
        if result.status not in ("processed", "processed_with_errors"):
            continue
        for segment in result.active_audio:
            start = int(segment.start_frame)
            end = start + int(segment.audio.size)
            if start < 0:
                trim = min(-start, segment.audio.size)
                audio = segment.audio[trim:]
                discarded += trim
                start = 0
                end = start + int(audio.size)
            else:
                audio = segment.audio
            if start >= total_frames:
                discarded += int(audio.size)
                continue
            if end > total_frames:
                trim = end - total_frames
                audio = audio[: max(0, audio.size - trim)]
                discarded += trim
                end = total_frames
            if end <= start:
                discarded += int(audio.size)
                continue
            mix[start:end] += audio[: end - start]
            last_active_end = max(last_active_end, end)

    peak_before = float(np.max(np.abs(mix))) if mix.size else 0.0
    clipped = np.clip(mix, -1.0, 1.0)
    clipped_count = int(np.count_nonzero(clipped != mix))
    peak_after = float(np.max(np.abs(clipped))) if clipped.size else 0.0
    return MixResult(
        samples=clipped,
        total_frames=total_frames,
        peak_before_clip=peak_before,
        peak_after_clip=peak_after,
        clipped_sample_count=clipped_count,
        out_of_bounds_frames_discarded=discarded,
        last_active_end_frame=last_active_end,
    )
