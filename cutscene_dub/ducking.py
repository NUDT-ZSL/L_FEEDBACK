from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .audio import SAMPLE_RATE
from .config import RenderConfig
from .errors import BusinessError


@dataclass
class MergedDuckInterval:
    start_frame: int
    end_frame: int
    dialogue_ids: tuple[str, ...]
    source_intervals: tuple[tuple[int, int, str], ...]
    fade_start_frame: int = 0
    fade_end_frame: int = 0
    max_jump_db: float = 0.0


@dataclass
class DuckingResult:
    gain_envelope: np.ndarray
    intervals: list[MergedDuckInterval]
    max_gain_jump_db: float
    measured_dialogue_attenuation_db: float

    def intervals_report(self) -> list[dict[str, Any]]:
        return [
            {
                "duck_start_s": interval.fade_start_frame / float(SAMPLE_RATE),
                "duck_end_s": interval.fade_end_frame / float(SAMPLE_RATE),
                "active_start_s": interval.start_frame / float(SAMPLE_RATE),
                "active_end_s": interval.end_frame / float(SAMPLE_RATE),
                "fade_in_s": (
                    interval.start_frame - interval.fade_start_frame
                )
                / float(SAMPLE_RATE),
                "fade_out_s": (
                    interval.fade_end_frame - interval.end_frame
                )
                / float(SAMPLE_RATE),
                "dialogue_ids": list(interval.dialogue_ids),
                "source_intervals": [
                    {
                        "dialogue_id": dialogue_id,
                        "start_s": start / float(SAMPLE_RATE),
                        "end_s": end / float(SAMPLE_RATE),
                    }
                    for start, end, dialogue_id in interval.source_intervals
                ],
                "max_observed_gain_jump_db": interval.max_jump_db,
            }
            for interval in self.intervals
        ]


def _merge_intervals(
    intervals: Iterable[tuple[int, int, str]],
    *,
    fade_in_frames: int,
    fade_out_frames: int,
) -> list[MergedDuckInterval]:
    sorted_intervals = sorted(
        ((start, end, dialogue_id) for start, end, dialogue_id in intervals if end > start),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[MergedDuckInterval] = []
    threshold = fade_in_frames + fade_out_frames
    for start, end, dialogue_id in sorted_intervals:
        if not merged:
            merged.append(
                MergedDuckInterval(
                    start_frame=start,
                    end_frame=end,
                    dialogue_ids=(dialogue_id,),
                    source_intervals=((start, end, dialogue_id),),
                )
            )
            continue
        current = merged[-1]
        gap = start - current.end_frame
        if gap < threshold:
            current.end_frame = max(current.end_frame, end)
            current.dialogue_ids = current.dialogue_ids + (dialogue_id,)
            current.source_intervals = current.source_intervals + (
                (start, end, dialogue_id),
            )
        else:
            merged.append(
                MergedDuckInterval(
                    start_frame=start,
                    end_frame=end,
                    dialogue_ids=(dialogue_id,),
                    source_intervals=((start, end, dialogue_id),),
                )
            )
    return merged


def apply_ducking(
    music: np.ndarray,
    *,
    dialogue_intervals: Iterable[tuple[int, int, str]],
    config: RenderConfig,
) -> DuckingResult:
    total_frames = int(music.shape[0])
    fade_in_frames = max(
        0, int(round(config.duck_fade_in_ms * SAMPLE_RATE / 1000.0))
    )
    fade_out_frames = max(
        0, int(round(config.duck_fade_out_ms * SAMPLE_RATE / 1000.0))
    )
    intervals = _merge_intervals(
        dialogue_intervals,
        fade_in_frames=fade_in_frames,
        fade_out_frames=fade_out_frames,
    )
    envelope = np.ones(total_frames, dtype=np.float64)
    if total_frames == 0 or not intervals:
        return DuckingResult(
            gain_envelope=envelope,
            intervals=[],
            max_gain_jump_db=0.0,
            measured_dialogue_attenuation_db=0.0,
        )

    floor_gain = 10.0 ** (-config.duck_depth_db / 20.0)
    for interval in intervals:
        start = min(total_frames, max(0, interval.start_frame))
        end = min(total_frames, max(start, interval.end_frame))
        attack_frames = min(fade_in_frames, start)
        release_frames = min(fade_out_frames, total_frames - end)
        interval.fade_start_frame = start - attack_frames
        interval.fade_end_frame = end + release_frames

        if attack_frames > 0:
            positions = np.arange(start - attack_frames, start)
            progress = np.arange(attack_frames, dtype=np.float64) / attack_frames
            envelope[positions] = floor_gain + (1.0 - floor_gain) * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )
        if end > start:
            envelope[start:end] = floor_gain
        if release_frames > 0:
            positions = np.arange(end, end + release_frames)
            progress = np.arange(release_frames, dtype=np.float64) / release_frames
            envelope[positions] = floor_gain + (1.0 - floor_gain) * 0.5 * (
                1.0 - np.cos(np.pi * progress)
            )

    if envelope.size > 1:
        lows = np.minimum(envelope[:-1], envelope[1:])
        highs = np.maximum(envelope[:-1], envelope[1:])
        ratios = np.divide(
            highs,
            lows,
            out=np.ones_like(highs),
            where=lows > 0,
        )
        step_db = np.abs(20.0 * np.log10(np.maximum(1.0, ratios)))
    else:
        step_db = np.zeros(1, dtype=np.float64)

    for interval in intervals:
        lo = max(0, interval.fade_start_frame)
        hi = min(total_frames, interval.fade_end_frame)
        interval.max_jump_db = float(np.max(step_db[lo:hi])) if hi > lo else 0.0
    max_jump = float(np.max(step_db)) if step_db.size else 0.0
    if max_jump > config.max_gain_jump_db + 1e-9:
        offender = max(intervals, key=lambda item: item.max_jump_db)
        raise BusinessError(
            code="E_DUCK_RAMP_UNSAFE",
            message=(
                "Configured ducking fades are too short for the requested depth: "
                "gain change per sample exceeds max_gain_jump_db."
            ),
            entry_id=offender.dialogue_ids[0] if offender.dialogue_ids else None,
            details={
                "max_observed_gain_jump_db": max_jump,
                "max_gain_jump_db": config.max_gain_jump_db,
                "duck_depth_db": config.duck_depth_db,
                "fade_in_ms": config.duck_fade_in_ms,
                "fade_out_ms": config.duck_fade_out_ms,
            },
        )

    active_frames: list[np.ndarray] = []
    for interval in intervals:
        if interval.end_frame > interval.start_frame:
            active_frames.append(
                envelope[interval.start_frame : interval.end_frame]
            )
    if active_frames:
        all_active = np.concatenate(active_frames)
        mean_gain = float(np.mean(all_active))
        measured_attenuation = (
            20.0 * math.log10(mean_gain) if mean_gain > 0 else -math.inf
        )
    else:
        measured_attenuation = 0.0

    return DuckingResult(
        gain_envelope=envelope,
        intervals=intervals,
        max_gain_jump_db=max_jump,
        measured_dialogue_attenuation_db=measured_attenuation,
    )

