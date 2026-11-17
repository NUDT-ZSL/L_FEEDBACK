"""Sample-accurate ducking envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


SUPPORTED_CURVES = ("linear", "sine", "cosine", "smoothstep", "exponential")


def db_to_linear(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 20.0))


def linear_to_db(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(value))


def curve_value(curve: str, progress: np.ndarray) -> np.ndarray:
    progress = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    name = curve.strip().lower()
    if name == "linear":
        return progress
    if name in ("sine", "equal_power"):
        return np.sin(progress * np.pi / 2.0)
    if name == "cosine":
        return 0.5 - 0.5 * np.cos(progress * np.pi)
    if name == "smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    if name == "exponential":
        exponent = 5.0
        return (np.exp(exponent * progress) - 1.0) / (np.exp(exponent) - 1.0)
    raise ValueError("unsupported envelope curve: {0}".format(curve))


@dataclass(frozen=True)
class DuckEvent:
    dialogue_id: str
    target_track: str
    start_sample: int
    end_sample: int
    attack_samples: int
    release_samples: int
    depth_db: float
    attack_curve: str
    release_curve: str

    @property
    def depth_linear(self) -> float:
        return db_to_linear(self.depth_db)

    def affected_end_sample(self, total_samples: int) -> int:
        return min(total_samples, self.end_sample + self.release_samples)

    def apply(self, envelope: np.ndarray) -> Dict[str, object]:
        """Apply this event to a track envelope using a per-sample minimum."""

        total_samples = int(envelope.size)
        affected_start = max(0, self.start_sample)
        affected_end = self.affected_end_sample(total_samples)
        if affected_end <= affected_start:
            return {
                "applied_samples": 0,
                "minimum": float(envelope[affected_start])
                if total_samples
                else 1.0,
            }

        duck_gain = self.depth_linear
        local = np.ones(affected_end - affected_start, dtype=np.float64)
        dialogue_start = self.start_sample
        dialogue_end = self.end_sample
        attack_end = min(dialogue_end, dialogue_start + self.attack_samples)

        attack_start_offset = max(0, dialogue_start - affected_start)
        attack_end_offset = max(0, attack_end - affected_start)
        if self.attack_samples > 0 and attack_end_offset > attack_start_offset:
            attack_count = attack_end_offset - attack_start_offset
            progress = (
                np.arange(attack_count, dtype=np.float64) + 1.0
            ) / float(max(1, self.attack_samples))
            shape = curve_value(self.attack_curve, progress)
            local[attack_start_offset:attack_end_offset] = 1.0 + (duck_gain - 1.0) * shape

        active_start_offset = max(0, attack_end - affected_start)
        active_end_offset = max(0, dialogue_end - affected_start)
        if active_end_offset > active_start_offset:
            local[active_start_offset:active_end_offset] = duck_gain

        release_start_offset = max(0, dialogue_end - affected_start)
        release_end_offset = affected_end - affected_start
        if self.release_samples > 0 and release_end_offset > release_start_offset:
            release_count = release_end_offset - release_start_offset
            progress = (
                np.arange(release_count, dtype=np.float64) + 1.0
            ) / float(max(1, self.release_samples))
            shape = curve_value(self.release_curve, progress)
            local[release_start_offset:release_end_offset] = duck_gain + (
                1.0 - duck_gain
            ) * shape

        current = envelope[affected_start:affected_end]
        np.minimum(current, local, out=current)
        return {
            "applied_samples": int(affected_end - affected_start),
            "minimum": float(np.min(current)),
        }


def build_track_envelopes(
    events: Iterable[DuckEvent],
    target_tracks: Iterable[str],
    total_samples: int,
) -> Dict[str, np.ndarray]:
    envelopes = {
        track: np.ones(total_samples, dtype=np.float64) for track in target_tracks
    }
    for event in events:
        if event.target_track in envelopes:
            event.apply(envelopes[event.target_track])
    return envelopes


def envelope_stats(envelope: np.ndarray) -> Dict[str, float]:
    if envelope.size == 0:
        return {"minimum": 1.0, "maximum": 1.0, "ducked_samples": 0}
    return {
        "minimum": float(np.min(envelope)),
        "maximum": float(np.max(envelope)),
        "ducked_samples": int(np.count_nonzero(envelope < 1.0 - 1e-12)),
    }


def find_interval_overlaps(
    intervals: List[Tuple[str, int, int]],
) -> List[Dict[str, object]]:
    """Find pairwise overlaps in half-open sample intervals."""

    ordered = sorted(intervals, key=lambda item: (item[1], item[2], item[0]))
    overlaps: List[Dict[str, object]] = []
    for left_index, left in enumerate(ordered):
        left_id, left_start, left_end = left
        for right in ordered[left_index + 1 :]:
            right_id, right_start, right_end = right
            if right_start >= left_end:
                break
            if right_end <= left_start:
                continue
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_end > overlap_start:
                overlaps.append(
                    {
                        "dialogue_a": left_id,
                        "dialogue_b": right_id,
                        "start_sample": int(overlap_start),
                        "end_sample": int(overlap_end),
                        "duration_samples": int(overlap_end - overlap_start),
                    }
                )
    return overlaps

