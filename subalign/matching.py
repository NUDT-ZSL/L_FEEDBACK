"""Monotonic event matching between a reference and a target timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .subtitles import SubtitleEvent
from .textutil import text_similarity

SKIP_COST = 1.0
TEXT_WEIGHT = 0.6
TIME_TAU_MS = 750.0


@dataclass
class Match:
    reference_index: int
    target_index: int
    reference_start: float
    target_start: float
    text_similarity: float
    time_distance_ms: float
    confidence: float


def _median_gap(events: list[SubtitleEvent]) -> float:
    if len(events) < 2:
        return 1000.0
    starts = np.asarray([event.start for event in events], dtype=float)
    gaps = np.diff(np.sort(starts))
    median = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1000.0
    return max(median, 1.0)


def monotonic_align(
    reference: list[SubtitleEvent],
    target: list[SubtitleEvent],
    shift: float = 0.0,
    text_weight: float = TEXT_WEIGHT,
) -> list[tuple[int, int, float, float]]:
    """Align two ordered cue lists with a deterministic monotonic DP.

    Each pairing combines text dissimilarity with the cost of the residual
    time distance; rows or columns may be skipped at a fixed penalty so that
    missing or extra events are tolerated.
    """

    if not reference or not target:
        return []

    reference_gap = _median_gap(reference)
    target_gap = _median_gap(target)
    tau = max(reference_gap, target_gap, 1.0)
    rows = len(reference)
    cols = len(target)

    similarity = np.zeros((rows, cols), dtype=float)
    time_cost = np.zeros((rows, cols), dtype=float)
    for row, ref_event in enumerate(reference):
        for col, tgt_event in enumerate(target):
            similarity[row, col] = text_similarity(ref_event.text, tgt_event.text)
            distance = abs((tgt_event.start + shift) - ref_event.start)
            time_cost[row, col] = distance / (distance + tau)

    pair_cost = (1.0 - text_weight) * time_cost + text_weight * (1.0 - similarity)

    dp = np.full((rows + 1, cols + 1), np.inf, dtype=float)
    back = np.zeros((rows + 1, cols + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for row in range(rows + 1):
        for col in range(cols + 1):
            if row == 0 and col == 0:
                continue
            candidates: list[tuple[float, int]] = []
            if row > 0:
                candidates.append((dp[row - 1, col] + SKIP_COST, 1))
            if col > 0:
                candidates.append((dp[row, col - 1] + SKIP_COST, 2))
            if row > 0 and col > 0:
                candidates.append((dp[row - 1, col - 1] + pair_cost[row - 1, col - 1], 3))
            best_cost, best_move = min(candidates, key=lambda item: item[0])
            dp[row, col] = best_cost
            back[row, col] = best_move

    pairs: list[tuple[int, int, float, float]] = []
    row, col = rows, cols
    while row > 0 and col > 0:
        move = back[row, col]
        if move == 3:
            ref_index = row - 1
            tgt_index = col - 1
            pairs.append(
                (
                    ref_index,
                    tgt_index,
                    float(similarity[ref_index, tgt_index]),
                    float(
                        abs((target[tgt_index].start + shift) - reference[ref_index].start)
                    ),
                )
            )
            row -= 1
            col -= 1
        elif move == 1:
            row -= 1
        else:
            col -= 1
    pairs.reverse()
    return pairs


def _median_shift(pairs: list[tuple[int, int, float, float]], reference, target) -> float:
    if not pairs:
        return 0.0
    deltas = [
        reference[ref_index].start - target[tgt_index].start
        for ref_index, tgt_index, _sim, _dist in pairs
    ]
    return float(np.median(np.asarray(deltas, dtype=float)))


def _is_translated(pairs: list[tuple[int, int, float, float]]) -> bool:
    if not pairs:
        return False
    mean_similarity = float(np.mean([item[2] for item in pairs]))
    return mean_similarity < 0.35


def match_events(
    reference: list[SubtitleEvent],
    target: list[SubtitleEvent],
) -> tuple[list[Match], bool]:
    """Match cues in two passes so large drifts do not trap the DP locally.

    The first pass uses text anchors and a zero shift; a median shift from its
    result seeds a second pass. Translated targets (no shared script) switch to
    time-only costs.
    """

    first_pairs = monotonic_align(reference, target, shift=0.0)
    translated = _is_translated(first_pairs)
    if translated:
        # No shared script: rely purely on temporal order and proximity.
        pairs = monotonic_align(reference, target, shift=0.0, text_weight=0.0)
    else:
        shift = _median_shift(first_pairs, reference, target)
        pairs = monotonic_align(reference, target, shift=shift, text_weight=TEXT_WEIGHT)

    gap = max(_median_gap(reference), _median_gap(target), 1.0)
    matches: list[Match] = []
    alpha = 0.15 if translated else 0.5
    for ref_index, tgt_index, similarity, _first_distance in pairs:
        ref_event = reference[ref_index]
        tgt_event = target[tgt_index]
        distance = abs(tgt_event.start - ref_event.start)
        time_confidence = float(np.exp(-distance / gap))
        confidence = alpha * time_confidence + (1.0 - alpha) * similarity
        matches.append(
            Match(
                reference_index=ref_index,
                target_index=tgt_index,
                reference_start=ref_event.start,
                target_start=tgt_event.start,
                text_similarity=similarity,
                time_distance_ms=distance,
                confidence=float(round(confidence, 6)),
            )
        )
    return matches, translated
