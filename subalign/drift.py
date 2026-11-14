"""Drift estimation, piecewise segmentation and deterministic correction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .matching import Match
from .subtitles import SubtitleEvent

MIN_SEGMENT_MATCHES = 4
SPLIT_RESIDUAL_MS = 200.0
SEGMENT_SSE_RATIO = 0.5


@dataclass
class SegmentModel:
    index: int
    start_time_ms: float
    end_time_ms: float
    scale: float
    offset_ms: float
    match_count: int
    rms_residual_ms: float


@dataclass
class DriftModel:
    kind: str
    scale: float
    offset_ms: float
    segments: list[SegmentModel] = field(default_factory=list)
    rms_residual_ms: float = 0.0


@dataclass
class AlignmentResult:
    events: list[SubtitleEvent]
    model: DriftModel
    matches: list[Match]
    warnings: list[dict]
    stats: dict


def _weighted_linear_fit(
    target_times: np.ndarray,
    reference_times: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Fit ``reference = scale * target + offset`` with weighted least squares."""

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones_like(weights)
        weight_sum = float(np.sum(weights))
    weighted_target = np.sum(weights * target_times) / weight_sum
    weighted_reference = np.sum(weights * reference_times) / weight_sum
    centered_target = target_times - weighted_target
    denominator = float(np.sum(weights * centered_target * centered_target))
    if denominator <= 1e-9:
        scale = 1.0
        offset = float(weighted_reference - weighted_target)
    else:
        scale = float(
            np.sum(weights * centered_target * (reference_times - weighted_reference))
            / denominator
        )
        offset = float(weighted_reference - scale * weighted_target)
    predicted = scale * target_times + offset
    rms = float(np.sqrt(np.mean((predicted - reference_times) ** 2)))
    return scale, offset, rms


def fit_global(matches: list[Match]) -> DriftModel:
    target_times = np.asarray([match.target_start for match in matches], dtype=float)
    reference_times = np.asarray([match.reference_start for match in matches], dtype=float)
    weights = np.asarray([max(match.confidence, 1e-3) for match in matches], dtype=float)

    scale, offset, rms = _weighted_linear_fit(target_times, reference_times, weights)

    if matches:
        residuals = np.abs(scale * target_times + offset - reference_times)
        inlier_mask = residuals <= max(2.0 * rms, 25.0)
        if int(np.sum(inlier_mask)) >= 3 and int(np.sum(inlier_mask)) < len(matches):
            scale, offset, rms = _weighted_linear_fit(
                target_times[inlier_mask],
                reference_times[inlier_mask],
                weights[inlier_mask],
            )

    return DriftModel(kind="linear", scale=scale, offset_ms=offset, rms_residual_ms=rms)


def _fit_segment(
    index: int,
    segment_matches: list[Match],
    start_time: float,
    end_time: float,
) -> tuple[SegmentModel, np.ndarray, float]:
    target_times = np.asarray([match.target_start for match in segment_matches], dtype=float)
    reference_times = np.asarray(
        [match.reference_start for match in segment_matches], dtype=float
    )
    weights = np.asarray(
        [max(match.confidence, 1e-3) for match in segment_matches], dtype=float
    )
    scale, offset, rms = _weighted_linear_fit(target_times, reference_times, weights)
    model = SegmentModel(
        index=index,
        start_time_ms=float(round(start_time, 3)),
        end_time_ms=float(round(end_time, 3)),
        scale=float(round(scale, 6)),
        offset_ms=float(round(offset, 3)),
        match_count=len(segment_matches),
        rms_residual_ms=float(round(rms, 3)),
    )
    residuals = scale * target_times + offset - reference_times
    sse = float(np.sum(residuals * residuals))
    return model, residuals, sse


def fit_segmented(matches: list[Match]) -> DriftModel:
    """Detect abrupt drift changes with deterministic top-down binary splitting."""

    ordered = sorted(matches, key=lambda item: item.target_start)
    global_model = fit_global(ordered)
    if len(ordered) < MIN_SEGMENT_MATCHES * 2:
        return global_model

    target_times = np.asarray([match.target_start for match in ordered], dtype=float)
    reference_times = np.asarray([match.reference_start for match in ordered], dtype=float)
    global_sse = float(
        np.sum(
            (
                global_model.scale * target_times
                + global_model.offset_ms
                - reference_times
            )
            ** 2
        )
    )
    global_residual = float(np.sqrt(global_sse / len(ordered)))
    if global_residual < SPLIT_RESIDUAL_MS:
        return global_model

    ranges: list[tuple[int, int]] = [(0, len(ordered))]
    segment_models: list[SegmentModel] = []
    segmented_sse = 0.0

    while ranges:
        begin, end = ranges.pop(0)
        segment_matches = ordered[begin:end]
        start_time = segment_matches[0].target_start
        end_time = segment_matches[-1].target_start
        model, residuals, segment_sse = _fit_segment(
            len(segment_models), segment_matches, start_time, end_time
        )

        split_at = -1
        best_sse = segment_sse
        if len(segment_matches) >= MIN_SEGMENT_MATCHES * 2:
            for cut in range(begin + MIN_SEGMENT_MATCHES, end - MIN_SEGMENT_MATCHES + 1):
                left = ordered[begin:cut]
                right = ordered[cut:end]
                _, _, left_sse = _fit_segment(0, left, left[0].target_start, left[-1].target_start)
                _, _, right_sse = _fit_segment(
                    0, right, right[0].target_start, right[-1].target_start
                )
                if left_sse + right_sse < best_sse:
                    best_sse = left_sse + right_sse
                    split_at = cut

        should_split = (
            split_at != -1
            and best_sse <= SEGMENT_SSE_RATIO * segment_sse
            and np.max(np.abs(residuals)) >= SPLIT_RESIDUAL_MS
        )
        if should_split:
            ranges.append((begin, split_at))
            ranges.append((split_at, end))
        else:
            segment_models.append(model)
            segmented_sse += segment_sse

    segment_models.sort(key=lambda item: item.start_time_ms)
    for index, model in enumerate(segment_models):
        model.index = index

    if len(segment_models) < 2 or segmented_sse > SEGMENT_SSE_RATIO * global_sse:
        return global_model

    rms = float(np.sqrt(segmented_sse / len(ordered)))
    return DriftModel(
        kind="segmented",
        scale=segment_models[0].scale,
        offset_ms=segment_models[0].offset_ms,
        segments=segment_models,
        rms_residual_ms=rms,
    )


def predict_time(model: DriftModel, target_start: float) -> float:
    if model.kind != "segmented" or not model.segments:
        return model.scale * target_start + model.offset_ms
    selected = model.segments[0]
    for segment in model.segments:
        if segment.start_time_ms <= target_start <= segment.end_time_ms:
            selected = segment
            break
        if segment.start_time_ms <= target_start:
            selected = segment
    return selected.scale * target_start + selected.offset_ms


def preflight_warnings(events: list[SubtitleEvent]) -> list[dict]:
    warnings: list[dict] = []
    for event in events:
        if event.duration <= 0:
            warnings.append(
                {
                    "code": "ZERO_DURATION",
                    "event_index": event.index,
                    "message": "event has zero or negative duration; timing preserved",
                }
            )
        if event.start < 0:
            warnings.append(
                {
                    "code": "NEGATIVE_START",
                    "event_index": event.index,
                    "start_ms": int(round(event.start)),
                    "message": "event starts before zero; it is clamped to 0 ms",
                }
            )
    for previous, current in zip(events, events[1:]):
        if current.start < previous.start:
            warnings.append(
                {
                    "code": "OUT_OF_ORDER",
                    "event_index": current.index,
                    "message": "event start precedes the previous event; order is preserved",
                }
            )
    return warnings


def sanitize(events: list[SubtitleEvent]) -> list[dict]:
    """Apply deterministic rules: clamp negatives and keep the first overlap."""

    warnings: list[dict] = []
    for event in events:
        if event.start < 0.0:
            event.start = 0.0
            event.end = max(event.end, 0.0)

    latest_end: float | None = None
    latest_owner: int | None = None
    for event in events:
        if latest_end is not None and event.start < latest_end:
            warnings.append(
                {
                    "code": "OVERLAP",
                    "event_index": event.index,
                    "conflicts_with": latest_owner,
                    "message": "overlapping event kept but shifted to start after the preceding event",
                }
            )
            event.start = latest_end
            event.end = max(event.end, event.start)
        latest_end = event.end
        latest_owner = event.index
    return warnings


def align_events(
    reference: list[SubtitleEvent],
    target: list[SubtitleEvent],
    matches: list[Match],
    model: DriftModel,
) -> AlignmentResult:
    warnings = preflight_warnings(target)
    aligned = [
        SubtitleEvent(
            index=event.index,
            start=event.start,
            end=event.end,
            text=event.text,
            style=event.style,
            name=event.name,
            effect=event.effect,
            layer=event.layer,
            marginl=event.marginl,
            marginr=event.marginr,
            marginv=event.marginv,
        )
        for event in target
    ]

    matched_targets = {match.target_index for match in matches}
    for event in aligned:
        if event.index in matched_targets:
            duration = event.duration
            event.start = predict_time(model, event.start)
            event.end = event.start + duration
        else:
            warnings.append(
                {
                    "code": "UNMATCHED_EVENT",
                    "event_index": event.index,
                    "message": "no reference match found; original timing retained",
                }
            )

    warnings.extend(sanitize(aligned))

    confidences = np.asarray([match.confidence for match in matches], dtype=float)
    stats = {
        "reference_event_count": len(reference),
        "target_event_count": len(target),
        "matched_event_count": len(matches),
        "unmatched_target_count": len(target) - len(matched_targets),
        "mean_confidence": float(round(float(np.mean(confidences)), 6))
        if len(confidences)
        else 0.0,
        "min_confidence": float(round(float(np.min(confidences)), 6))
        if len(confidences)
        else 0.0,
    }
    return AlignmentResult(
        events=aligned,
        model=model,
        matches=matches,
        warnings=warnings,
        stats=stats,
    )
