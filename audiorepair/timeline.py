"""Timeline loading, schema validation and sample-accurate segmentation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import (
    INPUT_MISSING,
    STRETCH_UNREACHABLE,
    TIMELINE_INVALID,
    TIMELINE_OUT_OF_RANGE,
    RepairError,
)

DEFAULT_TARGET_LUFS = -23.0
DEFAULT_TOLERANCE_LU = 0.5
DEFAULT_CROSSFADE_MS = 20.0
DEFAULT_STRETCH_RANGE = (0.8, 1.25)


@dataclass(frozen=True)
class SegmentSpec:
    index: int
    segment_id: str
    start_sample: int
    end_sample: int
    target_lufs: float
    stretch_ratio: float
    stretch_min: float
    stretch_max: float
    source_anchors: Tuple[int, ...]
    expected_anchors: Tuple[int, ...]
    output_length: int
    crossfade_out_ms: Optional[float]


@dataclass(frozen=True)
class Timeline:
    path: str
    target_lufs: float
    tolerance_lu: float
    crossfade_ms: Tuple[float, ...]
    segments: Tuple[SegmentSpec, ...]


def _timeline_invalid(message: str, **details: Any) -> RepairError:
    return RepairError(TIMELINE_INVALID, message, details or None)


def _timeline_range(message: str, **details: Any) -> RepairError:
    return RepairError(TIMELINE_OUT_OF_RANGE, message, details or None)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: Any, field: str) -> float:
    if not _is_number(value):
        raise _timeline_invalid("{} must be a finite number".format(field),
                                field=field)
    return float(value)


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    if number != math.floor(number):
        raise _timeline_invalid("{} must be an integer".format(field),
                                field=field)
    return int(number)


def _sample_from_seconds(value: Any, field: str, sample_rate: int) -> int:
    return int(round(_number(value, field) * sample_rate))


def _resolve_point(
    item: Dict[str, Any],
    time_keys: Sequence[str],
    sample_keys: Sequence[str],
    sample_rate: int,
    field: str,
) -> int:
    for key in sample_keys:
        if key in item:
            return _integer(item[key], "{}.{}".format(field, key))
    for key in time_keys:
        if key in item:
            return _sample_from_seconds(
                item[key], "{}.{}".format(field, key), sample_rate
            )
    raise _timeline_invalid(
        "{} needs one of {} or {}".format(
            field, ", ".join(time_keys), ", ".join(sample_keys)
        ),
        field=field,
    )


def _parse_range(segment: Dict[str, Any], index: int) -> Tuple[float, float]:
    raw = None
    for key in ("stretch_range", "stretch_ratio_range", "reachable_range"):
        if key in segment:
            raw = segment[key]
            break
    if raw is None:
        return DEFAULT_STRETCH_RANGE
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise _timeline_invalid(
            "segments[{}].stretch_range must contain [min, max]".format(index)
        )
    lower = _number(raw[0], "segments[{}].stretch_range[0]".format(index))
    upper = _number(raw[1], "segments[{}].stretch_range[1]".format(index))
    if lower <= 0.0 or upper <= 0.0 or lower > upper:
        raise _timeline_invalid(
            "segments[{}].stretch_range must satisfy 0 < min <= max".format(index)
        )
    return lower, upper


def _parse_anchors(
    segment: Dict[str, Any],
    index: int,
    segment_frames: int,
    output_frames: int,
    sample_rate: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    source: List[int] = []
    expected: List[int] = []
    boundaries = segment.get("boundaries")

    if boundaries is not None:
        if not isinstance(boundaries, list):
            raise _timeline_invalid(
                "segments[{}].boundaries must be an array".format(index)
            )
        for boundary_index, boundary in enumerate(boundaries):
            field = "segments[{}].boundaries[{}]".format(index, boundary_index)
            if not isinstance(boundary, dict):
                raise _timeline_invalid("{} must be an object".format(field))
            source.append(
                _resolve_point(
                    boundary,
                    ("source_sec", "input_sec", "from_sec"),
                    ("source_sample", "input_sample", "from_sample"),
                    sample_rate,
                    field,
                )
            )
            expected.append(
                _resolve_point(
                    boundary,
                    ("expected_sec", "target_sec", "output_sec", "to_sec"),
                    ("expected_sample", "target_sample", "output_sample", "to_sample"),
                    sample_rate,
                    field,
                )
            )
    else:
        source_values = segment.get("source_boundaries_sec")
        expected_values = segment.get("expected_boundaries_sec")
        if source_values is None and expected_values is None:
            return (), ()
        if not isinstance(source_values, list) or not isinstance(expected_values, list):
            raise _timeline_invalid(
                "segments[{}] source_boundaries_sec and "
                "expected_boundaries_sec must both be arrays".format(index)
            )
        if len(source_values) != len(expected_values):
            raise _timeline_invalid(
                "segments[{}] boundary arrays differ in length".format(index)
            )
        for boundary_index, value in enumerate(source_values):
            source.append(
                _sample_from_seconds(
                    value,
                    "segments[{}].source_boundaries_sec[{}]".format(
                        index, boundary_index
                    ),
                    sample_rate,
                )
            )
        for boundary_index, value in enumerate(expected_values):
            expected.append(
                _sample_from_seconds(
                    value,
                    "segments[{}].expected_boundaries_sec[{}]".format(
                        index, boundary_index
                    ),
                    sample_rate,
                )
            )

    # Accept explicit segment endpoints as anchors, but do not duplicate them.
    if source and source[0] == 0 and expected and expected[0] == 0:
        source.pop(0)
        expected.pop(0)
    if (
        source
        and source[-1] == segment_frames
        and expected
        and expected[-1] == output_frames
    ):
        source.pop()
        expected.pop()

    if len(source) != len(expected):
        raise _timeline_invalid(
            "segments[{}] source and expected anchors differ in length".format(index)
        )
    previous_source = 0
    previous_expected = 0
    for anchor_index, (source_sample, expected_sample) in enumerate(
        zip(source, expected)
    ):
        if source_sample <= previous_source or source_sample >= segment_frames:
            raise _timeline_range(
                "segments[{}] source anchor {} is outside (0, {})".format(
                    index, anchor_index, segment_frames
                ),
                segment=index,
                anchor=anchor_index,
            )
        if expected_sample <= previous_expected or expected_sample >= output_frames:
            raise _timeline_range(
                "segments[{}] expected anchor {} is outside (0, {})".format(
                    index, anchor_index, output_frames
                ),
                segment=index,
                anchor=anchor_index,
            )
        previous_source = source_sample
        previous_expected = expected_sample
    return tuple(source), tuple(expected)


def _validate_anchor_ratios(
    segment: SegmentSpec,
) -> None:
    source_points = (0,) + segment.source_anchors + (segment.end_sample - segment.start_sample,)
    expected_points = (0,) + segment.expected_anchors + (segment.output_length,)
    for span in range(len(source_points) - 1):
        source_length = source_points[span + 1] - source_points[span]
        target_length = expected_points[span + 1] - expected_points[span]
        if source_length <= 0 or target_length <= 0:
            raise _timeline_range(
                "segments[{}] anchors produce a zero-length span".format(
                    segment.index
                ),
                segment=segment.index,
                span=span,
            )
        ratio = target_length / float(source_length)
        if ratio < segment.stretch_min or ratio > segment.stretch_max:
            raise RepairError(
                STRETCH_UNREACHABLE,
                "segments[{}] span {} requires ratio {:.6f} outside [{}, {}]".format(
                    segment.index,
                    span,
                    ratio,
                    segment.stretch_min,
                    segment.stretch_max,
                ),
                {
                    "segment": segment.index,
                    "span": span,
                    "required_ratio": round(ratio, 9),
                    "allowed": [segment.stretch_min, segment.stretch_max],
                },
            )


def parse_timeline_document(
    document: Any,
    frame_count: int,
    sample_rate: int,
    path: str = "<memory>",
) -> Timeline:
    if not isinstance(document, dict):
        raise _timeline_invalid("timeline root must be a JSON object")

    target_lufs = _number(
        document.get(
            "target_lufs", document.get("target_loudness_lufs", DEFAULT_TARGET_LUFS)
        ),
        "target_lufs",
    )
    tolerance_lu = _number(
        document.get("tolerance_lu", document.get("loudness_tolerance_lu",
                                                  DEFAULT_TOLERANCE_LU)),
        "tolerance_lu",
    )
    if tolerance_lu <= 0.0:
        raise _timeline_invalid("tolerance_lu must be positive")

    raw_segments = document.get("segments")
    raw_splices = document.get("splice_points_sec")
    if raw_segments is None and raw_splices is None:
        raise _timeline_invalid("timeline needs segments or splice_points_sec")
    if raw_segments is not None and not isinstance(raw_segments, list):
        raise _timeline_invalid("segments must be an array")
    if raw_splices is not None and not isinstance(raw_splices, list):
        raise _timeline_invalid("splice_points_sec must be an array")

    splice_samples: List[int] = []
    splice_crossfade_ms: List[Optional[float]] = []
    if raw_splices is not None:
        previous = -1
        for point_index, point in enumerate(raw_splices):
            if isinstance(point, dict):
                sample = _resolve_point(
                    point,
                    ("time_sec", "at_sec", "splice_sec"),
                    ("sample", "splice_sample"),
                    sample_rate,
                    "splice_points_sec[{}]".format(point_index),
                )
                crossfade = point.get("crossfade_ms")
                if crossfade is not None:
                    crossfade = _number(
                        crossfade,
                        "splice_points_sec[{}].crossfade_ms".format(point_index),
                    )
                    if crossfade < 0.0:
                        raise _timeline_invalid("crossfade_ms cannot be negative")
            else:
                sample = _sample_from_seconds(
                    point,
                    "splice_points_sec[{}]".format(point_index),
                    sample_rate,
                )
                crossfade = None
            if sample <= previous:
                raise _timeline_invalid("splice points must be strictly increasing")
            if sample < 0 or sample > frame_count:
                raise _timeline_range(
                    "splice point {} lies outside the input audio".format(
                        point_index
                    ),
                    splice=point_index,
                )
            splice_samples.append(sample)
            splice_crossfade_ms.append(crossfade)
            previous = sample

    segments_data: List[Dict[str, Any]]
    if raw_segments is None:
        segments_data = [{} for _ in range(len(splice_samples) + 1)]
    else:
        if not raw_segments:
            raise _timeline_invalid("segments must not be empty")
        segments_data = []
        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                raise _timeline_invalid(
                    "segments[{}] must be an object".format(index)
                )
            segments_data.append(item)
    if raw_splices is not None and len(segments_data) != len(splice_samples) + 1:
        raise _timeline_invalid(
            "segments count must equal splice point count + 1"
        )

    default_crossfade = _number(
        document.get("crossfade_ms", DEFAULT_CROSSFADE_MS), "crossfade_ms"
    )
    if default_crossfade < 0.0:
        raise _timeline_invalid("crossfade_ms cannot be negative")
    raw_crossfades = document.get("crossfades_ms")
    crossfade_values: List[float] = []
    if raw_crossfades is not None:
        if not isinstance(raw_crossfades, list):
            raise _timeline_invalid("crossfades_ms must be an array")
        if len(raw_crossfades) > len(segments_data) - 1:
            raise _timeline_invalid("too many crossfade values")
        for item_index, value in enumerate(raw_crossfades):
            value = _number(value, "crossfades_ms[{}]".format(item_index))
            if value < 0.0:
                raise _timeline_invalid("crossfade duration cannot be negative")
            crossfade_values.append(value)
        while len(crossfade_values) < max(0, len(segments_data) - 1):
            crossfade_values.append(default_crossfade)
    else:
        crossfade_values = [default_crossfade] * max(0, len(segments_data) - 1)
    for index, override in enumerate(splice_crossfade_ms):
        if override is not None:
            crossfade_values[index] = override

    specs: List[SegmentSpec] = []
    previous_end = 0
    for index, item in enumerate(segments_data):
        has_start = any(
            key in item for key in ("start_sec", "start_sample", "source_start_sec")
        )
        has_end = any(
            key in item for key in ("end_sec", "end_sample", "source_end_sec")
        )
        if raw_splices is not None and not has_start and not has_end:
            start_sample = 0 if index == 0 else splice_samples[index - 1]
            end_sample = frame_count if index == len(segments_data) - 1 else splice_samples[index]
        else:
            start_sample = _resolve_point(
                item,
                ("start_sec", "source_start_sec", "at_sec"),
                ("start_sample", "source_start_sample"),
                sample_rate,
                "segments[{}].start".format(index),
            )
            end_sample = _resolve_point(
                item,
                ("end_sec", "source_end_sec"),
                ("end_sample", "source_end_sample"),
                sample_rate,
                "segments[{}].end".format(index),
            )
        if start_sample < 0 or end_sample > frame_count:
            raise _timeline_range(
                "segments[{}] range [{}, {}] exceeds input frame count {}".format(
                    index, start_sample, end_sample, frame_count
                ),
                segment=index,
            )
        if start_sample >= end_sample:
            raise _timeline_invalid(
                "segments[{}] start must be before end".format(index)
            )
        if start_sample != previous_end:
            raise _timeline_invalid(
                "segments[{}] must start at frame {} to cover the input without gaps".format(
                    index, previous_end
                ),
                segment=index,
            )
        if raw_splices is not None:
            expected_start = 0 if index == 0 else splice_samples[index - 1]
            expected_end = (
                frame_count
                if index == len(segments_data) - 1
                else splice_samples[index]
            )
            if start_sample != expected_start or end_sample != expected_end:
                raise _timeline_invalid(
                    "segments[{}] does not match the declared splice points".format(
                        index
                    ),
                    segment=index,
                )
        previous_end = end_sample

        stretch_min, stretch_max = _parse_range(item, index)
        if "stretch_ratio" in item:
            ratio = _number(item["stretch_ratio"], "segments[{}].stretch_ratio".format(index))
        elif "stretch" in item:
            ratio = _number(item["stretch"], "segments[{}].stretch".format(index))
        else:
            ratio = 1.0
        if ratio < stretch_min or ratio > stretch_max:
            raise RepairError(
                STRETCH_UNREACHABLE,
                "segments[{}] stretch ratio {:.6f} is outside [{}, {}]".format(
                    index, ratio, stretch_min, stretch_max
                ),
                {
                    "segment": index,
                    "required_ratio": ratio,
                    "allowed": [stretch_min, stretch_max],
                },
            )
        segment_frames = end_sample - start_sample
        output_length = int(round(segment_frames * ratio))
        if output_length <= 0:
            raise _timeline_range("segments[{}] output is empty".format(index))
        source_anchors, expected_anchors = _parse_anchors(
            item, index, segment_frames, output_length, sample_rate
        )
        segment_id = str(item.get("id", "segment-{}".format(index)))
        segment_target = _number(
            item.get("target_lufs", target_lufs),
            "segments[{}].target_lufs".format(index),
        )
        crossfade_out = item.get("crossfade_out_ms")
        if crossfade_out is not None:
            crossfade_out = _number(
                crossfade_out, "segments[{}].crossfade_out_ms".format(index)
            )
            if crossfade_out < 0.0:
                raise _timeline_invalid("crossfade_out_ms cannot be negative")
        spec = SegmentSpec(
            index=index,
            segment_id=segment_id,
            start_sample=start_sample,
            end_sample=end_sample,
            target_lufs=segment_target,
            stretch_ratio=ratio,
            stretch_min=stretch_min,
            stretch_max=stretch_max,
            source_anchors=source_anchors,
            expected_anchors=expected_anchors,
            output_length=output_length,
            crossfade_out_ms=crossfade_out,
        )
        _validate_anchor_ratios(spec)
        specs.append(spec)

    if previous_end != frame_count:
        raise _timeline_range(
            "timeline ends at frame {} but input has {} frames".format(
                previous_end, frame_count
            )
        )

    for splice_index, spec in enumerate(specs[:-1]):
        if spec.crossfade_out_ms is not None:
            crossfade_values[splice_index] = spec.crossfade_out_ms
    for splice_index, duration_ms in enumerate(crossfade_values):
        samples = int(round(duration_ms * sample_rate / 1000.0))
        if samples >= min(specs[splice_index].output_length,
                          specs[splice_index + 1].output_length):
            raise _timeline_range(
                "crossfade at splice {} is longer than an adjacent segment".format(
                    splice_index
                ),
                splice=splice_index,
            )

    return Timeline(
        path=path,
        target_lufs=target_lufs,
        tolerance_lu=tolerance_lu,
        crossfade_ms=tuple(crossfade_values),
        segments=tuple(specs),
    )


def load_timeline(path: str, frame_count: int, sample_rate: int) -> Timeline:
    if not os.path.isfile(path):
        raise RepairError(INPUT_MISSING, "timeline file not found: {}".format(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RepairError(
            TIMELINE_INVALID,
            "invalid timeline JSON at line {} column {}: {}".format(
                exc.lineno, exc.colno, exc.msg
            ),
        )
    except UnicodeDecodeError as exc:
        raise RepairError(
            TIMELINE_INVALID, "timeline is not valid UTF-8: {}".format(exc)
        )
    except OSError as exc:
        raise RepairError(INPUT_MISSING, "cannot read timeline: {}".format(exc))
    return parse_timeline_document(document, frame_count, sample_rate, path)

