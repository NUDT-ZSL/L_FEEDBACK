"""Timeline parsing and validation.

The JSON document has the shape::

    {
      "video_duration_ms": 120000,          // optional, may also be probed
      "default_ratio_range": [0.5, 2.0],   // optional
      "segments": [
        {
          "id": "line_001",
          "start_ms": 1000,
          "end_ms": 3000,
          "audio": "line_001.wav",
          "ratio_range": [0.8, 1.25]       // optional
        },
        {
          "id": "group_a",
          "segments": [ ... ]              // nested group, arbitrary depth
        }
      ]
    }

All times are normalised to integer milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import DEFAULT_MAX_RATIO, DEFAULT_MIN_RATIO
from .model import Segment


class TimelineFatalError(Exception):
    """Unrecoverable timeline problem (bad JSON / malformed document)."""


def load_timeline(path: Path) -> Tuple[List[Segment], Dict[str, Any]]:
    """Parse a timeline JSON file.

    Returns the flattened segment list and document-level metadata.
    Raises :class:`TimelineFatalError` when the document cannot be parsed or
    structurally interpreted.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TimelineFatalError(f"cannot read timeline file: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TimelineFatalError(
            f"timeline JSON parse failed at line {exc.lineno} col {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(document, dict):
        raise TimelineFatalError("timeline root must be a JSON object")

    default_range = _read_ratio_range(
        document.get("default_ratio_range"),
        DEFAULT_MIN_RATIO,
        DEFAULT_MAX_RATIO,
        context="default_ratio_range",
    )

    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise TimelineFatalError("timeline must contain a non-empty 'segments' list")

    segments: List[Segment] = []
    _flatten(raw_segments, segments, group=None,
             default_min=default_range[0], default_max=default_range[1])
    if not segments:
        raise TimelineFatalError("no usable segment objects found in timeline")

    metadata = {
        "video_duration_ms": _read_optional_int(document.get("video_duration_ms")),
    }
    return segments, metadata


def validate_segments(segments: List[Segment],
                      video_duration_ms: Optional[int]) -> "ValidationReport":
    """Run semantic validation and return the list of *valid* segments.

    Invalid segments (start >= end, outside the video window, overlapping a
    neighbour) are dropped. Their identifiers are returned through the
    ``invalid`` attribute on the exception-free ``ValidationReport`` so the
    caller can log them and keep silence placeholders.
    """
    valid: List[Segment] = []
    invalid: List[Tuple[str, str]] = []

    for segment in segments:
        if segment.start_ms >= segment.end_ms:
            invalid.append((segment.segment_id,
                            f"start_ms ({segment.start_ms}) >= end_ms ({segment.end_ms})"))
            continue
        if video_duration_ms is not None and (
            segment.start_ms < 0 or segment.end_ms > video_duration_ms
        ):
            invalid.append((segment.segment_id,
                            f"segment [{segment.start_ms},{segment.end_ms}] outside "
                            f"video duration {video_duration_ms}ms"))
            continue
        valid.append(segment)

    valid.sort(key=lambda item: (item.start_ms, item.end_ms))

    non_overlapping: List[Segment] = []
    previous: Optional[Segment] = None
    for segment in valid:
        if previous is not None and segment.start_ms < previous.end_ms:
            invalid.append((segment.segment_id,
                            f"overlaps segment {previous.segment_id}: "
                            f"[{segment.start_ms},{segment.end_ms}) vs "
                            f"[{previous.start_ms},{previous.end_ms})"))
            continue
        non_overlapping.append(segment)
        previous = segment

    return ValidationReport(valid=non_overlapping, invalid=invalid)


class ValidationReport:
    """Container describing the outcome of :func:`validate_segments`."""

    def __init__(self, valid: List[Segment],
                 invalid: List[Tuple[str, str]]) -> None:
        self.valid = valid
        self.invalid = invalid


def _flatten(nodes: List[Any], out: List[Segment], *, group: Optional[str],
             default_min: float, default_max: float) -> None:
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TimelineFatalError(
                f"segment entry at index {index} must be an object, got {type(node).__name__}"
            )
        nested = node.get("segments")
        if nested is not None:
            if not isinstance(nested, list):
                raise TimelineFatalError(
                    f"group '{node.get('id', index)}' has a non-list 'segments' field"
                )
            group_name = str(node.get("id", f"group_{index}"))
            child_group = group_name if group is None else f"{group}/{group_name}"
            _flatten(nested, out, group=child_group,
                     default_min=default_min, default_max=default_max)
            continue

        segment_id = str(node.get("id") or f"segment_{index}")
        start_ms = _read_time(node.get("start_ms"), segment_id, "start_ms")
        end_ms = _read_time(node.get("end_ms"), segment_id, "end_ms")
        audio = node.get("audio") or node.get("audio_path")
        if not isinstance(audio, str) or not audio:
            raise TimelineFatalError(f"segment {segment_id} missing string 'audio' field")
        min_ratio, max_ratio = _read_ratio_range(
            node.get("ratio_range"), default_min, default_max,
            context=f"segment {segment_id} ratio_range",
        )
        out.append(Segment(
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_path=audio,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            group=group,
        ))


def _read_time(value: Any, segment_id: str, field_name: str) -> int:
    if isinstance(value, bool):
        raise TimelineFatalError(f"segment {segment_id} {field_name} must be a number")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TimelineFatalError(
        f"segment {segment_id} {field_name} must be an integer millisecond value"
    )


def _read_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _read_ratio_range(value: Any, default_min: float, default_max: float,
                      *, context: str) -> Tuple[float, float]:
    if value is None:
        return default_min, default_max
    if not (isinstance(value, list) and len(value) == 2):
        raise TimelineFatalError(f"{context} must be a [min, max] list")
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise TimelineFatalError(f"{context} must contain numbers") from exc
    if low <= 0 or high <= 0 or low > high:
        raise TimelineFatalError(
            f"{context} invalid: require 0 < min <= max, got [{low}, {high}]"
        )
    return low, high
