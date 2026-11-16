"""Beat-to-sample timeline construction and validation.

The chart describes tempo as a sorted list of ``(beat, bpm)`` anchor points.
Time is piecewise linear in beat position: between two anchors every beat
lasts ``60 / bpm`` seconds.  A single :class:`TimeMap` instance therefore
answers all beat/second/frame conversions for a build and is the authority
for section durations and synchronization markers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .errors import BuildError, ExitCode


def _load_json(chart_path: str) -> dict:
    try:
        with open(chart_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise BuildError(ExitCode.MISSING_SOURCE,
                         f"chart file not found: {chart_path}",
                         {"path": chart_path}) from exc
    except json.JSONDecodeError as exc:
        raise BuildError(
            ExitCode.CHART_SYNTAX,
            f"chart is not valid JSON: {exc.msg}",
            {"path": chart_path, "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "chart root must be a JSON object",
                         {"path": chart_path})
    return payload


@dataclass(frozen=True)
class BpmPoint:
    beat: float
    bpm: float


@dataclass(frozen=True)
class TimeSignature:
    beat: float
    numerator: int
    denominator: int


class TimeMap:
    """Piecewise-linear mapping beat position -> seconds / sample frames."""

    def __init__(self, bpm_points: list[BpmPoint], sample_rate: int,
                 end_beat: float | None = None):
        if not bpm_points:
            raise BuildError(ExitCode.MAPPING_CONFLICT,
                             "bpm_points must contain at least one entry")
        beats = np.asarray([point.beat for point in bpm_points], dtype=np.float64)
        bpms = np.asarray([point.bpm for point in bpm_points], dtype=np.float64)

        if np.any(bpms <= 0.0) or not np.all(np.isfinite(bpms)):
            bad = int(np.argmax(bpms <= 0.0)) if np.any(bpms <= 0.0) else -1
            raise BuildError(
                ExitCode.MAPPING_CONFLICT,
                "bpm values must be finite and strictly positive",
                {"offending_index": bad},
            )
        if np.any(np.diff(beats) <= 0.0):
            raise BuildError(
                ExitCode.MAPPING_CONFLICT,
                "bpm anchor beats must be strictly increasing",
                {"beats": beats.tolist()},
            )

        # Cumulative seconds at every anchor; segments start at beat[0].
        segment_beats = np.diff(beats)
        segment_seconds = segment_beats * (60.0 / bpms[:-1])
        anchor_seconds = np.concatenate(([0.0], np.cumsum(segment_seconds)))

        self.sample_rate = int(sample_rate)
        self.beats = beats
        self.bpms = bpms
        self.anchor_seconds = anchor_seconds
        if end_beat is not None:
            if not np.isfinite(end_beat) or end_beat <= float(beats[-1]):
                raise BuildError(
                    ExitCode.MAPPING_CONFLICT,
                    "end_beat must be finite and beyond the last BPM anchor",
                    {"end_beat": float(end_beat),
                     "last_anchor": float(beats[-1])},
                )
            self._end_beat = float(end_beat)
        else:
            self._end_beat = float(beats[-1])

    @property
    def start_beat(self) -> float:
        return float(self.beats[0])

    @property
    def end_beat(self) -> float:
        return self._end_beat

    def bpm_at(self, beat: float) -> float:
        index = int(np.searchsorted(self.beats, beat, side="right") - 1)
        index = max(0, min(index, len(self.bpms) - 1))
        return float(self.bpms[index])

    def beat_to_seconds(self, beat: float | np.ndarray) -> np.ndarray:
        beats = np.asarray(beat, dtype=np.float64)
        index = np.searchsorted(self.beats, beats, side="right") - 1
        index = np.clip(index, 0, len(self.beats) - 1)
        local_beats = beats - self.beats[index]
        return self.anchor_seconds[index] + local_beats * (60.0 / self.bpms[index])

    def beat_to_frame_exact(self, beat: float | np.ndarray) -> np.ndarray:
        """Unrounded frame position (float64); rounding adds <= 0.5 frame."""
        return self.beat_to_seconds(beat) * self.sample_rate

    def beat_to_frame(self, beat: float | np.ndarray) -> np.ndarray:
        return np.rint(self.beat_to_frame_exact(beat)).astype(np.int64)

    def section_frames(self, start_beat: float, end_beat: float) -> int:
        """Exact frame length of a half-open beat interval (>= 0)."""
        if end_beat < start_beat:
            raise BuildError(
                ExitCode.MAPPING_CONFLICT,
                "section end_beat precedes start_beat",
                {"start_beat": start_beat, "end_beat": end_beat},
            )
        start_frame = self.beat_to_frame_exact(start_beat)
        end_frame = self.beat_to_frame_exact(end_beat)
        return int(round(float(end_frame - start_frame)))

    def validate_switch_accuracy(self) -> dict:
        """Check one-beat neighbourhood around each BPM switch.

        For every anchor after the first, evaluate the frame mapping on a
        dense grid spanning one beat on each side.  Because the mapping is
        linear per segment and continuous at anchors, the deviation from the
        analytic float64 reference after integer rounding must never exceed
        one sample frame.
        """
        worst = 0.0
        for anchor_beat in self.beats[1:]:
            grid = np.linspace(anchor_beat - 1.0, anchor_beat + 1.0, 4001)
            exact = self.beat_to_frame_exact(grid)
            rounded = np.rint(exact)
            worst = max(worst, float(np.max(np.abs(rounded - exact))))
        if worst > 1.0:
            raise BuildError(
                ExitCode.MAPPING_CONFLICT,
                "beat/frame mapping error exceeds one sample frame near a BPM switch",
                {"worst_error_frames": worst},
            )
        return {"worst_rounding_error_frames": worst, "checked_switches": len(self.beats) - 1}


def _require_number(payload: dict, key: str, context: str) -> float:
    if key not in payload or not isinstance(payload[key], (int, float)) \
            or isinstance(payload[key], bool) or not np.isfinite(payload[key]):
        raise BuildError(ExitCode.CHART_SYNTAX,
                         f"{context}: numeric field '{key}' is missing or invalid",
                         {"field": key})
    return float(payload[key])


def parse_bpm_points(payload: dict) -> list[BpmPoint]:
    raw = payload.get("bpm_points")
    if not isinstance(raw, list) or not raw:
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "bpm_points must be a non-empty list")
    points: list[BpmPoint] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"bpm_points[{index}] must be an object")
        beat = _require_number(item, "beat", f"bpm_points[{index}]")
        bpm = _require_number(item, "bpm", f"bpm_points[{index}]")
        if bpm <= 0.0:
            raise BuildError(ExitCode.MAPPING_CONFLICT,
                             "bpm must be strictly positive",
                             {"index": index, "bpm": bpm})
        points.append(BpmPoint(beat=beat, bpm=bpm))
    if any(points[i].beat >= points[i + 1].beat for i in range(len(points) - 1)):
        raise BuildError(ExitCode.MAPPING_CONFLICT,
                         "bpm anchor beats must be strictly increasing",
                         {"beats": [point.beat for point in points]})
    return points


def parse_time_signatures(payload: dict) -> list[TimeSignature]:
    raw = payload.get("time_signatures")
    if raw is None:
        return [TimeSignature(0.0, 4, 4)]
    if not isinstance(raw, list) or not raw:
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "time_signatures must be a non-empty list when present")
    signatures: list[TimeSignature] = []
    previous_beat = None
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"time_signatures[{index}] must be an object")
        beat = _require_number(item, "beat", f"time_signatures[{index}]")
        numerator = item.get("numerator")
        denominator = item.get("denominator")
        if not isinstance(numerator, int) or numerator <= 0:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"time_signatures[{index}].numerator must be a positive integer")
        if not isinstance(denominator, int) or denominator <= 0 or (denominator & (denominator - 1)):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"time_signatures[{index}].denominator must be a positive power of two")
        if previous_beat is not None and beat <= previous_beat:
            raise BuildError(ExitCode.MAPPING_CONFLICT,
                             "time signature beats must be strictly increasing",
                             {"index": index})
        previous_beat = beat
        signatures.append(TimeSignature(beat, numerator, denominator))
    return signatures


@dataclass
class Section:
    section_id: str
    start_beat: float
    end_beat: float
    native_bpm: float

    @property
    def beat_length(self) -> float:
        return self.end_beat - self.start_beat


def parse_sections(payload: dict, time_map: TimeMap) -> list[Section]:
    raw = payload.get("sections")
    if not isinstance(raw, list) or not raw:
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "sections must be a non-empty list")
    sections: list[Section] = []
    cursor = time_map.start_beat
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"sections[{index}] must be an object")
        section_id = item.get("id")
        if not isinstance(section_id, str) or not section_id:
            raise BuildError(ExitCode.CHART_SYNTAX,
                             f"sections[{index}].id must be a non-empty string")
        start_beat = _require_number(item, "start_beat", f"sections[{index}]")
        end_beat = _require_number(item, "end_beat", f"sections[{index}]")
        native_bpm = _require_number(item, "native_bpm", f"sections[{index}]")
        if native_bpm <= 0.0:
            raise BuildError(ExitCode.MAPPING_CONFLICT,
                             "section native_bpm must be positive",
                             {"section": section_id})
        if end_beat <= start_beat:
            raise BuildError(ExitCode.MAPPING_CONFLICT,
                             "section end_beat must exceed start_beat",
                             {"section": section_id})
        if abs(start_beat - cursor) > 1e-9:
            raise BuildError(
                ExitCode.MAPPING_CONFLICT,
                "sections must partition the timeline without gaps or overlaps",
                {"section": section_id, "expected_start_beat": cursor,
                 "actual_start_beat": start_beat},
            )
        sections.append(Section(section_id, start_beat, end_beat, native_bpm))
        cursor = end_beat
    if abs(cursor - time_map.end_beat) > 1e-9:
        raise BuildError(
            ExitCode.MAPPING_CONFLICT,
            "sections must cover the full BPM timeline",
            {"expected_end_beat": time_map.end_beat, "actual_end_beat": cursor},
        )
    return sections


def load_chart(chart_path: str) -> tuple[dict, TimeMap, list[TimeSignature], list[Section]]:
    """Load and structurally validate a chart JSON file."""
    payload = _load_json(chart_path)
    sample_rate = payload.get("internal_sample_rate", 48000)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "internal_sample_rate must be a positive integer")
    bpm_points = parse_bpm_points(payload)
    end_beat = payload.get("end_beat")
    if end_beat is not None and (not isinstance(end_beat, (int, float))
                                 or isinstance(end_beat, bool)):
        raise BuildError(ExitCode.CHART_SYNTAX,
                         "end_beat must be a number when present")
    time_map = TimeMap(
        bpm_points, sample_rate,
        end_beat=float(end_beat) if end_beat is not None else None)
    time_map.validate_switch_accuracy()
    signatures = parse_time_signatures(payload)
    sections = parse_sections(payload, time_map)
    return payload, time_map, signatures, sections
