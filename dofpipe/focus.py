from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigurationError, FocusTrackError


@dataclass(frozen=True)
class Keyframe:
    time_s: float
    focus_distance_m: float


@dataclass(frozen=True)
class TrackIssue:
    frame_index: int
    reason: str


@dataclass(frozen=True)
class TrackWarning:
    code: str
    frame_index: int
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "frame_index": int(self.frame_index),
            "message": self.message,
        }


def _parse_keyframe(entry: Any, index: int, fps: float, time_unit: str) -> Keyframe:
    if not isinstance(entry, dict):
        raise ConfigurationError(f"focus keyframe {index} must be an object")
    if "frame" in entry:
        raw_time = entry["frame"]
        unit = "frames"
    elif "time" in entry:
        raw_time = entry["time"]
        unit = time_unit
    elif "timestamp" in entry:
        raw_time = entry["timestamp"]
        unit = time_unit
    else:
        raise ConfigurationError(
            f"focus keyframe {index} must contain time/timestamp/frame"
        )

    focus_value = None
    for key in ("focus_distance", "focus_distance_m", "focus", "distance"):
        if key in entry:
            focus_value = entry[key]
            break
    if focus_value is None:
        raise ConfigurationError(
            f"focus keyframe {index} must contain focus_distance"
        )
    if isinstance(raw_time, bool) or not isinstance(raw_time, (int, float)):
        raise ConfigurationError(f"focus keyframe {index} time must be numeric")
    if isinstance(focus_value, bool) or not isinstance(focus_value, (int, float)):
        raise ConfigurationError(f"focus keyframe {index} distance must be numeric")

    time = float(raw_time)
    focus = float(focus_value)
    if not math.isfinite(time):
        raise ConfigurationError(f"focus keyframe {index} time must be finite")
    if not math.isfinite(focus):
        raise ConfigurationError(f"focus keyframe {index} distance must be finite")
    if unit == "frames":
        time /= fps
    elif unit != "seconds":
        raise ConfigurationError(
            "focus track time_unit must be 'seconds' or 'frames'"
        )
    return Keyframe(time_s=time, focus_distance_m=focus)


def load_focus_track(path: Path, fps: float) -> list[Keyframe]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            root = json.load(handle)
    except OSError as exc:
        raise ConfigurationError(f"cannot read focus track {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid focus track JSON {path}: {exc}") from exc

    if isinstance(root, list):
        entries = root
        time_unit = "seconds"
    elif isinstance(root, dict):
        entries = root.get("keyframes")
        time_unit = root.get("time_unit", "seconds")
        if not isinstance(time_unit, str):
            raise ConfigurationError("focus track time_unit must be a string")
    else:
        raise ConfigurationError("focus track must be a list or an object")
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError("focus track must contain a non-empty keyframes list")
    return [
        _parse_keyframe(entry, index, fps, time_unit)
        for index, entry in enumerate(entries)
    ]


def _pchip_slopes(
    times: np.ndarray, values: np.ndarray
) -> np.ndarray:
    count = len(times)
    if count == 1:
        return np.zeros(1, dtype=np.float64)
    h = np.diff(times)
    delta = np.diff(values) / h
    slopes = np.zeros(count, dtype=np.float64)
    if count == 2:
        slopes[:] = delta[0]
        return slopes

    for index in range(1, count - 1):
        left = delta[index - 1]
        right = delta[index]
        if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
            slopes[index] = 0.0
        else:
            weight_left = 2.0 * h[index] + h[index - 1]
            weight_right = h[index] + 2.0 * h[index - 1]
            slopes[index] = (weight_left + weight_right) / (
                weight_left / left + weight_right / right
            )

    def edge_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(slope) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(slope) > abs(3.0 * d0):
            return 3.0 * d0
        return slope

    slopes[0] = edge_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = edge_slope(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def pchip_interpolate(
    key_times: np.ndarray,
    key_values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    times = np.asarray(key_times, dtype=np.float64)
    values = np.asarray(key_values, dtype=np.float64)
    queries = np.asarray(query_times, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or len(times) == 0:
        raise ValueError("key times/values must be non-empty 1-D arrays of equal size")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("key times must be strictly increasing")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("key times and values must be finite")

    slopes = _pchip_slopes(times, values)
    flat_queries = np.atleast_1d(queries)
    result = np.empty(flat_queries.shape, dtype=np.float64)

    for position, query in enumerate(flat_queries):
        if not math.isfinite(query):
            raise ValueError("query times must be finite")
        if query <= times[0]:
            result[position] = values[0]
            continue
        if query >= times[-1]:
            result[position] = values[-1]
            continue
        segment = int(np.searchsorted(times, query, side="right") - 1)
        if query == times[segment]:
            result[position] = values[segment]
            continue
        h = times[segment + 1] - times[segment]
        t = (query - times[segment]) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        value = (
            h00 * values[segment]
            + h10 * h * slopes[segment]
            + h01 * values[segment + 1]
            + h11 * h * slopes[segment + 1]
        )
        lower = min(values[segment], values[segment + 1])
        upper = max(values[segment], values[segment + 1])
        result[position] = min(max(value, lower), upper)

    if queries.ndim == 0:
        return np.asarray(result[0])
    return result.reshape(queries.shape)


class FocusTrack:
    """PCHIP focus track with deferred, frame-indexed metadata validation."""

    def __init__(
        self,
        keyframes: list[Keyframe],
        fps: float,
        total_frames: int,
        frame_start: int = 0,
    ):
        if not keyframes:
            raise ConfigurationError("focus track must not be empty")
        if fps <= 0.0 or not math.isfinite(fps):
            raise ConfigurationError("frame rate must be positive and finite")
        if total_frames <= 0:
            raise ConfigurationError("frame count must be positive")
        if frame_start < 0:
            raise ConfigurationError("frame start must be >= 0")
        self.keyframes = list(keyframes)
        self.fps = float(fps)
        self.total_frames = int(total_frames)
        self.frame_start = int(frame_start)
        self.end_frame = self.frame_start + self.total_frames - 1
        self.start_time = self.frame_start / self.fps
        self.end_time = self.end_frame / self.fps
        self.issues: list[TrackIssue] = []
        self.warnings: list[TrackWarning] = []
        self._active = self._validate_and_select_active()
        self._active_times = np.asarray(
            [frame.time_s for frame in self._active], dtype=np.float64
        )
        self._active_values = np.asarray(
            [frame.focus_distance_m for frame in self._active], dtype=np.float64
        )
        self._record_coverage_warnings()

    def _frame_for_time(self, time_s: float) -> int:
        if not math.isfinite(time_s):
            return self.frame_start
        raw = math.floor(time_s * self.fps + 0.5)
        return min(
            self.frame_start + self.total_frames,
            max(self.frame_start, raw),
        )

    def _add_issue(self, frame_index: int, reason: str) -> None:
        item = TrackIssue(
            frame_index=max(
                self.frame_start,
                min(self.frame_start + self.total_frames, frame_index),
            ),
            reason=reason,
        )
        if all(
            not (existing.frame_index == item.frame_index and existing.reason == item.reason)
            for existing in self.issues
        ):
            self.issues.append(item)

    def _validate_and_select_active(self) -> list[Keyframe]:
        active: list[Keyframe] = []
        order_broken = False
        for index, frame in enumerate(self.keyframes):
            if not math.isfinite(frame.focus_distance_m) or frame.focus_distance_m <= 0.0:
                self._add_issue(
                    self._frame_for_time(frame.time_s),
                    f"keyframe {index} focus distance must be finite and > 0",
                )
            if frame.time_s < -1e-9:
                self._add_issue(
                    self.frame_start,
                    f"keyframe {index} timestamp {frame.time_s:.9g} s is before "
                    f"sequence range [{self.start_time:.9g}, "
                    f"{self.end_time:.9g}] s",
                )
            elif (
                frame.time_s < self.start_time - 1e-9
                or frame.time_s > self.end_time + 1e-9
            ):
                self._add_issue(
                    self.frame_start + self.total_frames
                    if frame.time_s > self.end_time
                    else self.frame_start,
                    f"keyframe {index} timestamp {frame.time_s:.9g} s is outside "
                    f"sequence range [{self.start_time:.9g}, "
                    f"{self.end_time:.9g}] s",
                )

            if active and frame.time_s <= active[-1].time_s:
                if not order_broken:
                    self._add_issue(
                        self._frame_for_time(frame.time_s),
                        f"keyframe {index} timestamp {frame.time_s:.9g} s is not "
                        f"strictly after keyframe {index - 1} timestamp "
                        f"{active[-1].time_s:.9g} s",
                    )
                order_broken = True
            if not order_broken:
                active.append(frame)

        if not active:
            active = [self.keyframes[0]]
        self.issues.sort(key=lambda issue: issue.frame_index)
        return active

    def _record_coverage_warnings(self) -> None:
        first = self._active[0].time_s
        last = self._active[-1].time_s
        if first > self.start_time + 1e-9:
            first_frame = min(
                self.frame_start + self.total_frames,
                max(
                    self.frame_start,
                    math.ceil(first * self.fps - 1e-9),
                ),
            )
            if first_frame > self.frame_start:
                self.warnings.append(
                    TrackWarning(
                        code="track_starts_after_sequence",
                        frame_index=self.frame_start,
                        message=(
                            f"focus track starts at {first:.9g} s after frame "
                            f"{self.frame_start}; frames {self.frame_start}.."
                            f"{first_frame - 1} use the first keyframe value"
                        ),
                    )
                )
        if last < self.end_time - 1e-9:
            first_after = max(
                self.frame_start,
                math.floor(last * self.fps + 1e-9) + 1,
            )
            if first_after <= self.end_frame:
                self.warnings.append(
                    TrackWarning(
                        code="track_ends_before_sequence",
                        frame_index=first_after,
                        message=(
                            f"focus track ends at {last:.9g} s before the final frame "
                            f"at {self.end_time:.9g} s; frames "
                            f"{first_after}..{self.end_frame} use the last "
                            "keyframe value"
                        ),
                    )
                )

    def _raise_issue_for_frame(self, frame_index: int) -> None:
        for issue in self.issues:
            if frame_index >= issue.frame_index:
                raise FocusTrackError(issue.frame_index, issue.reason)

    def sample(self, frame_index: int) -> float:
        if frame_index < self.frame_start or frame_index > self.end_frame:
            raise FocusTrackError(
                frame_index,
                f"frame index outside [{self.frame_start}, {self.end_frame}]",
            )
        self._raise_issue_for_frame(frame_index)
        time_s = frame_index / self.fps
        value = pchip_interpolate(
            self._active_times, self._active_values, np.asarray(time_s)
        )
        return float(value)

    def validate_after_frames(self) -> None:
        self._raise_issue_for_frame(self.frame_start + self.total_frames)

    def warning_dicts(self) -> list[dict[str, object]]:
        return [warning.as_dict() for warning in self.warnings]
