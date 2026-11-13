"""Gain-automation validation, interpolation, and step smoothing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .util import seconds_to_samples


class AutomationError(Exception):
    def __init__(self, code: str, node_index: int, detail: str):
        super().__init__(detail)
        self.code = code
        self.node_index = node_index
        self.detail = detail


def validate_automation(track_name: str, nodes: Any) -> list[dict[str, float]]:
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise AutomationError("AUTOMATION_EMPTY", 0, "automation must contain at least one node")
    normalized: list[dict[str, float]] = []
    previous_time: float | None = None
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise AutomationError(
                "AUTOMATION_NODE_INVALID", index, "node must contain numeric time and gain_db"
            )
        time_key = next(
            (key for key in ("time", "time_seconds", "t") if key in node),
            None,
        )
        gain_key = next(
            (key for key in ("gain_db", "gain", "gain_db_value") if key in node),
            None,
        )
        if time_key is None or gain_key is None:
            raise AutomationError(
                "AUTOMATION_NODE_INVALID", index, "node must contain numeric time and gain_db"
            )
        try:
            time_value = float(node[time_key])
            gain_value = float(node[gain_key])
        except (TypeError, ValueError) as exc:
            raise AutomationError("AUTOMATION_NODE_INVALID", index, "node time/gain is not numeric") from exc
        if not math.isfinite(time_value) or not math.isfinite(gain_value):
            raise AutomationError("AUTOMATION_NODE_INVALID", index, "node time/gain must be finite")
        if time_value < 0.0:
            raise AutomationError("AUTOMATION_TIME_NEGATIVE", index, "node time must be non-negative")
        if previous_time is not None and time_value < previous_time:
            raise AutomationError(
                "AUTOMATION_TIME_ORDER", index, "node times must be non-decreasing"
            )
        if gain_value < -96.0 or gain_value > 12.0:
            raise AutomationError(
                "GAIN_OUT_OF_RANGE",
                index,
                f"gain_db {gain_value:g} is outside [-96, +12] dB",
            )
        normalized.append({"time": time_value, "gain_db": gain_value})
        previous_time = time_value
    return normalized


def _raised_cosine(position: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * position)


def build_gain_curve(
    nodes: list[dict[str, float]],
    sample_rate: int,
    frame_count: int,
    smooth_window_ms: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build sample-aligned gain in dB and smooth configurable step edges."""
    times = np.asarray([node["time"] for node in nodes], dtype=np.float64)
    gains = np.asarray([node["gain_db"] for node in nodes], dtype=np.float64)
    frame_times = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
    node_samples = np.asarray(
        [seconds_to_samples(float(node["time"]), sample_rate) for node in nodes],
        dtype=np.float64,
    )
    frame_indices = np.arange(frame_count, dtype=np.float64)
    base = np.interp(
        frame_indices,
        node_samples,
        gains,
        left=float(gains[0]),
        right=float(gains[-1]),
    )
    curve_start = float(times[0])
    curve_end = float(times[-1])
    curve_span = max(0.0, curve_end - curve_start)
    configured_window_seconds = max(0.0, float(smooth_window_ms) / 1000.0)
    clamp_warnings: list[str] = []
    if configured_window_seconds > curve_span and curve_span > 0.0:
        clamp_warnings.append("SMOOTH_WINDOW_CLAMPED")
    effective_window = min(configured_window_seconds, curve_span) if curve_span > 0.0 else 0.0
    steps: list[dict[str, Any]] = []
    for index in range(len(nodes) - 1):
        interval = float(times[index + 1] - times[index])
        difference = abs(float(gains[index + 1] - gains[index]))
        if interval < 0.005 and difference > 6.0:
            step_time = float(times[index + 1])
            start = step_time - effective_window / 2.0
            end = step_time + effective_window / 2.0
            if curve_span <= 0.0:
                start = end = step_time
            else:
                if start < curve_start:
                    end += curve_start - start
                    start = curve_start
                if end > curve_end:
                    start -= end - curve_end
                    end = curve_end
                start = max(curve_start, start)
                end = min(curve_end, end)
                if configured_window_seconds > curve_span:
                    start, end = curve_start, curve_end
            first_sample = max(0, seconds_to_samples(start, sample_rate))
            last_sample = min(frame_count, seconds_to_samples(end, sample_rate))
            if last_sample > first_sample:
                actual_start = float(first_sample) / float(sample_rate)
                actual_end = float(last_sample) / float(sample_rate)
                span = actual_end - actual_start
                if span > 0.0:
                    mask_times = frame_times[first_sample:last_sample]
                    position = np.clip((mask_times - actual_start) / span, 0.0, 1.0)
                    blend = _raised_cosine(position)
                    base[first_sample:last_sample] = float(gains[index]) + (
                        float(gains[index + 1] - gains[index]) * blend
                    )
            steps.append(
                {
                    "edge_node_index": index + 1,
                    "time": step_time,
                    "from_gain_db": float(gains[index]),
                    "to_gain_db": float(gains[index + 1]),
                    "window_ms": float((end - start) * 1000.0),
                    "start_sample": first_sample,
                    "end_sample": last_sample,
                }
            )
    gain_linear = np.power(10.0, base / 20.0).astype(np.float64)
    stats = {
        "smooth_steps": len(steps),
        "smooth_window_requested_ms": float(smooth_window_ms),
        "smooth_window_effective_ms": float(effective_window * 1000.0),
        "smooth_steps_detail": steps,
        "smooth_warnings": clamp_warnings,
        "min_gain_db": float(np.min(base)) if base.size else 0.0,
        "max_gain_db": float(np.max(base)) if base.size else 0.0,
        "mean_gain_db": float(np.mean(base)) if base.size else 0.0,
        "rms_gain_db": float(np.sqrt(np.mean(base * base, dtype=np.float64))) if base.size else 0.0,
        "time_to_sample_method": "round-half-to-even per automation node sample position",
    }
    return gain_linear, stats
