"""Deterministic JSON report generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .channels import channel_weights
from .loudness import (
    LoudnessBlock,
    LoudnessMeasurement,
    peak_dbfs,
)


def _round2(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    rounded = round(numeric, 2)
    if rounded == 0.0:
        return 0.0
    return rounded


def _measurement_summary(
    measurement: LoudnessMeasurement, audio: np.ndarray
) -> Dict[str, Any]:
    block_values = [
        block.loudness_lufs
        for block in measurement.blocks
        if block.loudness_lufs is not None
        and math.isfinite(float(block.loudness_lufs))
    ]
    return {
        "integrated_lufs": _round2(measurement.integrated_lufs),
        "relative_gate_lufs": _round2(measurement.relative_gate_lufs),
        "block_min_lufs": _round2(min(block_values) if block_values else None),
        "block_max_lufs": _round2(max(block_values) if block_values else None),
        "block_count": len(measurement.blocks),
        "gated_block_count": measurement.gated_block_count,
        "sample_peak_dbfs": _round2(peak_dbfs(audio)),
    }


def _block_dict(block: LoudnessBlock) -> Dict[str, Any]:
    return {
        "start_seconds": _round2(block.start_seconds),
        "end_seconds": _round2(block.end_seconds),
        "duration_seconds": _round2(block.duration_seconds),
        "loudness_lufs": _round2(block.loudness_lufs),
        "passes_absolute_gate": block.passes_absolute_gate,
        "passes_relative_gate": block.passes_relative_gate,
        "is_partial": block.is_partial,
    }


def _segment_measurement(
    audio: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
) -> LoudnessMeasurement:
    from .loudness import measure_loudness

    return measure_loudness(
        audio[start:end],
        sample_rate,
        allow_partial=True,
    )


def _segment_dict(
    index: int,
    start: int,
    end: int,
    total_frames: int,
    sample_rate: int,
    segment_seconds: float,
    before_audio: np.ndarray,
    after_audio: np.ndarray,
) -> Dict[str, Any]:
    before = _segment_measurement(before_audio, sample_rate, start, end)
    after = _segment_measurement(after_audio, sample_rate, start, end)
    duration = (end - start) / float(sample_rate)
    before_lufs = (
        before.integrated_lufs
        if math.isfinite(float(before.integrated_lufs))
        else None
    )
    after_lufs = (
        after.integrated_lufs if math.isfinite(float(after.integrated_lufs)) else None
    )
    delta = (
        float(after_lufs) - float(before_lufs)
        if before_lufs is not None and after_lufs is not None
        else None
    )
    return {
        "index": index,
        "start_seconds": _round2(start / float(sample_rate)),
        "end_seconds": _round2(end / float(sample_rate)),
        "duration_seconds": _round2(duration),
        "before_lufs": _round2(before_lufs),
        "after_lufs": _round2(after_lufs),
        "delta_lufs": _round2(delta),
        "before_gated_block_count": before.gated_block_count,
        "after_gated_block_count": after.gated_block_count,
        "is_partial_segment": end == total_frames
        and duration < segment_seconds - 1.0e-12,
        "window_duration_seconds": _round2(min(duration, 0.4)),
        "is_partial_window": duration < 0.4 - 1.0e-12,
    }


def build_report(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    format_name: str,
    subtype: str,
    channel_count: int,
    target_lufs: float,
    segment_seconds: float,
    before_audio: np.ndarray,
    after_audio: np.ndarray,
    before_measurement: LoudnessMeasurement,
    after_measurement: LoudnessMeasurement,
    requested_gain_db: float,
    applied_base_gain_db: float,
    limiter_enabled: bool,
    limiter_applied: bool,
    max_limiting_db: float,
    limited_sample_count: int,
    maximum_overshoot: float,
    limiter_iterations: int = 0,
) -> Dict[str, Any]:
    """Build the JSON-serializable before/after segment report."""

    total_frames = before_audio.shape[0]
    segment_frames = max(1, int(round(segment_seconds * sample_rate)))
    segments: List[Dict[str, Any]] = []
    index = 0
    for start in range(0, total_frames, segment_frames):
        end = min(total_frames, start + segment_frames)
        segments.append(
            _segment_dict(
                index=index,
                start=start,
                end=end,
                total_frames=total_frames,
                sample_rate=sample_rate,
                segment_seconds=segment_seconds,
                before_audio=before_audio,
                after_audio=after_audio,
            )
        )
        index += 1

    before_summary = _measurement_summary(before_measurement, before_audio)
    after_summary = _measurement_summary(after_measurement, after_audio)
    delta_lufs = None
    if (
        before_summary["integrated_lufs"] is not None
        and after_summary["integrated_lufs"] is not None
    ):
        delta_lufs = (
            float(after_summary["integrated_lufs"])
            - float(before_summary["integrated_lufs"])
        )

    return {
        "schema_version": 1,
        "standard": "ITU-R BS.1770-4 / EBU R128",
        "input": str(input_path),
        "output": str(output_path),
        "audio": {
            "sample_rate": int(sample_rate),
            "channels": int(channel_count),
            "format": str(format_name),
            "subtype": str(subtype),
            "frames": int(total_frames),
            "duration_seconds": _round2(total_frames / float(sample_rate)),
            "channel_weights": [
                float(weight) for weight in channel_weights(channel_count)
            ],
        },
        "parameters": {
            "target_lufs": _round2(target_lufs),
            "segment_seconds": _round2(segment_seconds),
            "limiter_enabled": bool(limiter_enabled),
        },
        "processing": {
            "requested_gain_db": _round2(requested_gain_db),
            "applied_base_gain_db": _round2(applied_base_gain_db),
            "limiter_applied": bool(limiter_applied),
            "max_limiting_db": _round2(max_limiting_db),
            "limited_sample_count": int(limited_sample_count),
            "maximum_overshoot": _round2(maximum_overshoot),
            "limiter_iterations": int(limiter_iterations),
        },
        "overall": {
            "before": before_summary,
            "after": after_summary,
            "delta_lufs": _round2(delta_lufs),
        },
        "segments": segments,
        "integration_blocks": {
            "before": [_block_dict(block) for block in before_measurement.blocks],
            "after": [_block_dict(block) for block in after_measurement.blocks],
        },
    }


def _format_json_value(value: Any, level: int) -> str:
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "null"
        if numeric == 0.0:
            numeric = 0.0
        return "{0:.2f}".format(numeric)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        items = list(value.items())
        if not items:
            return "{}"
        rendered = [
            "{0}{1}: {2}".format(
                child_indent,
                json.dumps(str(key), ensure_ascii=False),
                _format_json_value(item, level + 1),
            )
            for key, item in items
        ]
        return "{\n" + ",\n".join(rendered) + "\n" + indent + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        rendered = [
            "{0}{1}".format(child_indent, _format_json_value(item, level + 1))
            for item in value
        ]
        return "[\n" + ",\n".join(rendered) + "\n" + indent + "]"
    raise TypeError("unsupported report value type: {0}".format(type(value).__name__))


def serialize_report(report: Mapping[str, Any]) -> str:
    """Serialize JSON with every floating-point value formatted to 2 places."""

    return _format_json_value(report, 0) + "\n"

