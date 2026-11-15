"""Cross-channel checks for parameters that must agree in every pass."""

from __future__ import annotations

from typing import Any

import numpy as np

from .graph import CHANNELS, ParsedGraph
from .renderer import ChannelRenderResult


SHARED_PARAMETERS = {
    "roughness": ("roughness", 1),
    "metallic": ("metallic", 1),
    "normal": ("normal", 3),
}


def check_consistency(
    graph: ParsedGraph,
    results: dict[str, ChannelRenderResult],
    *,
    tolerance: float,
) -> dict[str, Any]:
    channels = [channel for channel in CHANNELS if channel in results]
    checks: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    if len(channels) < 2:
        return {
            "shared_parameters": list(SHARED_PARAMETERS),
            "tolerance": tolerance,
            "conclusion": "insufficient_channels",
            "compared_channels": channels,
            "checks": [],
            "differences": [],
        }

    for parameter, (attribute, component_count) in SHARED_PARAMETERS.items():
        channel_values: dict[str, dict[str, float]] = {}
        parameter_differences: list[dict[str, Any]] = []
        max_difference = 0.0
        mean_difference = 0.0
        for channel in channels:
            values = getattr(results[channel], attribute)[..., :component_count]
            finite = values[np.isfinite(values)]
            channel_values[channel] = {
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "min": float(np.min(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
            }
        for left_index, left_channel in enumerate(channels):
            left = getattr(results[left_channel], attribute)[..., :component_count]
            for right_channel in channels[left_index + 1 :]:
                right = getattr(results[right_channel], attribute)[
                    ..., :component_count
                ]
                if left.shape != right.shape:
                    difference = float("inf")
                    exceeded = int(left.size + right.size)
                    location = None
                    finite_delta = None
                else:
                    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                    finite_delta = delta[np.isfinite(delta)]
                    difference = (
                        float(np.max(finite_delta)) if finite_delta.size else 0.0
                    )
                    exceeded = int(np.count_nonzero(delta > tolerance))
                    if exceeded:
                        flat_index = int(np.argmax(delta))
                        y, x, component = np.unravel_index(flat_index, delta.shape)
                        location = {
                            "x": int(x),
                            "y": int(y),
                            "component": int(component),
                            "reference_value": float(left[y, x, component]),
                            "actual_value": float(right[y, x, component]),
                        }
                    else:
                        location = None
                max_difference = max(max_difference, difference)
                if finite_delta is not None and finite_delta.size:
                    mean_difference = max(
                        mean_difference, float(np.mean(finite_delta))
                    )
                if difference > tolerance:
                    node_path = graph.output_path(parameter)
                    item = {
                        "parameter": parameter,
                        "reference_channel": left_channel,
                        "actual_channel": right_channel,
                        "max_abs_diff": difference,
                        "exceeded_pixels": exceeded,
                        "tolerance": tolerance,
                        "node_path": node_path,
                        "first_difference": location,
                    }
                    parameter_differences.append(item)
                    differences.append(item)
        checks.append(
            {
                "parameter": parameter,
                "compared_channels": channels,
                "tolerance": tolerance,
                "max_abs_diff": max_difference,
                "max_mean_abs_diff": mean_difference,
                "passed": not parameter_differences,
                "node_path": graph.output_path(parameter),
                "channel_values": channel_values,
            }
        )

    return {
        "shared_parameters": list(SHARED_PARAMETERS),
        "tolerance": tolerance,
        "conclusion": "fail" if differences else "pass",
        "compared_channels": channels,
        "checks": checks,
        "differences": differences,
    }

