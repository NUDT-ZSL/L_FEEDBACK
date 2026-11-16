from __future__ import annotations

import numpy as np
import pytest

from material_validator.compare import analyze_gbuffer, compare_images
from material_validator.manifest import ComparisonSettings, RenderSettings


def _settings() -> tuple[ComparisonSettings, RenderSettings]:
    comparison = ComparisonSettings(absolute=0.01, relative=0.0, channels=["R", "G", "B"])
    render = RenderSettings(
        width=2,
        height=2,
        background=[0.0, 0.0, 0.0, 1.0],
        normal_fallback=[0.0, 0.0, 1.0],
        normal_tolerance=1.0e-4,
    )
    return comparison, render


def test_threshold_classification_and_max_location() -> None:
    comparison, render = _settings()
    deferred = np.zeros((2, 2, 4), dtype=np.float32)
    forward = np.zeros((2, 2, 4), dtype=np.float32)
    normal = np.zeros((2, 2, 4), dtype=np.float32)
    normal[:, :, 2] = 1.0
    normal[:, :, 3] = 1.0
    params = np.zeros((2, 2, 4), dtype=np.float32)
    params[:, :, 1] = 0.5
    blend = np.zeros((2, 2, 1), dtype=np.float32)

    deferred[0, 0, 0] = 0.8
    params[0, 0, 2] = 1.0
    deferred[0, 1, 1] = 0.4
    normal[0, 1, 3] = 0.0
    deferred[1, 0, 2] = 0.2
    blend[1, 0, 0] = 1.0
    deferred[1, 1, 0] = 0.05

    result = compare_images(deferred, forward, normal, params, blend, comparison, render)

    assert result["pass"] is False
    assert result["pixels_over_threshold"] == 4
    assert result["max_abs_error"] == pytest.approx(0.8, abs=1.0e-6)
    assert result["max_channel"] == "R"
    assert result["max_location"] == {"x": 0, "y": 0}
    assert result["categories"]["material_parameter_range"]["pixels"] == 1
    assert result["categories"]["normal_encoding_precision"]["pixels"] == 1
    assert result["categories"]["floating_blend_order"]["pixels"] == 1
    assert result["categories"]["unclassified"]["pixels"] == 1


def test_gbuffer_normal_fallback_and_range_diagnostics() -> None:
    _, render = _settings()
    position = np.zeros((2, 2, 4), dtype=np.float32)
    normal = np.zeros((2, 2, 4), dtype=np.float32)
    normal[:, :, 2] = 1.0
    normal[:, :, 3] = 1.0
    normal[0, 0, :3] = (0.0, 0.0, 1.0)
    normal[0, 0, 3] = 0.0
    normal[0, 1, :3] = (1.0, 2.0, 3.0)
    albedo = np.zeros((2, 2, 4), dtype=np.float32)
    albedo[:, :, 3] = 1.0
    params = np.zeros((2, 2, 4), dtype=np.float32)
    params[:, :, 0] = 1.0
    params[:, :, 2] = 0.0
    params[1, 0, 0] = 1.0
    params[1, 0, 2] = 1.0
    params[1, 1, 2] = 1.0
    params[1, 1, 3] = 1.0

    stats = analyze_gbuffer(position, normal, albedo, params, render)

    assert stats["coverage_pixels"] == 4
    assert stats["normal_fallback_pixels"] == 1
    assert stats["normal_non_unit_pixels"] >= 1
    assert stats["metallic_out_of_range_pixels"] == 2
    assert stats["roughness_out_of_range_pixels"] == 1
