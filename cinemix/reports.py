"""Quality-check report construction."""

from __future__ import annotations

from typing import Dict

from .config import CUT_TRANSITION_MS, DUCK_TRANSITION_MS
from .mixing import MixResult
from .scene import SceneSpec


def build_switch_report(
    scene: SceneSpec,
    mix_result: MixResult,
    sample_rate: int,
) -> Dict[str, object]:
    cuts = []
    for check in mix_result.cut_checks:
        cuts.append(
            {
                "index": check.index,
                "time_seconds": check.cut_time_seconds,
                "cut_sample": check.cut_sample,
                "fade_seconds": check.fade_samples / sample_rate,
                "fade_samples": check.fade_samples,
                "window_start_sample": check.window_start_sample,
                "window_end_sample_exclusive": check.window_end_sample,
                "pre_samples": check.pre_samples,
                "post_samples": check.post_samples,
                "equal_power": True,
                "equal_power_weight_max_error": check.equal_power_max_error,
                "left_boundary_value": check.left_boundary_value,
                "left_boundary_expected": check.left_boundary_expected,
                "right_boundary_value": check.right_boundary_value,
                "right_boundary_expected": check.right_boundary_expected,
                "boundary_continuity_error": check.boundary_continuity_error,
                "boundary_continuous": check.boundary_continuity_error <= 1e-12,
                "max_abs_step_before": check.max_abs_step_before,
                "max_abs_step_after": check.max_abs_step_after,
                "peak_in_window": check.peak_in_window,
                "clipped_samples_in_window": check.clipped_samples,
                "passed": check.passed,
            }
        )

    ducking = []
    for check in mix_result.duck_intervals:
        ducking.append(
            {
                "index": check.index,
                "start_sample": check.start_sample,
                "end_sample": check.end_sample,
                "attack_samples": check.attack_samples,
                "release_samples": check.release_samples,
                "transition_seconds": DUCK_TRANSITION_MS / 1000.0,
                "equal_power": True,
                "steady_duck_gain": check.duck_gain,
                "gain_at_start_boundary": check.gain_at_start,
                "gain_at_end_boundary": check.gain_at_end,
                "max_adjacent_gain_step": check.max_adjacent_step,
                "boundary_continuous": (
                    abs(check.gain_at_start - 1.0) <= 1e-12
                    and abs(check.gain_at_end - 1.0) <= 1e-12
                ),
            }
        )

    all_passed = all(item["passed"] for item in cuts)
    return {
        "scene_id": scene.scene_id,
        "sample_rate_hz": sample_rate,
        "total_samples": mix_result.total_samples,
        "duration_seconds": mix_result.total_samples / sample_rate,
        "cut_crossfade_ms": CUT_TRANSITION_MS,
        "cuts": cuts,
        "all_cuts_passed": all_passed,
        "dialogue_ducking": ducking,
        "all_ducking_boundaries_continuous": all(
            item["boundary_continuous"] for item in ducking
        ),
        "mix_peak_abs": mix_result.peak_abs,
        "mix_clipped_samples": mix_result.clipped_samples,
    }
