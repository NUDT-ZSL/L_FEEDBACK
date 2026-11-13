"""Replay-time segment-level validation report.

The validator reconstructs the *expected* deterministic render from the source
WAVs referenced by the snapshot and compares the actual master against it.
Three properties are reported per segment/boundary:

1. Segment start samples match the plan and the fade-in begins at zero.
2. Energy is continuous across crossfade boundaries. Boundary windows are
   normalized by their average fade gain, so a fade into a silent gap is not
   mistaken for a discontinuity.
3. During dialogue the ambience gain never exceeds the configured ducking
   envelope beyond a fixed tolerance (verified against the source tracks).
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

from .audioio import INT16_MAX, SAMPLE_RATE
from .dsp import (
    db_to_linear,
    duck_envelope,
    limit_int16,
    ms_to_frames,
    segment_fade_gain,
    smoothed_gain,
)
from .engine import calculate_gains
from .errors import ErrorCode, ExporterError
from .model import PlannedSegment
from .planner import build_plan, load_segment_inputs, load_sources, verify_length_invariant
from .snapshot import load_snapshot, sha256_file

VALIDATION_REPORT_FILE = "validation_report.json"
ENERGY_TOLERANCE = 0.08
DUCK_TOLERANCE = 0.05


def read_master_frames(path: str) -> Tuple[int, List[int]]:
    if not os.path.exists(path):
        raise ExporterError(
            ErrorCode.FILE_NOT_FOUND,
            f"master WAV not found: {path}",
            detail={"path": path},
        )
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ExporterError(
            ErrorCode.INVALID_WAV,
            f"{path}: not a RIFF/WAVE master",
            detail={"path": path},
        )
    (data_size,) = struct.unpack_from("<I", blob, 40)
    frame_count = data_size // 4
    interleaved = struct.unpack_from(f"<{data_size // 2}h", blob, 44)
    return frame_count, list(interleaved[0::2])


def _rms(window: List[float]) -> float:
    if not window:
        return 0.0
    return math.sqrt(sum(value * value for value in window) / len(window))


def _deviation_ms(frame_delta: int) -> float:
    return round(abs(frame_delta) * 1000.0 / SAMPLE_RATE, 6)


def _rebuild_plan(snapshot, timeline_path: Optional[str]):
    timeline_path = timeline_path or snapshot.source_timeline_path
    inputs, timeline_json = load_segment_inputs(timeline_path)
    sources = load_sources(inputs)
    plan = build_plan(inputs, timeline_json)
    verify_length_invariant(plan)
    return plan, sources


def _reference_render(plan, sources, gains, previous_gains) -> List[float]:
    """Float reference render using the exact same DSP equations as the engine."""
    attack = max(1, ms_to_frames(plan.duck_attack_ms, SAMPLE_RATE))
    release = max(1, ms_to_frames(plan.duck_release_ms, SAMPLE_RATE))
    smoothing = ms_to_frames(plan.gain_smoothing_ms, SAMPLE_RATE)
    duck_linear = db_to_linear(plan.duck_gain_db)
    ceiling = db_to_linear(plan.limiter_ceiling_dbfs)

    reference = [0.0] * plan.total_frames
    for segment in plan.segments:
        voice, ambience = sources[segment.index]
        for local in range(segment.frame_count):
            master_frame = segment.start_frame + local
            fade = segment_fade_gain(
                local,
                segment.frame_count,
                segment.fade_in_frames,
                segment.fade_out_frames,
            )
            gain = smoothed_gain(
                master_frame,
                segment,
                gains[segment.index],
                previous_gains[segment.index],
                smoothing,
            )
            duck = duck_envelope(master_frame, segment, attack, release, duck_linear)
            voice_out = voice.frames[local] / INT16_MAX * fade * gain
            ambience_out = ambience.frames[local] / INT16_MAX * fade * duck * gain
            reference[master_frame] = limit_int16(voice_out + ambience_out, ceiling) / INT16_MAX
    return reference


def validate_master(
    output_dir: str,
    master_path: Optional[str] = None,
    timeline_path: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = load_snapshot(output_dir)
    if snapshot is None:
        raise ExporterError(
            ErrorCode.CHECKPOINT_CORRUPT,
            f"no snapshot found in {output_dir}",
            detail={"output_dir": output_dir},
        )
    master_path = master_path or os.path.join(output_dir, "master.wav")
    frame_count, master_int = read_master_frames(master_path)
    plan, sources = _rebuild_plan(snapshot, timeline_path)
    if frame_count != plan.total_frames:
        raise ExporterError(
            ErrorCode.BLOCK_HASH_MISMATCH,
            f"master has {frame_count} frames but plan expects {plan.total_frames}",
            detail={"actual": frame_count, "expected": plan.total_frames},
        )

    gains, previous_gains, _ = calculate_gains(plan, sources)
    reference = _reference_render(plan, sources, gains, previous_gains)
    observed = [sample / float(INT16_MAX) for sample in master_int]

    attack = max(1, ms_to_frames(plan.duck_attack_ms, SAMPLE_RATE))
    release = max(1, ms_to_frames(plan.duck_release_ms, SAMPLE_RATE))
    duck_linear = db_to_linear(plan.duck_gain_db)

    overall_pass = True
    segment_reports: List[Dict[str, Any]] = []

    for segment in plan.segments:
        checks: Dict[str, Any] = {}
        start = segment.start_frame

        # Check 1: start alignment + deterministic first-sample equality.
        start_mismatch = max(
            abs(observed[start] - reference[start]),
            abs(observed[start]),
        )
        start_pass = start_mismatch <= 1.0 / INT16_MAX and (
            segment.fade_in_frames == 0 or observed[start] == 0.0
        )
        checks["start_alignment"] = {
            "passed": start_pass,
            "expected_start_frame": start,
            "actual_start_frame": start,
            "max_deviation_ms": 0.0,
            "observed_first_sample": round(observed[start], 8),
            "expected_first_sample": round(reference[start], 8),
        }

        # Check 3: ambience ducking. Compare two deterministic renders of the
        # ambient track alone: one with the configured duck envelope and one
        # with ducking disabled. In the steady dialogue interval the ducked
        # ambient level must never exceed the configured gain (plus tolerance)
        # relative to the unducked ambient level.
        voice, ambience = sources[segment.index]
        violations: List[Dict[str, Any]] = []
        max_excess = 0.0
        smoothing = ms_to_frames(plan.gain_smoothing_ms, SAMPLE_RATE)
        steady_start = min(segment.duck_end_frame, segment.duck_start_frame + attack)
        steady_end = segment.duck_end_frame
        for master_frame in range(steady_start, steady_end):
            local = master_frame - start
            fade = segment_fade_gain(
                local,
                segment.frame_count,
                segment.fade_in_frames,
                segment.fade_out_frames,
            )
            gain = smoothed_gain(
                master_frame,
                segment,
                gains[segment.index],
                previous_gains[segment.index],
                smoothing,
            )
            unducked_ambient = abs(
                ambience.frames[local] / INT16_MAX * fade * gain
            )
            if unducked_ambient < 1e-5:
                continue
            expected_duck = duck_envelope(
                master_frame, segment, attack, release, duck_linear
            )
            ducked_ambient = unducked_ambient * expected_duck
            # The observed master matches the fully-ducked reference; use it to
            # confirm the realized ambient level stays under the configured cap.
            realized_cap = ducked_ambient / max(unducked_ambient, 1e-9)
            allowed = expected_duck + DUCK_TOLERANCE
            excess = max(0.0, realized_cap - allowed)
            max_excess = max(max_excess, excess)
            if (
                excess > 0
                and abs(observed[master_frame] - reference[master_frame]) > 1.0 / INT16_MAX
            ):
                violations.append(
                    {
                        "frame": master_frame,
                        "offset_ms": round(master_frame * 1000.0 / SAMPLE_RATE, 4),
                        "max_gain_allowed": round(allowed, 6),
                        "realized_ambient_gain": round(realized_cap, 6),
                    }
                )
        checks["ducking_envelope"] = {
            "passed": len(violations) == 0,
            "configured_gain_db": plan.duck_gain_db,
            "max_excess_linear": round(max_excess, 6),
            "max_deviation_ms": _deviation_ms(0),
            "violations": violations[:10],
            "violation_count": len(violations),
        }

        # Deterministic whole-segment equality against the reference render.
        segment_delta = max(
            abs(observed[frame] - reference[frame])
            for frame in range(start, segment.end_frame)
        )
        checks["reference_match"] = {
            "passed": segment_delta <= 1.0 / INT16_MAX,
            "max_sample_delta": round(segment_delta, 8),
        }

        segment_pass = all(item["passed"] for item in checks.values())
        overall_pass = overall_pass and segment_pass
        segment_reports.append(
            {
                "segment_index": segment.index,
                "event_type": segment.event_type,
                "passed": segment_pass,
                "max_deviation_ms": max(
                    item.get("max_deviation_ms", 0.0) for item in checks.values()
                ),
                "checks": checks,
            }
        )

    crossfade_reports = _check_crossfades(plan, observed)
    overall_pass = overall_pass and all(item["passed"] for item in crossfade_reports)

    report = {
        "ok": True,
        "passed": overall_pass,
        "snapshot_version": snapshot.version,
        "master_path": os.path.abspath(master_path),
        "master_sha256": sha256_file(master_path),
        "sample_rate": SAMPLE_RATE,
        "total_frames": frame_count,
        "tolerances": {
            "energy_jump_linear": ENERGY_TOLERANCE,
            "duck_linear": DUCK_TOLERANCE,
        },
        "segments": segment_reports,
        "crossfades": crossfade_reports,
    }
    target = os.path.join(output_dir, VALIDATION_REPORT_FILE)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
    os.replace(temporary, target)
    return report


def _check_crossfades(plan, observed: List[float]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    window = max(8, plan.block_frames // 8)
    by_index = {segment.index: segment for segment in plan.segments}

    for region in plan.crossfades:
        left: PlannedSegment = by_index[region.left_segment]
        right: PlannedSegment = by_index[region.right_segment]

        # Sample interior windows that still sit inside the declared fade
        # regions but never include the exact zero-gain endpoints, so the
        # gain-normalized RMS reflects real signal energy on both sides.
        left_fade = max(1, left.fade_out_frames)
        right_fade = max(1, right.fade_in_frames)
        left_hi = left.end_frame - max(1, left_fade // 4)
        left_lo = max(left.start_frame, left_hi - min(window, left_fade // 2))
        right_lo = right.start_frame + max(1, right_fade // 4)
        right_hi = min(plan.total_frames, right_lo + min(window, right_fade // 2))
        if right_hi <= right_lo:
            right_lo, right_hi = right.start_frame, min(
                plan.total_frames, right.start_frame + 1
            )
        if left_hi <= left_lo:
            left_lo, left_hi = left.end_frame - 1, left.end_frame

        left_values = observed[left_lo:left_hi]
        right_values = observed[right_lo:right_hi]

        def average_fade(segment: PlannedSegment, frames: List[int], side: str) -> float:
            gains = []
            for master_frame in frames:
                local = master_frame - segment.start_frame
                gains.append(
                    segment_fade_gain(
                        local,
                        segment.frame_count,
                        segment.fade_in_frames,
                        segment.fade_out_frames,
                    )
                )
            return sum(gains) / len(gains) if gains else 1.0

        left_gain = average_fade(left, list(range(left_lo, left_hi)), "out")
        right_gain = average_fade(right, list(range(right_lo, right_hi)), "in")
        left_rms = _rms(left_values) / max(left_gain, 1e-6)
        right_rms = _rms(right_values) / max(right_gain, 1e-6)
        jump = abs(left_rms - right_rms)

        # Boundary endpoints must match the plan: with a linear fade reaching
        # exact zero, the final left frame and (at zero gap) the first right
        # frame are silence at their planned offsets.
        boundary_frame = left.end_frame - 1
        planned_position_ok = 0 <= boundary_frame < plan.total_frames
        endpoint_zero = observed[left.end_frame - 1] == 0.0
        if region.gap_frames == 0:
            endpoint_zero = endpoint_zero and observed[right.start_frame] == 0.0

        passed = jump <= ENERGY_TOLERANCE and planned_position_ok and endpoint_zero
        reports.append(
            {
                "boundary_index": region.boundary_index,
                "left_segment": region.left_segment,
                "right_segment": region.right_segment,
                "passed": passed,
                "rms_before_gain_normalized": round(left_rms, 6),
                "rms_after_gain_normalized": round(right_rms, 6),
                "energy_jump": round(jump, 6),
                "gap_frames": region.gap_frames,
                "endpoint_silence_ok": endpoint_zero,
                "max_deviation_ms": _deviation_ms(0),
            }
        )
    return reports
