"""Construct deterministic telemetry and parameter samples without assets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FRAME_COUNT = 2400
FRAME_INTERVAL_MS = 5.0


PARAMS = {
    "schema_version": 1,
    "model_id": "deterministic_race_demo_v1",
    "engine": {
        "rpm_min": 800.0,
        "rpm_max": 9500.0,
        "firing_order": 4.0,
        "harmonics": [
            {"order": 0.5, "gain": 0.26, "phase_deg": 0.0},
            {"order": 1.0, "gain": 0.58, "phase_deg": 37.0},
            {"order": 2.0, "gain": 0.27, "phase_deg": 103.0},
            {"order": 3.0, "gain": 0.14, "phase_deg": 191.0},
            {"order": 4.0, "gain": 0.075, "phase_deg": 266.0},
            {"order": 6.0, "gain": 0.036, "phase_deg": 319.0},
        ],
        "base_amplitude": 0.24,
        "throttle_amplitude": 0.94,
        "shift_rpm_threshold": 1400.0,
        "shift_crossfade_ms": 20.0,
        "stereo_phase_spread_deg": 11.0,
    },
    "tires": {
        "level": 0.55,
        "max_slip": 0.8,
        "slip_threshold": 0.22,
        "release_ratio": 0.65,
        "attack_ms": 7.0,
        "release_ms": 34.0,
        "slip_full_scale": 0.52,
        "slip_load_floor": 0.35,
        "surfaces": {
            "asphalt": {
                "grip": {"cutoff_hz": 920.0, "gain": 0.18},
                "slip": {"cutoff_hz": 2450.0, "gain": 0.52},
            },
            "gravel": {
                "grip": {"cutoff_hz": 1550.0, "gain": 0.30},
                "slip": {"cutoff_hz": 3900.0, "gain": 0.66},
            },
            "kerb": {
                "grip": {"cutoff_hz": 2900.0, "gain": 0.26},
                "slip": {"cutoff_hz": 5600.0, "gain": 0.61},
            },
            "grass": {
                "grip": {"cutoff_hz": 720.0, "gain": 0.24},
                "slip": {"cutoff_hz": 1900.0, "gain": 0.47},
            },
        },
    },
    "events": {
        "overlap_window_ms": 45.0,
        "overlap_attenuation": 0.68,
        "types": {
            "suspension_impact": {
                "gain": 0.50,
                "duration_ms": 86.0,
                "attack_ms": 1.2,
                "decay_ms": 25.0,
                "waveform": "impact",
                "frequency_hz": 72.0,
                "frequency_end_hz": 43.0,
                "noise_mix": 0.42,
            },
            "surface_bump": {
                "gain": 0.34,
                "duration_ms": 62.0,
                "attack_ms": 1.6,
                "decay_ms": 18.0,
                "waveform": "impact",
                "frequency_hz": 118.0,
                "frequency_end_hz": 84.0,
                "noise_mix": 0.58,
            },
            "shift_alert": {
                "gain": 0.27,
                "duration_ms": 124.0,
                "attack_ms": 4.0,
                "decay_ms": 76.0,
                "waveform": "tone",
                "frequency_hz": 930.0,
                "frequency_end_hz": 690.0,
                "noise_mix": 0.03,
            },
        },
    },
    "mix": {
        "engine_gain": 0.78,
        "tire_gain": 0.80,
        "event_gain": 0.88,
        "master_gain": 0.72,
        "soft_limit_threshold": 0.92,
    },
}


def _surface_for_cycle(cycle: int) -> str:
    if cycle < 110:
        return "asphalt"
    if cycle < 210:
        return "gravel"
    if cycle < 300:
        return "kerb"
    if cycle < 390:
        return "gravel"
    if cycle < 525:
        return "grass"
    return "asphalt"


def construct_telemetry() -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for index in range(FRAME_COUNT):
        cycle = index % 600
        rpm = 900.0 + cycle * 12.8 + 34.0 * math.sin(cycle * 0.061)
        rpm = min(PARAMS["engine"]["rpm_max"], max(PARAMS["engine"]["rpm_min"], rpm))
        throttle_shape = 0.5 + 0.5 * math.sin(cycle * 0.034 - 0.8)
        throttle = 0.13 + 0.86 * throttle_shape * throttle_shape

        if 115 <= cycle < 190 or 305 <= cycle < 365:
            slip = 0.36 + 0.075 * math.sin(cycle * 0.17)
            slip = min(PARAMS["tires"]["max_slip"], slip)
        else:
            slip = 0.025 + 0.026 * (0.5 + 0.5 * math.sin(cycle * 0.09))

        frames.append(
            {
                "frame_index": index,
                "time_ms": round(index * FRAME_INTERVAL_MS, 6),
                "rpm": round(rpm, 6),
                "throttle": round(throttle, 6),
                "slip": round(slip, 6),
                "surface": _surface_for_cycle(cycle),
            }
        )

    events = [
        {"frame_index": 165, "type": "suspension_impact", "intensity": 0.88, "pan": -0.35},
        {"frame_index": 169, "type": "surface_bump", "intensity": 0.63, "pan": 0.22},
        {"frame_index": 172, "type": "suspension_impact", "intensity": 0.52, "pan": 0.48},
        {"frame_index": 208, "type": "surface_bump", "intensity": 0.77, "pan": -0.12},
        {"frame_index": 351, "type": "surface_bump", "intensity": 0.69, "pan": 0.31},
        {"frame_index": 600, "type": "shift_alert", "intensity": 0.82, "pan": -0.08},
        {"frame_index": 781, "type": "suspension_impact", "intensity": 0.74, "pan": 0.25},
        {"frame_index": 785, "type": "surface_bump", "intensity": 0.58, "pan": -0.40},
        {"frame_index": 1200, "type": "shift_alert", "intensity": 0.90, "pan": 0.10},
        {"frame_index": 1396, "type": "surface_bump", "intensity": 0.66, "pan": -0.27},
        {"frame_index": 1800, "type": "shift_alert", "intensity": 0.86, "pan": 0.0},
        {"frame_index": 2160, "type": "suspension_impact", "intensity": 0.61, "pan": 0.38},
        {"frame_index": 2398, "type": "surface_bump", "intensity": 0.72, "pan": -0.18},
    ]

    return {
        "schema_version": 1,
        "format": {
            "sample_rate": 48000,
            "channels": 2,
            "bit_depth": 16,
        },
        "frame_interval_ms": FRAME_INTERVAL_MS,
        "frame_count": len(frames),
        "frames": frames,
        "events": events,
    }


def write_example_files(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / "telemetry.json"
    params_path = output_dir / "params.json"
    telemetry_path.write_text(
        json.dumps(construct_telemetry(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    params_path.write_text(
        json.dumps(PARAMS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return telemetry_path, params_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("samples"),
        help="directory receiving telemetry.json and params.json",
    )
    args = parser.parse_args()
    telemetry_path, params_path = write_example_files(args.out_dir)
    print(f"wrote {telemetry_path}")
    print(f"wrote {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

