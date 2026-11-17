"""Generate deterministic demo WAV and source configuration files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def make_demo(
    output_dir: str | Path,
    sample_rate: int = 48000,
    duration_seconds: float = 1.0,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames = int(round(sample_rate * duration_seconds))
    time = np.arange(frames, dtype=np.float64) / sample_rate

    left = 0.30 * np.sin(2.0 * np.pi * 440.0 * time)
    left *= 0.65 + 0.35 * np.sin(2.0 * np.pi * 2.0 * time)
    right = 0.24 * np.sin(2.0 * np.pi * 660.0 * time + 0.2)
    right += 0.10 * np.sin(2.0 * np.pi * 1320.0 * time)
    click_positions = [
        int(0.10 * sample_rate),
        int(0.43 * sample_rate),
        int(0.77 * sample_rate),
    ]
    for frame in click_positions:
        if 0 <= frame < frames:
            left[frame] += 0.25
            right[frame] += 0.18

    stereo = np.column_stack((left, right))
    stereo = np.clip(stereo, -0.98, 0.98)
    wav_path = destination / "input_2ch.wav"
    sf.write(
        str(wav_path),
        stereo,
        samplerate=sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    config = {
        "hrtf_version": "local-hrtf-v1",
        "distance_model": {
            "type": "free_field",
            "reference_distance": 1.0,
            "minimum_distance": 0.05,
        },
        "air_absorption": {
            "enabled": True,
            "threshold": 3.0,
            "coefficient_per_meter": 0.16,
            "min_cutoff_hz": 650.0,
        },
        "sources": [
            {
                "id": "left-near",
                "channel": 0,
                "position": [-1.2, 0.15, 2.0],
                "gain": 0.72,
                "delay_samples": 0,
                "spatialize": True,
                "hrtf_gain": 0.95,
                "azimuth_offset_deg": -3.0,
            },
            {
                "id": "right-mid",
                "channel": 1,
                "position": [1.5, 0.45, 1.4],
                "gain": 0.55,
                "delay_samples": 82,
                "spatialize": True,
                "distance_model": {
                    "type": "custom",
                    "points": [[0.0, 1.15], [1.0, 1.0], [4.0, 0.42], [8.0, 0.18]],
                },
            },
            {
                "id": "moving-rear",
                "channel": 0,
                "position": [2.6, -0.25, -1.1],
                "positions": [
                    [0.0, 2.6, -0.25, -1.1],
                    [0.45, 1.7, 0.10, -0.3],
                    [0.90, 0.4, 0.35, 1.2],
                    [1.40, -0.8, 0.10, 1.8],
                ],
                "gain": 0.48,
                "delay_seconds": 0.002,
                "spatialize": True,
                "elevation_offset_deg": 4.0,
            },
        ],
    }
    config_path = destination / "sources_3.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return wav_path, config_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/demo")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    wav_path, config_path = make_demo(args.output_dir, args.sample_rate, args.duration)
    print(wav_path)
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
