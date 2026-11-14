#!/usr/bin/env python3
"""Generate deterministic local WAV inputs and a demo JSON configuration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48000


def tone(
    path: Path,
    duration_ms: int,
    amplitude: float,
    frequency: float,
    channels: int = 2,
    sample_rate: int = SAMPLE_RATE,
    subtype: str = "PCM_24",
) -> None:
    frames = int(round(duration_ms * sample_rate / 1000.0))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(
        2.0 * math.pi * frequency * time + math.pi / 2.0
    )
    data = np.repeat(mono[:, None], channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        data,
        sample_rate,
        format="WAVEX",
        subtype=subtype,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="demo_data",
        help="directory for generated inputs and config.json",
    )
    args = parser.parse_args()
    root = Path(args.output_dir)
    audio_dir = root / "audio"

    tone(audio_dir / "line_01.wav", 800, 0.01, 210.0)
    tone(audio_dir / "line_02.wav", 700, 0.01, 310.0)
    tone(
        audio_dir / "bad_format.wav",
        500,
        0.02,
        190.0,
        channels=1,
        sample_rate=44100,
    )
    tone(audio_dir / "late.wav", 1000, 0.02, 270.0)
    (audio_dir / "broken.wav").write_bytes(b"deterministic corrupt input")

    config = {
        "timeline_ms": 5000,
        "sample_rate": SAMPLE_RATE,
        "bit_depth": 24,
        "channels": 2,
        "channel_layout": "stereo",
        "clips": [
            {
                "id": "line_01",
                "input_path": "audio/line_01.wav",
                "start_ms": 300,
                "target_lufs": -23.0,
                "fade_in_ms": 30,
                "fade_out_ms": 50,
            },
            {
                "id": "line_02_overlap",
                "input_path": "audio/line_02.wav",
                "start_ms": 800,
                "target_lufs": -23.0,
                "fade_in_ms": 20,
                "fade_out_ms": 40,
            },
            {
                "id": "bad_format",
                "input_path": "audio/bad_format.wav",
                "start_ms": 1900,
                "target_lufs": -23.0,
                "fade_in_ms": 10,
                "fade_out_ms": 10,
            },
            {
                "id": "broken",
                "input_path": "audio/broken.wav",
                "start_ms": 1900,
                "target_lufs": -23.0,
                "fade_in_ms": 10,
                "fade_out_ms": 10,
            },
            {
                "id": "late",
                "input_path": "audio/late.wav",
                "start_ms": 4200,
                "target_lufs": -23.0,
                "fade_in_ms": 10,
                "fade_out_ms": 10,
            },
        ],
    }
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
