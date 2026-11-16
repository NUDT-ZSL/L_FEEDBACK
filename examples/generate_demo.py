#!/usr/bin/env python3
"""Generate deterministic multi-rate WAV fixtures and a demo configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
AUDIO = HERE / "demo_audio"


def write_tone(name, sample_rate, duration, frequency, amplitude, phase=0.0):
    time = np.arange(int(round(sample_rate * duration))) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
    sf.write(
        str(AUDIO / name),
        np.stack((signal, signal), axis=1),
        sample_rate,
        subtype="FLOAT",
    )


def main():
    AUDIO.mkdir(parents=True, exist_ok=True)
    write_tone("dialogue.wav", 44100, 4.0, 997.0, 0.40)
    write_tone("sfx.wav", 48000, 3.0, 300.0, 0.30)
    write_tone("music.wav", 96000, 4.5, 1200.0, 0.22, 0.4)
    config = {
        "output": {
            "path": "demo_mix.wav",
            "sample_rate": 48000,
            "channels": 2,
            "duration_seconds": 5.0,
            "subtype": "PCM_32",
            "limiter_placement": "bus",
        },
        "bus_limiter": {
            "ceiling_dbfs": -1.0,
            "lookahead_ms": 5.0,
            "release_ms": 80.0,
        },
        "processing": {"workers": 0, "chunk_frames": 16384},
        "tracks": [
            {
                "id": "dialogue",
                "input": "demo_audio/dialogue.wav",
                "offset_ms": 0.0,
                "gain_db": -2.0,
                "compressor": {
                    "threshold_db": -22.0,
                    "ratio": 3.0,
                    "knee_db": 6.0,
                    "attack_ms": 8.0,
                    "release_ms": 120.0,
                },
            },
            {
                "id": "sfx",
                "input": "demo_audio/sfx.wav",
                "offset_ms": 250.4,
                "gain_db": 10.0,
            },
            {
                "id": "music",
                "input": "demo_audio/music.wav",
                "offset_ms": -75.0,
                "gain_db": -5.0,
            },
        ],
    }
    config_path = HERE / "demo.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(config_path)


if __name__ == "__main__":
    main()
