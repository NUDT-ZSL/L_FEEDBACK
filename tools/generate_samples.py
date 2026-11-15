#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def frames_for_duration(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))


def generate(output_dir: Path, duration: float, sample_rate: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = int(round(duration * sample_rate))
    t = np.arange(count, dtype=np.float64) / sample_rate
    envelope = np.linspace(0.2, 1.0, count, dtype=np.float64)
    base = (
        0.24 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.04 * np.sin(2.0 * np.pi * 1320.0 * t)
    ) * envelope

    lead_25ms = np.zeros(
        frames_for_duration(0.025, sample_rate), dtype=np.float64
    )
    trimmed_tail = max(1, count - frames_for_duration(0.015, sample_rate))

    write_wav(output_dir / "audio_64k.wav", base, sample_rate)
    write_wav(output_dir / "audio_128k.wav", np.concatenate([lead_25ms, base]), sample_rate)
    write_wav(output_dir / "audio_192k.wav", base[:trimmed_tail], sample_rate)

    config = {
        "output_dir": "../demo_output",
        "segment_duration": 2,
        "overwrite": False,
        "variants": [
            {
                "bitrate_kbps": 64,
                "input": "audio_64k.wav",
                "offset_ms": 0,
            },
            {
                "bitrate_kbps": 128,
                "input": "audio_128k.wav",
                "offset_ms": 25,
            },
            {
                "bitrate_kbps": 192,
                "input": "audio_192k.wav",
                "offset_ms": 0,
            },
        ],
    }
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic 44.1 kHz WAV files for HLS validation."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=7.3)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()
    config_path = generate(args.output_dir, args.duration, args.sample_rate)
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

