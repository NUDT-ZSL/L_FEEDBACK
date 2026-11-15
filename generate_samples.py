#!/usr/bin/env python3
"""Generate deterministic WAV clips and a timeline for local verification."""

import argparse
import json
import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    mono = np.asarray(samples, dtype=np.float64)
    pcm = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def _write_stereo16(
    path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int
) -> None:
    interleaved = np.empty(left.size * 2, dtype=np.float64)
    interleaved[0::2] = left
    interleaved[1::2] = right
    pcm = np.clip(np.rint(interleaved * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def _write_float32_wave(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    raw = np.asarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack(
        "<HHIIHH",
        3,
        1,
        sample_rate,
        sample_rate * 4,
        4,
        32,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(raw))
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", riff_size))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(struct.pack("<I", len(fmt)))
        handle.write(fmt)
        handle.write(b"data")
        handle.write(struct.pack("<I", len(raw)))
        handle.write(raw)


def _add_decaying_burst(
    samples: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    frequency: float,
    amplitude: float,
    decay_seconds: float,
    length_seconds: float,
) -> None:
    start = int(round(start_seconds * sample_rate))
    length = int(round(length_seconds * sample_rate))
    local_time = np.arange(length, dtype=np.float64) / sample_rate
    burst = (
        amplitude
        * np.sin(2.0 * np.pi * frequency * local_time)
        * np.exp(-local_time / decay_seconds)
    )
    end = min(start + length, samples.size)
    samples[start:end] += burst[: end - start]


def _build_impact(sample_rate: int = 48000) -> np.ndarray:
    duration = 1.2
    samples = np.zeros(int(round(duration * sample_rate)), dtype=np.float64)
    _add_decaying_burst(
        samples, sample_rate, 0.204, 620.0, 0.72, 0.018, 0.24
    )
    _add_decaying_burst(
        samples, sample_rate, 0.704, 840.0, 0.66, 0.020, 0.22
    )
    return samples


def _build_stretch(sample_rate: int = 44100) -> np.ndarray:
    duration = 0.6
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64)
    time /= sample_rate
    samples = 0.28 * np.sin(2.0 * np.pi * 220.0 * time)
    _add_decaying_burst(
        samples, sample_rate, 0.402, 1100.0, 0.82, 0.012, 0.08
    )
    return samples


def _build_invalid_scale(sample_rate: int = 48000) -> np.ndarray:
    duration = 0.5
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64)
    time /= sample_rate
    return 0.45 * np.sin(2.0 * np.pi * 330.0 * time)


def _build_stereo(sample_rate: int = 48000) -> tuple:
    duration = 0.4
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64)
    time /= sample_rate
    left = 0.40 * np.sin(2.0 * np.pi * 190.0 * time)
    right = 0.35 * np.sin(2.0 * np.pi * 410.0 * time)
    return left, right


def _build_float_unsupported(sample_rate: int = 48000) -> np.ndarray:
    duration = 0.3
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64)
    time /= sample_rate
    return 0.30 * np.sin(2.0 * np.pi * 500.0 * time)


def _timeline(audio: list) -> dict:
    return {
        "version": 1,
        "description": "Deterministic animation-sound timeline fixture",
        "audio": audio,
    }


def generate(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _write_pcm16(audio_dir / "impact_48k.wav", _build_impact(), 48000)
    _write_pcm16(audio_dir / "stretch_44k.wav", _build_stretch(), 44100)
    _write_pcm16(
        audio_dir / "invalid_scale_48k.wav", _build_invalid_scale(), 48000
    )

    stereo_left, stereo_right = _build_stereo()
    _write_stereo16(
        audio_dir / "stereo_unsupported.wav",
        stereo_left,
        stereo_right,
        48000,
    )
    _write_float32_wave(
        audio_dir / "float_unsupported.wav",
        _build_float_unsupported(),
        48000,
    )

    timeline = _timeline(
        [
            {
                "id": "impact",
                "file": "impact_48k.wav",
                "events": [
                    {"name": "first_hit", "time": 0.200},
                    {"name": "second_hit", "time": 0.700},
                    {"name": "late_hit", "time": 1.500},
                ],
                "loops": [[0.100, 0.350], [0.500, 0.515]],
            },
            {
                "id": "stretch",
                "file": "stretch_44k.wav",
                "events": [{"name": "stretched_hit", "time": 0.800}],
                "loops": [[0.200, 0.700]],
                "time_scale": 2.0,
            },
            {
                "id": "invalid_scale",
                "file": "invalid_scale_48k.wav",
                "events": [{"name": "center_hit", "time": 0.250}],
                "loops": [{"start": 0.100, "end": 0.300}],
                "time_scale": 11.0,
            },
            {
                "id": "stereo_skip",
                "file": "stereo_unsupported.wav",
                "events": [{"name": "stereo_hit", "time": 0.100}],
                "loops": [[0.050, 0.200]],
            },
            {
                "id": "float_skip",
                "file": "float_unsupported.wav",
                "events": [{"name": "float_hit", "time": 0.100}],
                "loops": [[0.050, 0.150]],
            },
            {
                "id": "missing",
                "file": "does_not_exist.wav",
                "events": [{"name": "missing_hit", "time": 0.100}],
                "loops": [[0.050, 0.150]],
            },
        ]
    )

    timeline_path = output_dir / "timeline.json"
    with timeline_path.open("w", encoding="utf-8") as handle:
        json.dump(timeline, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return timeline


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic local WAV samples and timeline JSON."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples"),
        help="destination directory (default: samples)",
    )
    args = parser.parse_args(argv)
    generate(args.output_dir)
    print("Generated samples in {}".format(args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
