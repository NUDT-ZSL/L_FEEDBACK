#!/usr/bin/env python3
"""Generate deterministic local WAV samples for pipeline verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


def _time_axis(sample_rate: int, duration: float) -> np.ndarray:
    return np.arange(int(round(sample_rate * duration)), dtype=np.float64) / sample_rate


def _channel_specs(channel_count: int) -> List[Tuple[float, float]]:
    if channel_count == 2:
        return [(440.0, 0.10), (660.0, 0.08)]
    if channel_count == 6:
        return [
            (220.0, 0.12),
            (330.0, 0.11),
            (440.0, 0.10),
            (55.0, 0.35),
            (550.0, 0.08),
            (660.0, 0.07),
        ]
    return [(220.0 + 110.0 * index, 0.08) for index in range(channel_count)]


def make_audio(
    sample_rate: int,
    channel_count: int,
    duration: float,
) -> np.ndarray:
    time = _time_axis(sample_rate, duration)
    envelope = 0.60 + 0.25 * np.sin(2.0 * np.pi * 0.13 * time) ** 2
    audio = np.empty((time.shape[0], channel_count), dtype=np.float64)
    for channel, (frequency, amplitude) in enumerate(_channel_specs(channel_count)):
        phase = 0.17 * channel
        audio[:, channel] = (
            amplitude
            * envelope
            * np.sin(2.0 * np.pi * frequency * time + phase)
        )
    return audio


def make_limiter_audio(sample_rate: int = 48000) -> np.ndarray:
    time = _time_axis(sample_rate, 3.0)
    base = 0.012 * np.sin(2.0 * np.pi * 440.0 * time)
    pulse = (time >= 1.5) & (time < 1.51)
    base[pulse] += 0.40 * np.sin(2.0 * np.pi * 1000.0 * time[pulse])
    return np.stack([base, base], axis=1)


def generate(output_dir: Path, duration: float) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    cases = [
        ("stereo_32k_pcm32.wav", 32000, 2, "PCM_32"),
        ("stereo_44k1_pcm16.wav", 44100, 2, "PCM_16"),
        ("stereo_48k_pcm24.wav", 48000, 2, "PCM_24"),
        ("stereo_96k_float.wav", 96000, 2, "FLOAT"),
        ("surround_44k1_double.wav", 44100, 6, "DOUBLE"),
        ("surround_48k_pcm24.wav", 48000, 6, "PCM_24"),
        ("unsupported_22050_pcm16.wav", 22050, 2, "PCM_16"),
    ]
    for name, sample_rate, channels, subtype in cases:
        path = output_dir / name
        sf.write(
            str(path),
            make_audio(sample_rate, channels, duration),
            sample_rate,
            subtype=subtype,
            format="WAV",
        )
        generated.append(path)

    limiter_path = output_dir / "stereo_48k_limiter_float.wav"
    sf.write(
        str(limiter_path),
        make_limiter_audio(48000),
        48000,
        subtype="FLOAT",
        format="WAV",
    )
    generated.append(limiter_path)

    empty_path = output_dir / "empty_48k_pcm16.wav"
    sf.write(
        str(empty_path),
        np.zeros((0, 2), dtype=np.float64),
        48000,
        subtype="PCM_16",
        format="WAV",
    )
    generated.append(empty_path)

    corrupt_path = output_dir / "corrupt.wav"
    corrupt_path.write_bytes(b"not a RIFF/WAVE file\n")
    generated.append(corrupt_path)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testdata"),
        help="directory for generated WAV files (default: testdata)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.2,
        help="duration of normal samples in seconds (default: 5.2)",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    for path in generate(args.output_dir, args.duration):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

