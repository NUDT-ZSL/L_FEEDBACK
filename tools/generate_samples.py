#!/usr/bin/env python3
"""Generate deterministic local WAV/timeline fixtures with fixed seed 20261116."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audiorepair.wavio import encode_wav

SEED = 20261116
TARGET_LUFS = -23.0
TOLERANCE_LU = 0.5
CROSSFADE_MS = 20.0
GAP_SECONDS = 0.016
FADE_SECONDS = 0.003


def _raised_ramp(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    x = (np.arange(length, dtype=np.float64) + 0.5) / length
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _render_segment(
    sample_rate: int,
    duration: float,
    channels: int,
    amplitude: float,
    boundaries: Sequence[float],
    root_frequencies: Sequence[float],
    generator: np.random.Generator,
    dc: float,
) -> np.ndarray:
    length = int(round(duration * sample_rate))
    mono = np.zeros(length, dtype=np.float64)
    edge_points = [0.0] + list(boundaries) + [duration]
    gap = int(round(GAP_SECONDS * sample_rate))
    fade = max(1, int(round(FADE_SECONDS * sample_rate)))

    for index in range(len(edge_points) - 1):
        active_start = int(round(edge_points[index] * sample_rate)) + gap // 2
        active_end = int(round(edge_points[index + 1] * sample_rate)) - gap // 2
        active_start = max(0, active_start)
        active_end = min(length, active_end)
        if active_end <= active_start:
            continue
        count = active_end - active_start
        positions = np.arange(count, dtype=np.float64)
        frequency = root_frequencies[index % len(root_frequencies)]
        phase = (index * 0.371) % (2.0 * np.pi)
        tone = (
            0.78 * np.sin(2.0 * np.pi * frequency * positions / sample_rate + phase)
            + 0.18
            * np.sin(
                4.0 * np.pi * frequency * positions / sample_rate + 0.5 * phase
            )
            + 0.04
            * np.sin(
                6.0 * np.pi * frequency * positions / sample_rate + 1.7 * phase
            )
        )
        envelope = np.ones(count, dtype=np.float64)
        ramp_length = min(fade, max(1, count // 2))
        ramp = _raised_ramp(ramp_length)
        envelope[:ramp_length] *= ramp
        envelope[-ramp_length:] *= ramp[::-1]
        syllable_gain = amplitude * (0.82 + 0.09 * (index % 3))
        mono[active_start:active_end] += syllable_gain * tone * envelope

    noise = generator.normal(0.0, 0.0035, size=length)
    mono += noise
    mono += dc
    if channels == 1:
        return mono[:, None]
    right_generator = np.random.default_rng(SEED + length)
    right_noise = right_generator.normal(0.0, 0.0035, size=length)
    left = mono
    shifted = np.roll(mono, max(1, int(0.0007 * sample_rate)))
    right = 0.91 * shifted + right_noise + dc * 0.75
    return np.stack((left, right), axis=1)


def _sample_timeline(
    segments: List[Dict[str, Any]], splice_points: Sequence[float]
) -> Dict[str, Any]:
    return {
        "version": 1,
        "target_lufs": TARGET_LUFS,
        "tolerance_lu": TOLERANCE_LU,
        "crossfade_ms": CROSSFADE_MS,
        "splice_points_sec": list(splice_points),
        "segments": segments,
    }


def generate(output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, str] = {}
    segment_duration = 2.8

    layouts = [
        {
            "name": "dialogue_48k_mono",
            "sample_rate": 48000,
            "channels": 1,
            "amplitudes": [0.12, 0.64],
            "dc": [0.012, -0.009],
            "segment_boundaries": [[0.55, 1.15, 1.85], [0.65, 1.35, 2.00]],
            "stretch_ratios": [1.02, 0.985],
            "ranges": [[0.97, 1.05], [0.96, 1.02]],
            "expected_boundaries": [
                [0.558, 1.176, 1.887],
                [0.641, 1.330, 1.970],
            ],
            "frequencies": [
                [196.0, 294.0, 392.0, 247.0],
                [330.0, 440.0, 523.0, 392.0],
            ],
        },
        {
            "name": "dialogue_44k_stereo",
            "sample_rate": 44100,
            "channels": 2,
            "amplitudes": [0.18, 0.52],
            "dc": [-0.007, 0.011],
            "segment_boundaries": [[0.52, 1.10, 1.78], [0.61, 1.29, 1.96]],
            "stretch_ratios": [0.99, 1.015],
            "ranges": [[0.96, 1.03], [0.97, 1.04]],
            "expected_boundaries": [
                [0.516, 1.092, 1.764],
                [0.618, 1.305, 1.987],
            ],
            "frequencies": [
                [220.0, 330.0, 440.0, 277.0],
                [294.0, 392.0, 494.0, 349.0],
            ],
        },
    ]

    for layout in layouts:
        name = layout["name"]
        sample_rate = layout["sample_rate"]
        channels = layout["channels"]
        generator = np.random.default_rng(SEED)
        pieces = []
        for segment_index, duration in enumerate(
            [segment_duration, segment_duration]
        ):
            pieces.append(
                _render_segment(
                    sample_rate=sample_rate,
                    duration=duration,
                    channels=channels,
                    amplitude=layout["amplitudes"][segment_index],
                    boundaries=layout["segment_boundaries"][segment_index],
                    root_frequencies=layout["frequencies"][segment_index],
                    generator=generator,
                    dc=layout["dc"][segment_index],
                )
            )
        samples = np.concatenate(pieces, axis=0)
        wav_path = os.path.join(output_dir, name + ".wav")
        with open(wav_path, "wb") as handle:
            handle.write(encode_wav(samples, sample_rate, channels))

        segments = []
        for segment_index in range(2):
            segments.append(
                {
                    "id": "seg-{}".format(segment_index),
                    "start_sec": round(segment_index * segment_duration, 9),
                    "end_sec": round((segment_index + 1) * segment_duration, 9),
                    "stretch_ratio": layout["stretch_ratios"][segment_index],
                    "stretch_range": layout["ranges"][segment_index],
                    "source_boundaries_sec": layout["segment_boundaries"][
                        segment_index
                    ],
                    "expected_boundaries_sec": layout["expected_boundaries"][
                        segment_index
                    ],
                }
            )
        timeline_path = os.path.join(output_dir, name + ".timeline.json")
        with open(timeline_path, "w", encoding="utf-8") as handle:
            json.dump(
                _sample_timeline(segments, [segment_duration]),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        result[name] = {
            "wav": wav_path,
            "timeline": timeline_path,
            "sample_rate": sample_rate,
            "channels": channels,
        }

    batch_path = os.path.join(output_dir, "batch.json")
    batch = {
        "jobs": [
            {
                "id": "mono48",
                "input": "dialogue_48k_mono.wav",
                "timeline": "dialogue_48k_mono.timeline.json",
                "output": "batch_out/mono48.wav",
                "report": "batch_out/mono48.report.json",
            },
            {
                "id": "stereo44",
                "input": "dialogue_44k_stereo.wav",
                "timeline": "dialogue_44k_stereo.timeline.json",
                "output": "batch_out/stereo44.wav",
                "report": "batch_out/stereo44.report.json",
            },
        ]
    }
    with open(batch_path, "w", encoding="utf-8") as handle:
        json.dump(batch, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    result["batch"] = batch_path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    generated = generate(args.output_dir)
    print(json.dumps(generated, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

