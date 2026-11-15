"""Deterministic test-sample construction for the mastering tool."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .wavio import write_wav


DEFAULT_SEED = 130
DEFAULT_DURATION = 9.0


def _fade_envelope(length: int, sample_rate: int, fade_ms: float = 35.0) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float64)
    fade = min(length // 2, max(1, int(round(fade_ms * sample_rate / 1000.0))))
    if fade:
        phase = np.linspace(0.0, math.pi, fade, endpoint=False)
        ramp = np.sin(phase * 0.5) ** 2
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    return envelope


def _speech_like(
    sample_rate: int,
    duration: float,
    segments: Sequence[Tuple[float, float]],
    phase_offset: float,
) -> np.ndarray:
    frames = int(round(duration * sample_rate))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    signal = np.zeros(frames, dtype=np.float64)
    for segment_index, (start, end) in enumerate(segments):
        first = max(0, int(round(start * sample_rate)))
        last = min(frames, int(round(end * sample_rate)))
        if last <= first:
            continue
        local = time[first:last] - start
        phase = phase_offset + segment_index * 0.71
        f0 = 118.0 + 7.0 * np.sin(2.0 * math.pi * 0.45 * local + phase)
        vibrato = 0.025 * np.sin(2.0 * math.pi * 5.2 * local + phase * 0.3)
        fundamental = np.sin(
            2.0 * math.pi * np.cumsum(f0) / sample_rate + phase * 0.2 + vibrato
        )
        formants = (
            0.42 * np.sin(2.0 * math.pi * 620.0 * local + phase)
            + 0.27 * np.sin(2.0 * math.pi * 1180.0 * local + phase * 1.7)
            + 0.14 * np.sin(2.0 * math.pi * 2450.0 * local + phase * 2.3)
        )
        syllables = 0.58 + 0.42 * (
            0.5 + 0.5 * np.sin(2.0 * math.pi * 3.3 * local + phase)
        )
        utterance = (0.62 * fundamental + formants) * syllables
        utterance *= _fade_envelope(utterance.shape[0], sample_rate, 45.0)
        signal[first:last] += 0.29 * utterance
    return signal


def _music_bed(sample_rate: int, duration: float) -> np.ndarray:
    frames = int(round(duration * sample_rate))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    fade = _fade_envelope(frames, sample_rate, 80.0)
    output = np.zeros((frames, 6), dtype=np.float64)
    tones = [
        (110.0, 0.038, 0.0),
        (164.81, 0.030, 0.8),
        (220.0, 0.026, 1.7),
        (329.63, 0.018, 2.9),
    ]
    channel_phase = [0.0, 1.1, 2.2, 0.4, 3.1, 4.2]
    for frequency, amplitude, phase in tones:
        slow = 0.76 + 0.24 * np.sin(2.0 * math.pi * 0.11 * time + phase)
        for channel in range(6):
            if channel == 3:
                continue
            local_phase = phase + channel_phase[channel]
            pan = 1.0 if channel in (0, 2, 4) else 0.86
            output[:, channel] += (
                amplitude
                * pan
                * slow
                * np.sin(2.0 * math.pi * frequency * time + local_phase)
            )
    output[:, 0] += 0.022 * np.sin(2.0 * math.pi * 440.0 * time + 0.6)
    output[:, 1] += 0.022 * np.sin(2.0 * math.pi * 440.0 * time + 1.5)
    output[:, 2] += 0.016 * np.sin(2.0 * math.pi * 330.0 * time + 2.3)
    output[:, 3] = 0.026 * np.sin(2.0 * math.pi * 36.0 * time)
    output[:, 4] += 0.019 * np.sin(2.0 * math.pi * 176.0 * time + 3.7)
    output[:, 5] += 0.019 * np.sin(2.0 * math.pi * 176.0 * time + 4.4)
    return output * fade[:, np.newaxis]


def _effects_bed(sample_rate: int, duration: float, seed: int) -> np.ndarray:
    frames = int(round(duration * sample_rate))
    rng = np.random.default_rng(seed + 9001)
    output = np.zeros((frames, 6), dtype=np.float64)
    bursts = [
        (0.12, 0.34, 0),
        (2.28, 0.38, 1),
        (4.98, 0.42, 2),
        (7.52, 0.46, 3),
    ]
    kernel = np.hanning(127)
    kernel /= np.sum(kernel)
    pan_table = [
        [0.80, 0.15, 0.05, 0.00, 0.55, 0.08],
        [0.08, 0.78, 0.10, 0.00, 0.06, 0.52],
        [0.57, 0.05, 0.22, 0.03, 0.72, 0.08],
        [0.04, 0.60, 0.18, 0.03, 0.09, 0.74],
    ]
    for burst_index, (start, length, kind) in enumerate(bursts):
        first = int(round(start * sample_rate))
        count = int(round(length * sample_rate))
        last = min(frames, first + count)
        if last <= first:
            continue
        local_count = last - first
        local_time = np.arange(local_count, dtype=np.float64) / sample_rate
        noise = rng.standard_normal(local_count)
        noise = np.convolve(noise, kernel, mode="same")
        sweep_start = 1700.0 if burst_index % 2 == 0 else 940.0
        sweep_end = 170.0 if burst_index % 2 == 0 else 2600.0
        frequency = np.linspace(sweep_start, sweep_end, local_count)
        sweep = np.sin(
            2.0 * math.pi * np.cumsum(frequency) / sample_rate
            + burst_index * 0.8
        )
        if kind == 2:
            body = 0.58 * noise + 0.72 * sweep
        else:
            body = 0.82 * noise + 0.46 * sweep
        envelope = np.exp(-3.4 * local_time / max(length, 1.0e-9))
        envelope *= _fade_envelope(local_count, sample_rate, 8.0)
        body *= envelope
        body *= 0.24 / max(1.0e-12, np.max(np.abs(body)))
        for channel in range(6):
            output[first:last, channel] += pan_table[burst_index][channel] * body
    return output


def _config() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "target_sample_rate": 48000,
        "output_layout": "5.1",
        "target_lufs": -23.0,
        "true_peak_limit_dbtp": -1.0,
        "max_relative_gain_db": 4.0,
        "loudness": {
            "tolerance_lu": 0.5,
            "iteration_limit": 10,
        },
        "compression": {
            "enabled": True,
            "threshold_dbfs": -30.0,
            "ratio": 2.5,
            "attack_ms": 15.0,
            "release_ms": 180.0,
            "knee_db": 6.0,
        },
        "dialogue_detection": {
            "frame_ms": 400.0,
            "overlap": 0.75,
            "threshold_dbfs": -45.0,
            "min_segment_ms": 180.0,
        },
        "limiter": {
            "lookahead_ms": 10.0,
            "release_ms": 120.0,
            "oversample": 4,
            "iteration_limit": 10,
            "safety_margin_db": 0.25,
        },
        "output": {
            "basename": "master",
            "container": "mp4",
            "audio_codec": "aac",
            "aac_bitrate": "256k",
            "bits_per_sample": 24,
        },
        "tracks": [
            {
                "file": "dialogue.wav",
                "group": "dialogue",
                "channel_map": ["FC"],
            },
            {
                "file": "music_51.wav",
                "group": "music",
            },
            {
                "file": "effects_51.wav",
                "group": "effects",
            },
        ],
    }


def generate_sample(
    output_dir: str, seed: int = DEFAULT_SEED, duration: float = DEFAULT_DURATION
) -> Dict[str, Any]:
    """Create WAV tracks and config with bit-identical output for a fixed seed."""

    if duration < 2.0:
        raise ValueError("Sample duration must be at least 2 seconds")
    os.makedirs(output_dir, exist_ok=True)
    dialogue_segments = [
        (0.60, 2.00),
        (2.90, 4.80),
        (5.70, 7.40),
        (8.00, 8.65),
    ]
    dialogue_segments = [
        (start, min(end, duration))
        for start, end in dialogue_segments
        if start < duration
    ]
    dialogue = _speech_like(
        44100, duration, dialogue_segments, phase_offset=0.37
    )
    music = _music_bed(48000, duration)
    effects = _effects_bed(32000, duration, seed)

    write_wav(
        os.path.join(output_dir, "dialogue.wav"),
        dialogue[:, np.newaxis],
        44100,
        channel_mask=0,
        bits_per_sample=24,
        extensible=False,
    )
    write_wav(
        os.path.join(output_dir, "music_51.wav"),
        music,
        48000,
        channel_mask=0x3F,
        bits_per_sample=24,
        extensible=True,
    )
    write_wav(
        os.path.join(output_dir, "effects_51.wav"),
        effects,
        32000,
        channel_mask=0,
        bits_per_sample=24,
        extensible=False,
    )
    config = _config()
    with open(
        os.path.join(output_dir, "config.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "seed": seed,
        "duration_sec": duration,
        "dialogue_segments_sec": [
            {"start_sec": start, "end_sec": end}
            for start, end in dialogue_segments
        ],
        "tracks": [
            {
                "file": "dialogue.wav",
                "group": "dialogue",
                "sample_rate": 44100,
                "channels": 1,
                "container": "WAVE_FORMAT_PCM",
            },
            {
                "file": "music_51.wav",
                "group": "music",
                "sample_rate": 48000,
                "channels": 6,
                "container": "WAVE_FORMAT_EXTENSIBLE",
                "channel_mask": "0x0000003f",
            },
            {
                "file": "effects_51.wav",
                "group": "effects",
                "sample_rate": 32000,
                "channels": 6,
                "container": "WAVE_FORMAT_PCM",
            },
        ],
    }
    with open(
        os.path.join(output_dir, "sample_manifest.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinema-sample",
        description="Generate deterministic multirate 5.1 test tracks and config.",
    )
    parser.add_argument("output_dir", help="Input/sample directory to create")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_sample(args.output_dir, args.seed, args.duration)
    except Exception as exc:
        sys.stderr.write(
            json.dumps(
                {
                    "status": "error",
                    "error": {
                        "code": "E_SAMPLE_GENERATION",
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

