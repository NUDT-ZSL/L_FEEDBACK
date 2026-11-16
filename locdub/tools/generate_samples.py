"""Deterministically create a complete localisation test fixture.

Running this script produces::

    <out>/
      source.mp4
      timeline.json
      dubs/
        en/line_001.wav ...        (44.1kHz 16-bit, mono and stereo)
        ja/line_001.wav ...

All audio is synthesised from fixed sine/triangle formulas with a seeded
RNG, so repeated runs generate byte-identical WAV content. The timeline
deliberately includes an overlapping segment, a reversed segment and a
missing-WAV case to exercise error handling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from locdub import ffmpeg_utils

SAMPLE_RATE = 44100
LANGUAGES = ("en", "ja")

# id, start_ms, end_ms, spoken_ms, channels, frequency, ratio range
SEGMENTS = [
    ("line_001", 1000, 3500, 2200, 1, 330.0, (0.6, 1.8)),
    ("line_002", 4000, 7000, 3600, 2, 440.0, (0.7, 1.6)),
    ("line_003", 8000, 10500, 1800, 1, 520.0, (0.8, 1.5)),
    ("line_004", 11000, 14000, 2600, 2, 392.0, (0.5, 2.0)),
    ("line_005", 15000, 18000, 4200, 1, 587.0, (0.9, 1.4)),
    ("line_006", 19000, 22000, 2400, 2, 466.0, (0.7, 1.7)),
    ("line_007", 23000, 26000, 3100, 1, 622.0, (0.6, 1.9)),
    ("line_008", 27000, 30000, 2000, 2, 349.0, (0.8, 1.5)),
]
VIDEO_DURATION_MS = 33000

# Deliberate error fixtures:
#   line_bad_overlap overlaps line_008 -> timeline validation skip
#   line_bad_reverse has start > end -> timeline validation skip
#   line_missing exists in the timeline but no WAV is generated -> audio skip
ERROR_SEGMENTS = [
    ("line_bad_overlap", 29000, 31500, 2000, 1, 300.0, (0.5, 2.0)),
    ("line_bad_reverse", 5000, 4000, 1000, 1, 300.0, (0.5, 2.0)),
    ("line_missing", 31500, 32500, 800, 1, 300.0, (0.5, 2.0)),
]


def synth_tone(duration_ms: int, frequency: float, channels: int,
               seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = int(round(SAMPLE_RATE * duration_ms / 1000.0))
    timeline = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    # Slow vibrato + deterministic low-level noise makes the signal non-trivial
    # while staying fully reproducible.
    vibrato = 0.02 * np.sin(2 * math.pi * 5.0 * timeline)
    phase = 2 * math.pi * frequency * timeline + vibrato
    tone = 0.55 * np.sin(phase)
    tone += 0.02 * rng.standard_normal(frames)
    # Simple attack/decay envelope to avoid boundary clicks.
    fade = min(frames, int(SAMPLE_RATE * 0.01))
    envelope = np.ones(frames)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    tone *= envelope
    if channels == 2:
        right = 0.55 * np.sin(phase * 1.005) + 0.02 * rng.standard_normal(frames)
        right *= envelope
        return np.stack((tone, right), axis=1)
    return tone[:, np.newaxis]


def write_wav(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("samples/fixture"))
    parser.add_argument("--scale", type=int, default=1,
                        help="repeat the timeline N times for long/50+ tests.")
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    video = root / "source.mp4"
    scaled_duration = VIDEO_DURATION_MS * args.scale
    ffmpeg_utils.make_solid_video(video, scaled_duration)

    timeline_segments = []
    all_specs = SEGMENTS + ERROR_SEGMENTS
    for repeat in range(args.scale):
        offset = repeat * VIDEO_DURATION_MS
        for index, (segment_id, start, end, spoken, channels,
                    frequency, ratio_range) in enumerate(all_specs):
            unique_id = segment_id if args.scale == 1 else f"{segment_id}_r{repeat}"
            timeline_segments.append({
                "id": unique_id,
                "start_ms": start + offset,
                "end_ms": end + offset,
                "audio": f"{unique_id}.wav",
                "ratio_range": list(ratio_range),
            })
            if segment_id == "line_missing":
                continue
            for lang_index, language in enumerate(LANGUAGES):
                # Each language gets slightly different spoken durations so
                # the adaptation ratio genuinely differs between languages.
                lang_factor = 1.0 if language == "en" else 1.25
                data = synth_tone(
                    int(spoken * lang_factor),
                    frequency * (1.0 + 0.05 * lang_index),
                    channels,
                    seed=index * 100 + lang_index + repeat * 1000,
                )
                write_wav(root / "dubs" / language / f"{unique_id}.wav", data)

    document = {
        "video_duration_ms": scaled_duration,
        "default_ratio_range": [0.5, 2.0],
        "segments": [
            {
                "id": "root_group",
                "segments": timeline_segments,
            }
        ],
    }
    (root / "timeline.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"fixture written to {root}")
    print(f"  video: {video} ({scaled_duration} ms)")
    print(f"  segments: {len(timeline_segments)} x {len(LANGUAGES)} languages")


if __name__ == "__main__":
    main()
