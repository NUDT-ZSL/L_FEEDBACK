"""Deterministic local WAV/project fixtures shared by tests and acceptance."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np


def tone(
    duration: float,
    sample_rate: int,
    channels: int,
    frequency: float,
    amplitude: float,
    *,
    fade: float = 0.01,
) -> np.ndarray:
    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * time)
    fade_count = min(int(round(fade * sample_rate)), count // 2)
    if fade_count:
        ramp = np.linspace(0.0, 1.0, fade_count, endpoint=False)
        mono[:fade_count] *= ramp
        mono[-fade_count:] *= ramp[::-1]
    if channels == 1:
        return mono[:, None]
    second_frequency = frequency * 1.25
    right = amplitude * np.sin(2.0 * np.pi * second_frequency * time)
    if fade_count:
        right[:fade_count] *= ramp
        right[-fade_count:] *= ramp[::-1]
    return np.column_stack([mono, right])


def write_wav(path: Path, samples: np.ndarray, sample_rate: int, bit_depth: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = int(samples.shape[1])
    clipped = np.clip(samples, -1.0, 1.0)
    max_value = (1 << (bit_depth - 1)) - 1
    quantized = np.rint(clipped * max_value).astype(np.int64)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(bit_depth // 8)
        handle.setframerate(sample_rate)
        if bit_depth == 16:
            payload = quantized.astype("<i2").tobytes()
        elif bit_depth == 24:
            unsigned = (quantized & 0xFFFFFF).astype(np.uint32).reshape(-1)
            packed = np.empty((unsigned.size, 3), dtype=np.uint8)
            packed[:, 0] = (unsigned & 0xFF).astype(np.uint8)
            packed[:, 1] = ((unsigned >> 8) & 0xFF).astype(np.uint8)
            packed[:, 2] = ((unsigned >> 16) & 0xFF).astype(np.uint8)
            payload = packed.tobytes()
        else:
            raise ValueError("fixture writer supports 16 or 24-bit PCM")
        handle.writeframes(payload)


def create_project(root: Path) -> dict[str, Path]:
    media = root / "media"
    media.mkdir(parents=True, exist_ok=True)

    sample_1 = media / "dialog_01_44k1_mono16.wav"
    sample_2 = media / "dialog_02_48k_stereo24.wav"
    sample_3 = media / "dialog_03_48k_mono24.wav"
    sample_4 = media / "dialog_04_range_error.wav"
    sample_5 = media / "dialog_05_overlap_stereo24.wav"

    write_wav(sample_1, tone(1.40, 44_100, 1, 330.0, 0.34), 44_100, 16)
    write_wav(
        sample_2,
        tone(1.52, 48_000, 2, 440.0, 0.66),
        48_000,
        24,
    )
    write_wav(sample_3, tone(1.48, 48_000, 1, 660.0, 0.36), 48_000, 24)
    write_wav(sample_4, tone(2.20, 48_000, 1, 220.0, 0.30), 48_000, 16)
    write_wav(
        sample_5,
        tone(1.00, 48_000, 2, 880.0, 0.82),
        48_000,
        24,
    )

    srt_path = root / "dialogue.srt"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "First line\n\n"
        "2\n"
        "00:00:00,050 --> 00:00:01,550\n"
        "Second line\n\n"
        "3\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Third line\n\n"
        "4\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Fourth line (audio skipped)\n\n"
        "5\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Overlapping fifth line\n",
        encoding="utf-8",
    )

    timeline = {
        "schema_version": 1,
        "srt": "dialogue.srt",
        "total_duration": "00:00:12.000",
        "audio": {
            "sample_rate": 48000,
            "bit_depth": 24,
            "channels": 2,
            "target_lufs": -23.0,
            "max_true_peak_dbtp": -1.0,
        },
        "output": {
            "channels": 2,
            "language": "eng",
            "subtitle_language": "eng",
            "audio_filename": "dialogue_mix.wav",
            "subtitle_filename": "dialogue_mix.srt",
            "container_filename": "dialogue_mix.mkv",
            "report_filename": "report.json",
            "title": "Deterministic Dialogue Acceptance",
        },
        "dialogue_entries": [
            {
                "id": "stretch_44k1_mono16",
                "source": "media/dialog_01_44k1_mono16.wav",
                "source_start": "00:00:00.000",
                "target_start": "00:00:01.000",
                "subtitle_index": 1,
                "weight": 1.0,
            },
            {
                "id": "crop_48k_stereo24",
                "source": "media/dialog_02_48k_stereo24.wav",
                "source_start": "00:00:00.050",
                "target_start": "00:00:04.000",
                "subtitle_index": 2,
                "weight": 1.0,
            },
            {
                "id": "pad_48k_mono24",
                "source": "media/dialog_03_48k_mono24.wav",
                "source_start": "00:00:00.000",
                "target_start": "00:00:07.000",
                "subtitle_index": 3,
                "weight": 1.0,
            },
            {
                "id": "range_error_skip",
                "source": "media/dialog_04_range_error.wav",
                "source_start": "00:00:00.000",
                "target_start": "00:00:09.500",
                "subtitle_index": 4,
                "weight": 1.0,
            },
            {
                "id": "overlap_clip_stereo24",
                "source": "media/dialog_05_overlap_stereo24.wav",
                "source_start": "00:00:00.000",
                "target_start": "00:00:04.500",
                "subtitle_index": 5,
                "weight": 1.0,
            },
        ],
    }
    timeline_path = root / "timeline.json"
    timeline_path.write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "timeline": timeline_path,
        "srt": srt_path,
        "sample_1": sample_1,
        "sample_2": sample_2,
        "sample_3": sample_3,
        "sample_4": sample_4,
        "sample_5": sample_5,
    }

