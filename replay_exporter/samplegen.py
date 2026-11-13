"""Deterministic local sample generation.

All WAV samples are synthesized from a seeded LFSR and pure sine tones, so the
same project inputs always produce byte-identical sources. No network, no
external data sets.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

from .audioio import SAMPLE_RATE, write_mono_wav
from .dsp import ms_to_frames


def _lfsr_noise(length: int, seed: int) -> List[int]:
    state = seed & 0xFFFFFFFF or 1
    values: List[int] = []
    for _ in range(length):
        bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = ((state >> 1) | (bit << 31)) & 0xFFFFFFFF
        values.append(((state >> 16) & 0xFFFF) - 32768)
    return values


def _tone(length: int, frequency: float, amplitude: int, phase: float = 0.0) -> List[int]:
    frames = []
    for index in range(length):
        value = amplitude * math.sin(
            2 * math.pi * frequency * index / SAMPLE_RATE + phase
        )
        frames.append(max(-32768, min(32767, int(round(value)))))
    return frames


def _mix(base: List[int], overlay: List[int]) -> List[int]:
    return [
        max(-32768, min(32767, base[index] + overlay[index]))
        for index in range(len(base))
    ]


def generate_project(
    root: str,
    *,
    segment_count: int = 4,
    segment_ms: int = 1200,
    fade_ms: int = 80,
    gaps_ms: List[int] = None,
    duck_gain_db: float = -12.0,
    loudness_target_dbfs: float = -20.0,
) -> Dict[str, Any]:
    """Create a complete deterministic input project under ``root``."""
    os.makedirs(root, exist_ok=True)
    if gaps_ms is None:
        gaps_ms = [120] * (segment_count - 1)
    if len(gaps_ms) != segment_count - 1:
        raise ValueError("gaps_ms must have segment_count - 1 entries")

    event_types = ["kill", "teamfight", "comeback", "objective"]
    segment_entries = []
    for index in range(segment_count):
        segment_dir = os.path.join(root, f"segment_{index}")
        os.makedirs(segment_dir, exist_ok=True)
        frame_count = ms_to_frames(segment_ms, SAMPLE_RATE)

        # Voice: low-level noise floor + a steady tone, loudest in dialogue.
        voice = _lfsr_noise(frame_count, seed=1000 + index)
        voice = [int(value * 0.08) for value in voice]
        tone_frames = _tone(frame_count, 220.0 + index * 40.0, 12000)
        voice = _mix(voice, tone_frames)
        voice_path = os.path.join(segment_dir, "voice.wav")
        write_mono_wav(voice_path, voice)

        # Ambience: distinct tone + noise so ducking is observable.
        ambience = _lfsr_noise(frame_count, seed=5000 + index)
        ambience = [int(value * 0.25) for value in ambience]
        ambience = _mix(ambience, _tone(frame_count, 110.0 + index * 15.0, 9000))
        ambience_path = os.path.join(segment_dir, "ambience.wav")
        write_mono_wav(ambience_path, ambience)

        dialogue_start = round(segment_ms * 0.25, 3)
        dialogue_end = round(segment_ms * 0.75, 3)
        metadata = {
            "event_type": event_types[index % len(event_types)],
            "voice_wav": os.path.join(f"segment_{index}", "voice.wav"),
            "ambience_wav": os.path.join(f"segment_{index}", "ambience.wav"),
            "duration_ms": float(segment_ms),
            "dialogue_start_ms": dialogue_start,
            "dialogue_end_ms": dialogue_end,
            "target_peak_dbfs": -3.0,
            "fade_ms": float(fade_ms),
        }
        metadata_path = os.path.join(segment_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True, indent=2)
        segment_entries.append(
            {"metadata": os.path.join(f"segment_{index}", "metadata.json")}
        )

    timeline = {
        "version": 1,
        "loudness_target_dbfs": loudness_target_dbfs,
        "gain_smoothing_ms": 40.0,
        "limiter_ceiling_dbfs": -1.0,
        "gaps_ms": [float(value) for value in gaps_ms],
        "ducking": {
            "gain_db": duck_gain_db,
            "attack_ms": 30.0,
            "release_ms": 60.0,
        },
        "segments": segment_entries,
    }
    timeline_path = os.path.join(root, "timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as handle:
        json.dump(timeline, handle, sort_keys=True, indent=2)
    return {"root": os.path.abspath(root), "timeline": os.path.abspath(timeline_path)}
