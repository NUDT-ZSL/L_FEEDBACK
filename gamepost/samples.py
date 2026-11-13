from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np


def _write_wav(path: Path, sample_rate: int, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(pcm.shape[1])
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.astype("<i2", copy=False).tobytes())


def _deterministic_noise(frames: int, seed: int) -> np.ndarray:
    # NumPy's PCG64 stream is specified and stable for a given NumPy major version.
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.standard_normal(frames)


def generate_sample_set(out_dir: str | Path, *, sample_rate: int = 48_000) -> dict:
    """Create a deterministic WAV/event fixture without network access."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = 12
    frames = seconds * sample_rate
    t = np.arange(frames, dtype=np.float64) / sample_rate
    noise = _deterministic_noise(frames, 0x31A7)
    noise /= max(float(np.max(np.abs(noise))), 1e-12)

    game_left = (
        0.18 * np.sin(2.0 * math.pi * 110.0 * t)
        + 0.08 * np.sin(2.0 * math.pi * 330.0 * t + 0.3)
        + 0.025 * noise
    )
    game_right = (
        0.17 * np.sin(2.0 * math.pi * 110.0 * t + 0.15)
        + 0.07 * np.sin(2.0 * math.pi * 440.0 * t + 0.7)
        + 0.025 * np.roll(noise, 37)
    )

    spoken = np.zeros(frames, dtype=np.float64)
    voice_windows = (
        (0.10, 1.90),
        (2.55, 3.75),
        (4.45, 4.95),
        (5.70, 7.00),
        (7.55, 8.65),
        (9.25, 11.25),
    )
    for start_s, end_s in voice_windows:
        start = int(start_s * sample_rate)
        end = int(end_s * sample_rate)
        local_t = t[start:end] - t[start]
        envelope = np.sin(np.pi * np.linspace(0.0, 1.0, end - start)) ** 0.35
        syllable = 0.55 + 0.45 * np.sin(2.0 * math.pi * 4.2 * local_t) ** 2
        voice = (
            0.38 * np.sin(2.0 * math.pi * 185.0 * local_t)
            + 0.16 * np.sin(2.0 * math.pi * 370.0 * local_t + 0.2)
            + 0.07 * np.sin(2.0 * math.pi * 740.0 * local_t + 0.5)
            + 0.035 * _deterministic_noise(end - start, int(start_s * 1000))
        )
        spoken[start:end] = voice * envelope * syllable

    # Small DC offset exercises the gate's fade requirement.
    spoken += 0.0015

    # Plosive bursts: a short hard-limited run followed by an abrupt release.
    for burst_s in (3.20, 7.15, 10.10):
        start = int(burst_s * sample_rate)
        for index in range(36):
            if start + index < frames:
                spoken[start + index] = 1.0 if index % 2 == 0 else -1.0

    game = np.stack([game_left, game_right], axis=1)
    mic = spoken[:, None]
    game_pcm = np.clip(np.rint(game * 32767.0), -32768, 32767).astype("<i2")
    mic_pcm = np.clip(np.rint(mic * 32767.0), -32768, 32767).astype("<i2")
    mic_pcm[mic_pcm >= 32_767] = 32_767
    mic_pcm[mic_pcm <= -32_767] = -32_767

    events = {
        "events": [
            {"start_ms": 1500, "duration_ms": 600, "category": "kill", "rate": 1.5},
            {"start_ms": 4000, "duration_ms": 1000, "category": "teamfight", "rate": 0.8},
            {"start_ms": 7800, "duration_ms": 500, "category": "crate", "rate": 2.0},
        ]
    }
    invalid_overlap = {
        "events": [
            {"start_ms": 1500, "duration_ms": 1000, "category": "kill", "rate": 1.2},
            {"start_ms": 2000, "duration_ms": 300, "category": "crate", "rate": 1.0},
        ]
    }
    invalid_range = {
        "events": [
            {"start_ms": 11_800, "duration_ms": 500, "category": "teamfight", "rate": 1.0},
        ]
    }
    invalid_category = {
        "events": [
            {"start_ms": 1000, "duration_ms": 200, "category": "boss", "rate": 1.0},
        ]
    }

    _write_wav(out_dir / "game.wav", sample_rate, game_pcm)
    _write_wav(out_dir / "mic.wav", sample_rate, mic_pcm)
    for name, payload in (
        ("events.json", events),
        ("events_overlap.json", invalid_overlap),
        ("events_out_of_range.json", invalid_range),
        ("events_unknown_category.json", invalid_category),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "directory": str(out_dir),
        "game": str(out_dir / "game.wav"),
        "mic": str(out_dir / "mic.wav"),
        "events": str(out_dir / "events.json"),
        "invalid_events": {
            "overlap": str(out_dir / "events_overlap.json"),
            "out_of_range": str(out_dir / "events_out_of_range.json"),
            "unknown_category": str(out_dir / "events_unknown_category.json"),
        },
    }
