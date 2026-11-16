"""Deterministic sample-project generation for the test-suite.

All audio is synthesized with a fixed NumPy seed and written as plain PCM WAV
through the standard library ``wave`` module, so no external assets are
required. Helpers also produce a truncated/corrupt source and a missing-source
scenario.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

SEED = 20261116


def write_pcm_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    peak = 2.0 ** (bit_depth - 1)
    integer = np.clip(np.rint(audio * peak), -peak, peak - 1).astype("<i4")
    if bit_depth == 16:
        payload = integer.astype("<i2").tobytes()
    elif bit_depth == 32:
        payload = integer.astype("<i4").tobytes()
    else:
        triples = np.zeros((integer.reshape(-1).shape[0], 3), dtype=np.uint8)
        flat = integer.reshape(-1).astype(np.uint32)
        triples[:, 0] = flat & 0xFF
        triples[:, 1] = (flat >> 8) & 0xFF
        triples[:, 2] = (flat >> 16) & 0xFF
        payload = triples.tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(audio.shape[1])
        handle.setsampwidth(bit_depth // 8)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return path


def tone_track(
    rng: np.random.Generator,
    seconds: float,
    sample_rate: int,
    frequency: float,
    channels: int = 1,
    amplitude: float = 0.4,
) -> np.ndarray:
    length = int(round(seconds * sample_rate))
    timeline = np.arange(length, dtype=np.float64) / sample_rate
    # Deterministic low-level noise plus an amplitude envelope keeps the
    # compressor/loudness stages exercised on non-constant signals.
    envelope = 0.75 + 0.25 * np.sin(2.0 * np.pi * 2.0 * timeline)
    signal = amplitude * envelope * np.sin(2.0 * np.pi * frequency * timeline)
    signal += 0.01 * rng.standard_normal(length)
    if channels == 1:
        return signal[:, np.newaxis]
    stereo = np.column_stack([signal, 0.8 * signal])
    return stereo


def corrupt_wav(source: Path, target: Path) -> Path:
    """Truncate a WAV mid-data; the header still advertises full length."""
    raw = source.read_bytes()
    target.write_bytes(raw[: max(1, len(raw) // 2)])
    return target


def write_garbage_file(path: Path, size: int = 256) -> Path:
    rng = np.random.default_rng(SEED + 7)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rng.integers(0, 256, size=size, dtype=np.uint8).tobytes())
    return path


def build_sample_project(root: Path, *, corrupt: bool = False, missing: bool = False) -> Path:
    """Create a three-track, mixed-rate sample project under ``root``."""
    root = Path(root)
    samples_dir = root / "samples"
    rng = np.random.default_rng(SEED)

    # Voice track at 44.1 kHz mono (needs resampling to the 48 kHz project).
    voice = tone_track(rng, 1.2, 44100, 440.0, channels=1, amplitude=0.45)
    write_pcm_wav(samples_dir / "voice.wav", voice, 44100)

    # Ambience at 32 kHz stereo.
    ambience = tone_track(rng, 1.5, 32000, 220.0, channels=2, amplitude=0.3)
    write_pcm_wav(samples_dir / "ambience.wav", ambience, 32000)

    # Music at 48 kHz stereo, native project rate.
    music = tone_track(rng, 2.0, 48000, 110.0, channels=2, amplitude=0.35)
    write_pcm_wav(samples_dir / "music.wav", music, 48000)

    if corrupt:
        corrupt_wav(samples_dir / "music.wav", samples_dir / "music_bad.wav")
    if missing:
        # The configured path simply never gets created.
        pass

    music_source = "samples/music_bad.wav" if corrupt else (
        "samples/does_not_exist.wav" if missing else "samples/music.wav"
    )

    project = {
        "project_id": "cutscene-demo",
        "sample_rate": 48000,
        "channels": 2,
        "duration_ms": 2000,
        "target_lufs": -23.0,
        "output": {"path": "out/mix.wav", "format": "wav", "bit_depth": 16},
        "compressor": {
            "threshold_db": -18.0,
            "ratio": 4.0,
            "knee_db": 6.0,
            "attack_ms": 5.0,
            "release_ms": 50.0,
            "lookahead_ms": 0.0,
        },
        "tracks": [
            {
                "track_id": "voice",
                "source": "samples/voice.wav",
                "gain_db": -1.0,
                "muted": False,
                "events": [
                    {"event_id": "v1", "start_ms": 0, "duration_ms": 800, "gain_db": 0.0},
                    # Millisecond-level overlap with v1.
                    {"event_id": "v2", "start_ms": 750, "duration_ms": 500, "gain_db": -2.0},
                ],
            },
            {
                "track_id": "ambience",
                "source": "samples/ambience.wav",
                "gain_db": 0.0,
                "muted": False,
                "events": [
                    {"event_id": "a1", "start_ms": 100, "duration_ms": 1800, "gain_db": -3.0},
                ],
            },
            {
                "track_id": "music",
                "source": music_source,
                "gain_db": -2.0,
                "muted": False,
                "events": [
                    {"event_id": "m1", "start_ms": 200, "duration_ms": 1500, "gain_db": 0.0},
                ],
            },
        ],
    }
    project_path = root / "project.json"
    project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    return project_path
