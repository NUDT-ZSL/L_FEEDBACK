"""Deterministic local sample and project generator."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, Dict

import numpy as np


SAMPLE_RATE = 48000


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    mono = np.clip(mono, -0.999, 0.999)
    pcm = np.round(mono * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _voice(duration: float, fundamental: float, seed: int) -> np.ndarray:
    time = np.arange(int(round(duration * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    vibrato = 1.0 + 0.006 * np.sin(2.0 * np.pi * 5.1 * time + seed * 0.17)
    phase = 2.0 * np.pi * fundamental * np.cumsum(vibrato) / SAMPLE_RATE
    signal = (
        0.54 * np.sin(phase)
        + 0.23 * np.sin(2.0 * phase + 0.2)
        + 0.13 * np.sin(3.0 * phase + 0.5)
        + 0.06 * np.sin(4.0 * phase + 0.8)
    )
    words = 0.72 + 0.28 * np.sin(2.0 * np.pi * 3.7 * time + seed) ** 2
    attack = np.minimum(1.0, time / 0.035)
    release = np.minimum(1.0, (duration - time) / 0.06)
    envelope = np.clip(attack * release, 0.0, 1.0) * words
    rng = np.random.default_rng(seed)
    breath = rng.normal(0.0, 1.0, time.size)
    kernel = np.ones(41, dtype=np.float64) / 41.0
    breath = np.convolve(breath, kernel, mode="same")
    signal = signal * envelope + 0.025 * breath * envelope
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak * 0.62
    return signal


def _music(
    duration: float,
    root: float,
    seed: int,
    chord_shift: float = 0.0,
) -> np.ndarray:
    time = np.arange(int(round(duration * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    ratios = (1.0, 1.25, 1.5) if chord_shift == 0.0 else (1.0, 1.2, 1.5)
    signal = np.zeros_like(time)
    for index, ratio in enumerate(ratios):
        frequency = root * ratio
        signal += (0.20 - 0.025 * index) * np.sin(
            2.0 * np.pi * frequency * time + seed * 0.11 + index * 0.7
        )
        signal += 0.045 * np.sin(
            2.0 * np.pi * frequency * 2.0 * time + index * 0.3
        )
    pulse = 0.78 + 0.22 * np.sin(2.0 * np.pi * 1.7 * time + seed * 0.23) ** 8
    fade = np.minimum(1.0, time / 0.02) * np.minimum(
        1.0, (duration - time) / 0.02
    )
    signal = signal * pulse * np.clip(fade, 0.0, 1.0)
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak * 0.48
    return signal


def _sfx(duration: float, seed: int) -> np.ndarray:
    time = np.arange(int(round(duration * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    frequency = 700.0 + 900.0 * (time / max(duration, 1e-9)) ** 2
    chirp = 0.30 * np.sin(2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, time.size)
    kernel = np.hanning(129)
    kernel = kernel / np.sum(kernel)
    noise = np.convolve(noise, kernel, mode="same")
    hit_envelope = np.exp(-np.mod(time, 0.42) * 30.0)
    signal = chirp * hit_envelope + 0.16 * noise * hit_envelope
    fade = np.minimum(1.0, time / 0.01) * np.minimum(
        1.0, (duration - time) / 0.04
    )
    signal *= np.clip(fade, 0.0, 1.0)
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal = signal / peak * 0.5
    return signal


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate_valid(root: Path) -> Path:
    project_dir = root / "valid"
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_specs = {
        "voice_zh.wav": _voice(1.0, 185.0, 11),
        "voice_en.wav": _voice(0.8, 225.0, 23),
        "voice_ja.wav": _voice(0.75, 260.0, 37),
        "voice_cn2.wav": _voice(0.9, 195.0, 41),
        "music_a.wav": _music(4.0, 220.0, 53),
        "music_b.wav": _music(5.5, 196.0, 67, chord_shift=0.03),
        "sfx_a.wav": _sfx(1.6, 79),
    }
    for name, samples in audio_specs.items():
        _write_pcm16(audio_dir / name, samples, SAMPLE_RATE)

    project = {
        "schema_version": 1,
        "name": "deterministic_dialogue_demo",
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": 9.0,
        "tracks": [
            {"id": "dialogue", "kind": "dialogue", "gain": 1.0},
            {"id": "music", "kind": "music", "gain": 0.8},
            {"id": "sfx", "kind": "sfx", "gain": 0.7},
        ],
        "clips": [
            {
                "id": "voice_zh",
                "track": "dialogue",
                "file": "audio/voice_zh.wav",
                "start": 1.0,
                "gain": 1.0,
            },
            {
                "id": "voice_en",
                "track": "dialogue",
                "file": "audio/voice_en.wav",
                "start": 4.0,
                "gain": 1.0,
            },
            {
                "id": "voice_ja",
                "track": "dialogue",
                "file": "audio/voice_ja.wav",
                "start": 4.5,
                "gain": 0.95,
            },
            {
                "id": "voice_cn2",
                "track": "dialogue",
                "file": "audio/voice_cn2.wav",
                "start": 7.0,
                "gain": 0.95,
            },
            {
                "id": "music_a",
                "track": "music",
                "file": "audio/music_a.wav",
                "start": 0.0,
                "gain": 0.95,
            },
            {
                "id": "music_b",
                "track": "music",
                "file": "audio/music_b.wav",
                "start": 3.5,
                "gain": 0.92,
            },
            {
                "id": "sfx_a",
                "track": "sfx",
                "file": "audio/sfx_a.wav",
                "start": 2.0,
                "gain": 0.9,
            },
        ],
        "dialogues": [
            {
                "id": "line_zh_01",
                "start": 1.0,
                "end": 2.2,
                "stretch_ratio": 1.2,
                "voice_clip_id": "voice_zh",
                "duck_tracks": ["music", "sfx"],
            },
            {
                "id": "line_en_02",
                "start": 4.0,
                "end": 5.0,
                "stretch_ratio": 1.25,
                "voice_clip_id": "voice_en",
                "duck_tracks": ["music", "sfx"],
            },
            {
                "id": "line_ja_03",
                "start": 4.5,
                "end": 5.5,
                "stretch_ratio": 1.3333333333,
                "voice_clip_id": "voice_ja",
                "duck_tracks": ["music", "sfx"],
            },
            {
                "id": "line_cn_04",
                "start": 7.0,
                "end": 7.9,
                "stretch_ratio": 1.0,
                "voice_clip_id": "voice_cn2",
                "duck_tracks": ["music", "sfx"],
            },
        ],
        "ducking": {
            "enabled": True,
            "target_tracks": ["music", "sfx"],
            "depth_db": -12.0,
            "attack_seconds": 0.06,
            "release_seconds": 0.35,
            "attack_curve": "cosine",
            "release_curve": "cosine",
        },
        "crossfades": [
            {
                "from_clip_id": "music_a",
                "to_clip_id": "music_b",
                "duration_seconds": 0.5,
            }
        ],
        "normalization": {"peak_ceiling_dbfs": -0.2},
        "output": {
            "audio": "output/demo_mix.wav",
            "index": "output/track_index.json",
            "log": "output/run_log.json",
        },
    }
    project_path = project_dir / "project.json"
    _write_json(project_path, project)
    return project_path


def generate_invalid(root: Path) -> Path:
    project_dir = root / "invalid"
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    _write_pcm16(audio_dir / "voice_reused.wav", _voice(1.0, 210.0, 101), SAMPLE_RATE)
    _write_pcm16(audio_dir / "sfx_ok.wav", _sfx(1.6, 103), SAMPLE_RATE)

    project = {
        "schema_version": 1,
        "name": "deterministic_partial_failure_demo",
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": 6.0,
        "tracks": [
            {"id": "dialogue", "kind": "dialogue", "gain": 0.95},
            {"id": "music", "kind": "music", "gain": 0.5},
            {"id": "sfx", "kind": "sfx", "gain": 0.5},
        ],
        "clips": [
            {
                "id": "voice_reused",
                "track": "dialogue",
                "file": "audio/voice_reused.wav",
                "start": 1.0,
                "gain": 0.9,
            },
            {
                "id": "missing_music",
                "track": "music",
                "file": "audio/does_not_exist.wav",
                "start": 0.0,
                "gain": 0.8,
            },
            {
                "id": "sfx_ok",
                "track": "sfx",
                "file": "audio/sfx_ok.wav",
                "start": 3.0,
                "gain": 0.7,
            },
        ],
        "dialogues": [
            {
                "id": "line_first",
                "start": 1.0,
                "end": 2.3,
                "stretch_ratio": 1.3,
                "voice_clip_id": "voice_reused",
                "duck_tracks": ["music", "sfx", "ghost_track"],
            },
            {
                "id": "line_conflict",
                "start": 1.6,
                "end": 2.6,
                "stretch_ratio": 1.0,
                "voice_clip_id": "voice_reused",
                "duck_tracks": ["music", "sfx"],
            },
        ],
        "ducking": {
            "enabled": True,
            "target_tracks": ["music", "sfx"],
            "depth_db": -12.0,
            "attack_seconds": 0.05,
            "release_seconds": 0.25,
            "attack_curve": "cosine",
            "release_curve": "cosine",
        },
        "crossfades": [],
        "normalization": {"peak_ceiling_dbfs": 0.0},
        "output": {
            "audio": "output/partial_mix.wav",
            "index": "output/track_index.json",
            "log": "output/run_log.json",
        },
    }
    project_path = project_dir / "project.json"
    _write_json(project_path, project)
    return project_path


def generate_fatal(root: Path) -> Path:
    project_dir = root / "fatal"
    project_dir.mkdir(parents=True, exist_ok=True)
    stale_audio = project_dir / "output" / "should_not_exist.wav"
    if stale_audio.exists():
        stale_audio.unlink()
    project = {
        "schema_version": 1,
        "name": "invalid_sample_rate_demo",
        "sample_rate": 44100.5,
        "duration_seconds": 3.0,
        "tracks": [{"id": "music", "kind": "music", "gain": 1.0}],
        "clips": [],
        "dialogues": [],
        "ducking": {"enabled": True, "target_tracks": ["music"]},
        "crossfades": [],
        "normalization": {"peak_ceiling_dbfs": 0.0},
        "output": {
            "audio": "output/should_not_exist.wav",
            "index": "output/track_index.json",
            "log": "output/run_log.json",
        },
    }
    project_path = project_dir / "project.json"
    _write_json(project_path, project)
    return project_path


def generate_all(root: Path) -> Dict[str, Path]:
    root = Path(root).expanduser().resolve()
    return {
        "valid": generate_valid(root),
        "invalid": generate_invalid(root),
        "fatal": generate_fatal(root),
    }

