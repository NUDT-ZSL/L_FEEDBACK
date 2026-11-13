"""Deterministic local sample generation (no external assets required)."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

from .config import Config

LANGS = ("zh", "en", "ja")

#: Fixed documented clip durations in milliseconds; keep events in sync.
CLIP_DURATIONS_MS: dict[str, int] = {
    "kill": 420,
    "capture": 800,
    "countdown": 200,
    "victory": 1200,
    "defend": 350,
    "ult": 500,
    "quiet": 600,
}


def _stable_seed(voice_key: str, lang: str, seed: int) -> int:
    digest = 0
    for char in f"{seed}:{voice_key}:{lang}":
        digest = (digest * 131 + ord(char)) & 0xFFFFFFFF
    return digest or 1


def synth_clip(
    voice_key: str,
    lang: str,
    sample_rate: int,
    seed: int,
    amplitude: float,
) -> np.ndarray:
    """Build a deterministic mono float32 clip with stable content per key/lang."""
    duration_ms = CLIP_DURATIONS_MS[voice_key]
    frames = int(sample_rate * duration_ms / 1000)
    rng = np.random.default_rng(_stable_seed(voice_key, lang, seed))
    time_axis = np.arange(frames, dtype=np.float32) / sample_rate

    base_freq = 220.0 + (rng.integers(0, 240))
    tone_a = np.sin(2 * np.pi * base_freq * time_axis, dtype=np.float32)
    tone_b = 0.5 * np.sin(
        2 * np.pi * (base_freq * 1.5) * time_axis, dtype=np.float32
    )
    noise = rng.normal(0.0, 0.05, frames).astype(np.float32)
    signal = (tone_a + tone_b + noise).astype(np.float32)

    # 8 ms raised-cosine fade in/out keeps clip edges click-free.
    edge = int(sample_rate * 0.008)
    window = np.ones(frames, dtype=np.float32)
    if edge * 2 < frames:
        ramp = 0.5 - 0.5 * np.cos(
            np.linspace(0.0, np.pi, edge, dtype=np.float32)
        )
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]

    signal = signal * window
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * amplitude).astype(np.float32)


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm16 = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    sf.write(path, pcm16, sample_rate, subtype="PCM_16", format="WAV")


def generate_sample_assets(
    assets_dir: str,
    seed: int = 20261114,
    config: Config | None = None,
    amplitudes: dict[str, float] | None = None,
) -> list[dict]:
    """Generate every ``{voice_key}_{lang}.wav`` source clip deterministically."""
    config = config or Config()
    amplitudes = amplitudes or {
        "kill": 0.5,
        "capture": 0.45,
        "countdown": 0.4,
        "victory": 0.5,
        "defend": 0.45,
        "ult": 0.45,
        # Above the silence floor but far below target, forcing a +12 dB clamp.
        "quiet": 0.0005,
    }

    created: list[dict] = []
    for voice_key in CLIP_DURATIONS_MS:
        for lang in LANGS:
            samples = synth_clip(
                voice_key,
                lang,
                config.sample_rate,
                seed,
                amplitudes[voice_key],
            )
            path = os.path.join(assets_dir, f"{voice_key}_{lang}.wav")
            _write_wav(path, samples, config.sample_rate)
            created.append(
                {
                    "voice_key": voice_key,
                    "lang": lang,
                    "path": path,
                    "frames": int(samples.shape[0]),
                    "duration_ms": CLIP_DURATIONS_MS[voice_key],
                }
            )
    return created


def demo_events() -> list[dict]:
    """Manifest covering shift, conflict-drop, control and quiet events."""
    return [
        {
            "id": "EVT-001",
            "type": "kill",
            "time_ms": 1000,
            "priority": 5,
            "voice_key": "kill",
            "langs": ["zh", "en", "ja"],
            "max_delay_ms": 900,
            "tone": "aggressive",
        },
        {
            "id": "EVT-002",
            "type": "capture",
            "time_ms": 1050,
            "priority": 9,
            "voice_key": "capture",
            "langs": ["zh", "en", "ja"],
            "max_delay_ms": 500,
            "tone": "focused",
        },
        {
            "id": "EVT-003",
            "type": "ult",
            "time_ms": 1700,
            "priority": 4,
            "voice_key": "ult",
            "langs": ["zh", "en"],
            "max_delay_ms": 80,
        },
        {
            "id": "EVT-004",
            "type": "countdown",
            "time_ms": 2000,
            "priority": 3,
            "tone": "calm",
        },
        {
            "id": "EVT-005",
            "type": "defend",
            "time_ms": 3000,
            "priority": 6,
            "voice_key": "defend",
            "langs": ["zh", "en", "ja"],
        },
        {
            "id": "EVT-006",
            "type": "quiet_mark",
            "time_ms": 5000,
            "priority": 2,
            "voice_key": "quiet",
            "langs": ["zh", "en", "ja"],
        },
        {
            "id": "EVT-007",
            "type": "victory",
            "time_ms": 7000,
            "priority": 10,
            "voice_key": "victory",
            "langs": ["zh", "en", "ja"],
        },
    ]


def write_demo_manifest(path: str, created: list[dict] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated": "deterministic-local",
        "events": demo_events(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
