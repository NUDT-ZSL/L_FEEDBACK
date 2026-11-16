"""Deterministic built-in WAV source synthesizer for demos and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .wavio import encode_wav, sha256_file


def _seconds(frames: int, sample_rate: int) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / float(sample_rate)


def _attack_decay_envelope(
    t: np.ndarray,
    attack_ms: float,
    decay_ms: float,
) -> np.ndarray:
    attack = np.clip(t / max(attack_ms / 1000.0, 1e-9), 0.0, 1.0)
    return attack * np.exp(-t / max(decay_ms / 1000.0, 1e-9))


def _normalize_peak(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    maximum = float(np.max(np.abs(audio), initial=0.0))
    if maximum <= 0.0:
        return audio
    return audio * (peak / maximum)


def synthesize_slash(
    sample_rate: int,
    channels: int,
    *,
    duration_ms: float = 180.0,
) -> np.ndarray:
    frames = int(round(duration_ms * sample_rate / 1000.0))
    t = _seconds(frames, sample_rate)
    rng = np.random.default_rng(0x51A5)
    sweep = 420.0 + (1900.0 - 420.0) * (t / max(t[-1], 1e-9)) ** 1.45
    phase = 2.0 * np.pi * np.cumsum(sweep) / sample_rate
    body = (
        0.62 * np.sin(phase)
        + 0.22 * np.sin(2.02 * phase + 0.4)
        + 0.10 * np.sin(3.91 * phase + 1.1)
    )
    noise = rng.standard_normal(frames)
    # A short moving average acts as a deterministic high-frequency tilt.
    noise = (noise + np.roll(noise, 1) + np.roll(noise, 2)) / 3.0
    envelope = _attack_decay_envelope(t, 2.0, 62.0)
    mono = (body + 0.28 * noise) * envelope
    mono = _normalize_peak(mono, 0.86)
    if channels == 1:
        return mono[:, None]
    left = mono * 0.95
    right = mono * 0.78 + 0.04 * rng.standard_normal(frames) * envelope
    return np.stack([left, right], axis=1)


def synthesize_impact(
    sample_rate: int,
    channels: int,
    *,
    duration_ms: float = 300.0,
) -> np.ndarray:
    frames = int(round(duration_ms * sample_rate / 1000.0))
    t = _seconds(frames, sample_rate)
    rng = np.random.default_rng(0x1A9AC7)
    pitch = 118.0 * np.exp(-t / 0.11) + 54.0
    phase = 2.0 * np.pi * np.cumsum(pitch) / sample_rate
    body = np.sin(phase) * _attack_decay_envelope(t, 1.5, 105.0)
    click = (
        rng.standard_normal(frames)
        * np.exp(-t / 0.018)
        * _attack_decay_envelope(t, 0.2, 28.0)
    )
    ring = (
        0.28 * np.sin(2.0 * np.pi * 683.0 * t + 0.3)
        + 0.18 * np.sin(2.0 * np.pi * 1297.0 * t + 1.2)
    ) * _attack_decay_envelope(t, 2.0, 180.0)
    mono = 0.82 * body + 0.34 * click + ring
    mono = _normalize_peak(mono, 0.9)
    if channels == 1:
        return mono[:, None]
    center = mono
    side = 0.05 * np.sin(2.0 * np.pi * 247.0 * t) * np.exp(-t / 0.09)
    return np.stack([center + side, center - side], axis=1)


def synthesize_tail(
    sample_rate: int,
    channels: int,
    *,
    duration_ms: float = 2200.0,
) -> np.ndarray:
    frames = int(round(duration_ms * sample_rate / 1000.0))
    t = _seconds(frames, sample_rate)
    rng = np.random.default_rng(0x7A11)
    strike = rng.standard_normal(frames) * np.exp(-t / 0.025)
    partials = (1.0, 1.47, 2.09, 2.71, 3.37, 4.19, 5.03)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(partials))
    decays = (1.35, 1.10, 0.90, 0.72, 0.58, 0.47, 0.38)
    mono = np.zeros(frames, dtype=np.float64)
    for ratio, phase, decay in zip(partials, phases, decays):
        frequency = 1730.0 * ratio
        component = np.sin(2.0 * np.pi * frequency * t + phase)
        mono += component * np.exp(-t / decay) / ratio**0.65
    mono = 0.72 * mono + 0.20 * strike
    mono = _normalize_peak(mono, 0.88)
    if channels == 1:
        return mono[:, None]
    flutter = 0.035 * np.sin(2.0 * np.pi * 1.7 * t) * np.exp(-t / 1.2)
    left = mono * (1.0 + flutter)
    right = mono * (0.91 - flutter) + 0.015 * np.sin(2.0 * np.pi * 2173.0 * t)
    right *= np.exp(-t / 2.2)
    return np.stack([left, right], axis=1)


SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "filename": "slash_44k_mono.wav",
        "kind": "slash",
        "sample_rate": 44100,
        "channels": 1,
        "duration_ms": 180.0,
        "bit_depth": "float32",
    },
    {
        "filename": "impact_48k_stereo.wav",
        "kind": "impact",
        "sample_rate": 48000,
        "channels": 2,
        "duration_ms": 300.0,
        "bit_depth": 24,
    },
    {
        "filename": "tail_48k_stereo.wav",
        "kind": "tail",
        "sample_rate": 48000,
        "channels": 2,
        "duration_ms": 2200.0,
        "bit_depth": "float32",
    },
    {
        "filename": "tail_44k_mono.wav",
        "kind": "tail",
        "sample_rate": 44100,
        "channels": 1,
        "duration_ms": 1400.0,
        "bit_depth": 16,
    },
)


def synthesize_sources(output_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        kind = spec["kind"]
        if kind == "slash":
            audio = synthesize_slash(
                spec["sample_rate"],
                spec["channels"],
                duration_ms=spec["duration_ms"],
            )
        elif kind == "impact":
            audio = synthesize_impact(
                spec["sample_rate"],
                spec["channels"],
                duration_ms=spec["duration_ms"],
            )
        elif kind == "tail":
            audio = synthesize_tail(
                spec["sample_rate"],
                spec["channels"],
                duration_ms=spec["duration_ms"],
            )
        else:
            raise ValueError(f"unknown source kind: {kind}")
        path = directory / spec["filename"]
        path.write_bytes(encode_wav(audio, spec["sample_rate"], spec["bit_depth"]))
        results.append(
            {
                "path": path.as_posix(),
                "kind": kind,
                "sample_rate": spec["sample_rate"],
                "channels": spec["channels"],
                "bit_depth": spec["bit_depth"],
                "duration_ms": spec["duration_ms"],
                "sha256": sha256_file(path),
            }
        )
    return results
