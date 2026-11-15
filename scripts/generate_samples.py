#!/usr/bin/env python3
"""Generate deterministic local audio timelines without external datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _envelope(t: np.ndarray, duration: float, attack: float = 0.025, release: float = 0.06) -> np.ndarray:
    return np.minimum(
        1.0,
        np.minimum(t / max(attack, 1e-9), (duration - t) / max(release, 1e-9)),
    ).clip(0.0, 1.0)


def _dialogue(sample_rate: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(sample_rate * duration)), dtype=np.float64) / sample_rate
    env = _envelope(t, duration)
    syllable = 0.58 + 0.42 * np.sin(2.0 * np.pi * 3.2 * t + 0.7) ** 2
    frequency = 175.0 + 38.0 * np.sin(2.0 * np.pi * 0.43 * t + seed)
    carrier = (
        0.62 * np.sin(2.0 * np.pi * frequency * t)
        + 0.25 * np.sin(2.0 * np.pi * frequency * 2.01 * t + 0.4)
        + 0.13 * np.sin(2.0 * np.pi * frequency * 3.98 * t + 1.1)
    )
    breath = rng.standard_normal(t.size) * 0.035
    return ((carrier + breath) * env * syllable * 0.34).astype(np.float32)


def _ambience(sample_rate: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(sample_rate * duration)), dtype=np.float64) / sample_rate
    slow = 0.55 + 0.2 * np.sin(2.0 * np.pi * 0.11 * t)
    left = (
        0.07 * np.sin(2.0 * np.pi * 93.0 * t)
        + 0.035 * np.sin(2.0 * np.pi * 147.0 * t + 0.4)
        + 0.018 * rng.standard_normal(t.size)
    ) * slow
    right = (
        0.065 * np.sin(2.0 * np.pi * 97.0 * t + 0.2)
        + 0.03 * np.sin(2.0 * np.pi * 151.0 * t)
        + 0.018 * rng.standard_normal(t.size)
    ) * slow
    return np.column_stack((left, right)).astype(np.float32)


def _sfx(sample_rate: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = int(round(sample_rate * duration))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    env = np.exp(-t * 7.0)
    noise = rng.standard_normal(frames)
    low = np.sin(2.0 * np.pi * (220.0 * t + 560.0 * t * t))
    return ((0.55 * noise + 0.45 * low) * env * 0.32).astype(np.float32)


def _write(path: Path, data: np.ndarray, sample_rate: int, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype=subtype)


def make_demo(output: Path) -> dict[str, object]:
    assets = output / "assets"
    rate = 48_000
    dialogue_a = _dialogue(44_100, 3.2, 101)
    dialogue_b = _dialogue(48_000, 2.6, 202)
    ambience = _ambience(22_050, 7.4, 303)
    sfx = _sfx(48_000, 1.2, 404)
    _write(assets / "dialogue_a_44100_mono.wav", dialogue_a, 44_100)
    _write(
        assets / "dialogue_b_48000_stereo.wav",
        np.column_stack((dialogue_b, np.roll(dialogue_b, 19) * 0.97)),
        48_000,
    )
    _write(assets / "ambience_22050_stereo.wav", ambience, 22_050)
    _write(assets / "sfx_48000_mono.wav", sfx, 48_000)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": {
            "name": "deterministic-demo",
            "version": "v1",
            "segment_id": "scene-01",
            "sample_rate": rate,
            "duration_ms": 8_000,
            "channels": 2,
        },
        "loudness": {"target_lufs": -16.0, "true_peak_db": -1.0},
        "dialogue_ducking": {"duck_db": -12.0, "safe_headroom_db": -0.5},
        "tracks": [
            {
                "id": "dialogue_a",
                "path": "assets/dialogue_a_44100_mono.wav",
                "sample_rate": 44_100,
                "channels": 1,
                "start_ms": 300.0,
                "gain_db": -2.0,
                "role": "dialogue",
                "priority": 20,
            },
            {
                "id": "dialogue_b",
                "path": "assets/dialogue_b_48000_stereo.wav",
                "sample_rate": 48_000,
                "channels": 2,
                "start_ms": 1_200.0,
                "gain_db": -3.0,
                "role": "dialogue",
                "priority": 10,
            },
            {
                "id": "ambience",
                "path": "assets/ambience_22050_stereo.wav",
                "sample_rate": 22_050,
                "channels": 2,
                "start_ms": 0.0,
                "gain_db": -16.0,
                "role": "ambience",
                "priority": 0,
            },
            {
                "id": "sfx",
                "path": "assets/sfx_48000_mono.wav",
                "sample_rate": 48_000,
                "channels": 1,
                "start_ms": 4_200.0,
                "gain_db": -5.0,
                "role": "sfx",
                "priority": 0,
            },
        ],
        "output": {
            "master": "out/master.wav",
            "report": "out/timing_report.json",
            "sample_format": "PCM_24",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_long_track(
    path: Path,
    duration_seconds: float,
    kind: str,
    seed: int,
    sample_rate: int = 48_000,
) -> None:
    frames = int(round(duration_seconds * sample_rate))
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        str(path),
        mode="w",
        samplerate=sample_rate,
        channels=2,
        format="WAV",
        subtype="PCM_24",
    ) as handle:
        block_frames = 65_536
        for start in range(0, frames, block_frames):
            count = min(block_frames, frames - start)
            t = (start + np.arange(count, dtype=np.float64)) / sample_rate
            if kind == "dialogue":
                env = 0.45 + 0.35 * np.sin(2.0 * np.pi * 2.7 * t + seed) ** 2
                mono = (
                    0.16 * np.sin(2.0 * np.pi * (178.0 + seed % 41) * t)
                    + 0.07 * np.sin(2.0 * np.pi * (356.0 + seed % 17) * t + 0.3)
                    + 0.015 * rng.standard_normal(count)
                ) * env
                block = np.column_stack((mono, np.roll(mono, 13) * 0.96))
            elif kind == "ambience":
                block = np.column_stack(
                    (
                        0.09 * np.sin(2.0 * np.pi * 83.0 * t)
                        + 0.01 * rng.standard_normal(count),
                        0.085 * np.sin(2.0 * np.pi * 89.0 * t + 0.5)
                        + 0.01 * rng.standard_normal(count),
                    )
                )
            else:
                impulses = np.zeros(count, dtype=np.float64)
                impulses[:: max(1, sample_rate // 3)] = 0.5
                block = np.column_stack(
                    (
                        impulses + 0.03 * rng.standard_normal(count),
                        impulses * 0.9 + 0.03 * rng.standard_normal(count),
                    )
                )
            handle.write(block.astype(np.float32))


def make_performance(output: Path) -> dict[str, object]:
    assets = output / "assets"
    duration_seconds = 300.0
    definitions = [
        ("dialogue_1", "dialogue", 0.4, -10.0, 30, 501),
        ("dialogue_2", "dialogue", 2.0, -11.0, 20, 502),
        ("dialogue_3", "dialogue", 4.2, -11.5, 10, 503),
        ("ambience", "ambience", 0.0, -24.0, 0, 504),
        ("sfx_1", "sfx", 9.0, -18.0, 0, 505),
        ("sfx_2", "sfx", 27.0, -19.0, 0, 506),
        ("sfx_3", "sfx", 61.0, -20.0, 0, 507),
        ("music_bed", "music", 0.0, -21.0, 0, 508),
    ]
    tracks: list[dict[str, object]] = []
    for track_id, kind, start_seconds, gain_db, priority, seed in definitions:
        path = assets / f"{track_id}_48000_stereo.wav"
        _write_long_track(path, duration_seconds, kind, seed)
        tracks.append(
            {
                "id": track_id,
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sample_rate": 48_000,
                "channels": 2,
                "start_ms": start_seconds * 1_000.0,
                "gain_db": gain_db,
                "role": kind,
                "priority": priority,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": {
            "name": "eight-track-performance",
            "version": "perf-v1",
            "segment_id": "five-minutes",
            "sample_rate": 48_000,
            "duration_ms": 365_000,
            "channels": 2,
        },
        "loudness": {"target_lufs": -16.0, "true_peak_db": -1.0},
        "dialogue_ducking": {"duck_db": -10.0, "safe_headroom_db": -0.5},
        "tracks": tracks,
        "output": {
            "master": "out/master.wav",
            "report": "out/timing_report.json",
            "sample_format": "PCM_24",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", choices=("demo", "performance"), default="demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preset == "demo":
        make_demo(args.output)
    else:
        make_performance(args.output)
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
