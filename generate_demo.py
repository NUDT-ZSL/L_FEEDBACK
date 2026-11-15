#!/usr/bin/env python3
"""Generate a deterministic 48 kHz/24-bit multitrack podcast fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from podcast_post.wavio import write_wav


FS = 48000
TOTAL_SAMPLES = 18 * FS
ASSET_SAMPLES = 900000


def _smooth_gate(phase: np.ndarray, active_samples: int, edge_samples: int) -> np.ndarray:
    attack = np.clip(phase / edge_samples, 0.0, 1.0)
    release = np.clip((active_samples - phase) / edge_samples, 0.0, 1.0)
    return np.minimum(attack, release)


def _host_signal() -> np.ndarray:
    time = np.arange(ASSET_SAMPLES, dtype=np.float64) / FS
    phase = np.mod(np.arange(ASSET_SAMPLES, dtype=np.float64), 1.5 * FS)
    envelope = _smooth_gate(phase, int(0.90 * FS), int(0.025 * FS))
    carriage = (
        0.22 * np.sin(2.0 * np.pi * 173.0 * time)
        + 0.105 * np.sin(2.0 * np.pi * 347.0 * time + 0.4)
        + 0.045 * np.sin(2.0 * np.pi * 719.0 * time + 1.1)
    )
    rng = np.random.default_rng(48001)
    noise = 0.0025 * rng.standard_normal(ASSET_SAMPLES)
    mono = envelope * (carriage + noise)
    return np.column_stack((mono, mono))


def _guest_signal() -> np.ndarray:
    time = np.arange(ASSET_SAMPLES, dtype=np.float64) / FS
    phase = np.mod(np.arange(ASSET_SAMPLES, dtype=np.float64) + 0.37 * FS, 2.1 * FS)
    envelope = _smooth_gate(phase, int(1.22 * FS), int(0.035 * FS))
    carriage = (
        0.15 * np.sin(2.0 * np.pi * 241.0 * time)
        + 0.07 * np.sin(2.0 * np.pi * 463.0 * time + 0.8)
        + 0.03 * np.sin(2.0 * np.pi * 983.0 * time + 0.2)
    )
    rng = np.random.default_rng(48002)
    noise = 0.0018 * rng.standard_normal(ASSET_SAMPLES)
    mono = envelope * (carriage + noise)
    return np.column_stack((mono * 0.96, mono * 1.04))


def _music_signal() -> np.ndarray:
    time = np.arange(ASSET_SAMPLES, dtype=np.float64) / FS
    pulse = 0.92 + 0.08 * np.sin(2.0 * np.pi * 0.125 * time)
    left = pulse * (
        0.070 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.035 * np.sin(2.0 * np.pi * 164.81 * time + 0.5)
        + 0.018 * np.sin(2.0 * np.pi * 220.0 * time + 1.2)
    )
    right = pulse * (
        0.068 * np.sin(2.0 * np.pi * 110.0 * time + 0.08)
        + 0.034 * np.sin(2.0 * np.pi * 164.81 * time + 0.9)
        + 0.017 * np.sin(2.0 * np.pi * 220.0 * time + 1.8)
    )
    rng = np.random.default_rng(48003)
    noise = 0.0012 * rng.standard_normal((ASSET_SAMPLES, 2))
    return np.column_stack((left, right)) + noise


def _manifest() -> dict:
    cut_points = [144000, 288000, 432000, 576000, 720000]

    def clips(track: str, prefix: str) -> list[dict]:
        result = [
            {
                "id": f"{prefix}0",
                "track": track,
                "asset": f"assets/{track}.wav",
                "source_in": 0,
                "source_out": 144000,
                "timeline_in": 0,
                "timeline_out": 144000,
                "gain_db": 0.0,
            }
        ]
        for index, cut in enumerate(cut_points, start=1):
            result.append(
                {
                    "id": f"{prefix}{index}",
                    "track": track,
                    "asset": f"assets/{track}.wav",
                    "source_in": cut,
                    "source_out": cut + 148800,
                    "timeline_in": cut - 4800,
                    "timeline_out": cut + 144000,
                    "gain_db": 0.0,
                }
            )
        return result

    edit_points: list[dict] = []
    for track, prefix in (("host", "h"), ("guest", "g")):
        for index, cut in enumerate(cut_points, start=1):
            edit_points.append(
                {
                    "id": f"{prefix}_cut_{index}",
                    "track": track,
                    "outgoing_clip": f"{prefix}{index - 1}",
                    "incoming_clip": f"{prefix}{index}",
                    "at_sample": cut,
                    "fade_samples": 9600 if (track, cut) == ("host", 576000) else 4800,
                }
            )

    return {
        "version": 1,
        "sample_rate": FS,
        "channels": 2,
        "bit_depth": 24,
        "total_samples": TOTAL_SAMPLES,
        "tracks": [
            {
                "name": "host",
                "role": "host",
                "asset": "assets/host.wav",
                "gain_db": 0.0,
            },
            {
                "name": "guest",
                "role": "guest",
                "asset": "assets/guest.wav",
                "gain_db": -1.5,
            },
            {
                "name": "music",
                "role": "music",
                "asset": "assets/music.wav",
                "gain_db": -9.0,
            },
        ],
        "clips": (
            clips("host", "h")
            + clips("guest", "g")
            + [
                {
                    "id": "m0",
                    "track": "music",
                    "asset": "assets/music.wav",
                    "source_in": 0,
                    "source_out": TOTAL_SAMPLES,
                    "timeline_in": 0,
                    "timeline_out": TOTAL_SAMPLES,
                    "gain_db": 0.0,
                }
            ]
        ),
        "edit_points": edit_points,
        "chapters": [
            {"title": "开场", "start_sample": 0},
            {"title": "场景二", "start_sample": 141600},
            {"title": "场景三", "start_sample": 288000},
            {"title": "场景四", "start_sample": 430800},
            {"title": "场景五", "start_sample": 573600},
            {"title": "场景六", "start_sample": 720000},
        ],
        "processing": {
            "fade_samples": 4800,
            "ducking": {
                "enabled": True,
                "trigger_track": "host",
                "target_track": "music",
                "threshold_dbfs": -28.0,
                "energy_window_ms": 20.0,
                "depth_db": 10.0,
                "attack_ms": 15.0,
                "release_ms": 220.0,
            },
            "loudness": {
                "target_lufs": -16.0,
                "max_true_peak_dbtp": -1.0,
            },
        },
    }


def generate_demo(root: Path) -> Path:
    root = root.resolve()
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    write_wav(assets / "host.wav", _host_signal(), FS, 24)
    write_wav(assets / "guest.wav", _guest_signal(), FS, 24)
    write_wav(assets / "music.wav", _music_signal(), FS, 24)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("demo"),
        help="fixture directory (default: demo)",
    )
    args = parser.parse_args()
    manifest = generate_demo(args.output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

