"""Generate deterministic local WAV sources and YAML scenes.

Everything is derived from fixed NumPy seeds; running the script repeatedly
produces identical bytes. No network or audio hardware is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def tone(
    sample_rate: int,
    duration: float,
    frequency: float,
    amplitude: float = 0.5,
    channels: int = 1,
    seed: int = 0,
) -> np.ndarray:
    frames = int(round(duration * sample_rate))
    time_axis = np.arange(frames, dtype=np.float64) / sample_rate
    phase = 2.0 * np.pi * frequency * time_axis
    mono = amplitude * np.sin(phase)
    if channels >= 2:
        # Deterministic decorrelated right channel.
        right = amplitude * np.sin(phase + np.pi / 4.0)
        data = np.stack((mono, right), axis=1)
    else:
        data = mono[:, None]
    if channels == 6:
        base = np.tile(data[:, :1], (1, 6))
        shifts = np.linspace(0.0, np.pi / 2.0, 6)
        data = amplitude * np.sin(phase[:, None] + shifts[None, :])
        del base
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    data = data + rng.normal(scale=amplitude * 0.01, size=data.shape)
    return data.astype(np.float32)


def write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="FLOAT")


def generate(root: Path) -> None:
    sample_rate = 48000
    sources = root / "sources"
    write_wav(sources / "tone_mono.wav", tone(sample_rate, 1.0, 440.0, seed=1), sample_rate)
    write_wav(sources / "tone_mono_880.wav", tone(sample_rate, 1.0, 880.0, seed=2), sample_rate)
    write_wav(
        sources / "tone_stereo.wav",
        tone(sample_rate, 1.0, 330.0, channels=2, seed=3),
        sample_rate,
    )
    write_wav(
        sources / "tone_surround.wav",
        tone(sample_rate, 1.0, 220.0, channels=6, seed=4),
        sample_rate,
    )
    # A deliberately small-rate source for mismatch/recovery tests.
    write_wav(
        sources / "tone_24k.wav",
        tone(24000, 0.5, 440.0, seed=5),
        24000,
    )
    # A corrupt file that libsndfile cannot parse.
    broken = sources / "broken.wav"
    broken.write_bytes(b"RIFFnot-a-real-wav-file" * 16)

    scene = {
        "sample_rate": sample_rate,
        "layout": "stereo",
        "global_seed": 20261115,
        "block_size": 4096,
        "duration": 5.0,
        "min_distance": 1.0,
        "trajectory": [
            {"time": 0.0, "position": [0.0, 0.0, 0.0], "yaw": 0.0},
            {"time": 2.0, "position": [2.0, 0.0, 0.0], "yaw": 0.5},
            {"time": 5.0, "position": [5.0, 0.0, 0.0], "yaw": 1.0},
        ],
        "buses": [
            {
                "name": "fx",
                "gain_db": -1.0,
                "limiter": True,
                "delay_seconds": 0.1,
                "delay_feedback": 0.25,
                "route_to": "master",
            },
            {"name": "master", "gain_db": 0.0, "limiter": True},
        ],
        "events": [
            {
                "id": "ev_left",
                "time": 0.2,
                "source": "sources/tone_mono.wav",
                "position": [-3.0, 2.0, 0.0],
                "voice": "voice_a",
                "seed_offset": 11,
                "bus": "master",
                "gain_curve": [
                    {"t": 0.0, "value": -6.0},
                    {"t": 1.0, "value": 0.0},
                ],
                "lowpass_curve": [
                    {"t": 0.0, "value": 500.0},
                    {"t": 1.0, "value": 12000.0},
                ],
            },
            {
                "id": "ev_right",
                "time": 1.5,
                "source": "sources/tone_mono_880.wav",
                "position": [3.0, -2.0, 0.0],
                "voice": "voice_b",
                "seed_offset": 22,
                "bus": "fx",
            },
            {
                "id": "ev_stereo",
                "time": 2.5,
                "source": "sources/tone_stereo.wav",
                "position": [0.0, 4.0, 0.0],
                "seed_offset": 33,
                "bus": "master",
            },
            {
                "id": "ev_broken",
                "time": 3.2,
                "source": "sources/broken.wav",
                "position": [1.0, 1.0, 0.0],
                "seed_offset": 44,
                "bus": "master",
            },
            {
                "id": "ev_rate_mismatch",
                "time": 4.0,
                "source": "sources/tone_24k.wav",
                "position": [1.0, -1.0, 0.0],
                "seed_offset": 55,
                "bus": "master",
            },
        ],
    }
    scene_path = root / "scene_stereo.yaml"
    scene_path.write_text(yaml.safe_dump(scene, sort_keys=False), encoding="utf-8")

    surround = dict(scene)
    surround["layout"] = "5.1"
    surround["events"] = [
        {
            "id": "surround_mono",
            "time": 0.3,
            "source": "sources/tone_mono.wav",
            "position": [-2.0, -3.0, 0.0],
            "seed_offset": 66,
            "bus": "master",
        },
        {
            "id": "surround_downmix",
            "time": 1.4,
            "source": "sources/tone_surround.wav",
            "position": [2.0, 3.0, 0.0],
            "seed_offset": 77,
            "bus": "master",
        },
    ]
    (root / "scene_surround.yaml").write_text(
        yaml.safe_dump(surround, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="fixtures")
    args = parser.parse_args()
    generate(Path(args.root))
    print(f"fixtures generated under {args.root}")


if __name__ == "__main__":
    main()
