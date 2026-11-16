from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48000


def write_tone(
    path: Path,
    frequency: float = 440.0,
    amplitude: float = 0.25,
    duration: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(round(duration * sample_rate))) / float(sample_rate)
    samples = amplitude * np.sin(2.0 * np.pi * frequency * time)
    sf.write(
        str(path),
        samples,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )


def write_noise(
    path: Path,
    duration: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 7,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = 0.08 * rng.standard_normal(int(duration * sample_rate))
    sf.write(
        str(path),
        samples,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )


def base_scene(duration_samples: int = 96000) -> Dict[str, Any]:
    return {
        "format": "spatial-scene-v1",
        "sample_rate": SAMPLE_RATE,
        "frame_length": 256,
        "duration_samples": duration_samples,
        "head": {
            "position": [0.0, 0.0, 0.0],
            "quaternion_keys": [
                {"time": 0, "quaternion": [1.0, 0.0, 0.0, 0.0]},
                {
                    "time": duration_samples,
                    "quaternion": [0.9238795325, 0.0, 0.3826834324, 0.0],
                },
            ],
        },
        "sources": [
            {
                "name": "left_tone",
                "file": "left.wav",
                "gain": 0.8,
                "loop": False,
                "start_sample": 0,
                "position_keys": [
                    {"time": 0, "position": [-1.5, 0.0, 2.0]},
                    {
                        "time": duration_samples,
                        "position": [-1.5, 0.0, 2.0],
                    },
                ],
            },
            {
                "name": "moving_loop",
                "file": "moving.wav",
                "gain": 0.35,
                "loop": True,
                "start_sample": 0,
                "position_keys": [
                    {"time": 0, "position": [2.0, 0.2, 6.0]},
                    {
                        "time": duration_samples,
                        "position": [0.5, 0.2, 2.0],
                    },
                ],
            },
        ],
        "occluders": [
            {
                "name": "test_wall",
                "min": [-4.0, -2.0, 3.5],
                "max": [4.0, 2.0, 4.1],
                "attenuation": 0.6,
                "cutoff_hz": 5000.0,
            }
        ],
    }


def write_scene(
    directory: Path,
    scene: Dict[str, Any],
    name: str = "scene.json",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2)
    return path


def make_standard_scene(
    directory: Path,
    duration_samples: int = 96000,
    include_missing: bool = False,
    damaged_track: bool = False,
) -> Path:
    write_tone(directory / "left.wav", frequency=330.0, amplitude=0.30)
    write_tone(directory / "moving.wav", frequency=520.0, amplitude=0.20)
    scene = base_scene(duration_samples)
    if include_missing:
        scene["sources"].append(
            {
                "name": "missing_narration",
                "file": "does_not_exist.wav",
                "gain": 1.0,
                "loop": False,
                "position_keys": [
                    {"time": 0, "position": [0.0, 0.0, 1.0]},
                ],
            }
        )
    if damaged_track:
        damaged = directory / "damaged.wav"
        damaged.write_bytes(b"this is not a wave file")
        scene["sources"].append(
            {
                "name": "damaged_radio",
                "file": "damaged.wav",
                "gain": 1.0,
                "loop": False,
                "position_keys": [
                    {"time": 0, "position": [0.8, 0.0, 1.2]},
                ],
            }
        )
    return write_scene(directory, scene)


def clone_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(scene)
