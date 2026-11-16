"""Create deterministic local WAV assets and scene descriptions for cinemix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import soundfile as sf


def _tone(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float,
    phase: float = 0.0,
) -> np.ndarray:
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * time + phase)


def _event(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate
    envelope = np.exp(-time / max(duration * 0.32, 1e-6))
    attack = np.minimum(1.0, time / 0.008)
    tone = np.sin(2.0 * np.pi * frequency * time + 0.2 * np.sin(2.0 * np.pi * 3.0 * time))
    noise = rng.normal(0.0, 0.16, time.shape)
    return amplitude * envelope * attack * (0.82 * tone + 0.18 * noise)


def _ambience(duration: float, sample_rate: int, seed: int, stereo: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate
    slow = 0.028 * np.sin(2.0 * np.pi * 0.11 * time + 0.4)
    air = rng.normal(0.0, 0.023, time.shape)
    hum = 0.008 * np.sin(2.0 * np.pi * 73.0 * time + 0.1)
    base = slow + air + hum
    if stereo:
        side = rng.normal(0.0, 0.012, time.shape)
        return np.column_stack((base + side, base - side))
    return base


def _music(
    duration: float,
    sample_rate: int,
    seed: int,
    frequencies: Sequence[float],
    amplitude: float,
    stereo: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(round(duration * sample_rate)), dtype=np.float64) / sample_rate
    values = np.zeros_like(time)
    for index, frequency in enumerate(frequencies):
        values += (
            np.sin(2.0 * np.pi * frequency * time + index * 0.31)
            / float(index + 1)
        )
    values += 0.04 * rng.normal(0.0, 1.0, time.shape)
    envelope = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.17 * time + 1.2)
    values *= amplitude * envelope
    if not stereo:
        return values
    left = values * (0.92 + 0.08 * np.sin(2.0 * np.pi * 0.09 * time))
    right = values * (0.92 + 0.08 * np.cos(2.0 * np.pi * 0.07 * time))
    return np.column_stack((left, right))


def _write(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float64), sample_rate, subtype=subtype)


def _write_scene(directory: Path, scene: Dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "scene.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _scene_forest(root: Path) -> None:
    directory = root / "forest"
    sample_rate = 48000
    _write(
        directory / "ambience.wav",
        _ambience(5.0, sample_rate, 101, stereo=False),
        sample_rate,
        "PCM_16",
    )
    _write(
        directory / "bird.wav",
        _event(1450.0, 0.45, sample_rate, 0.42, 102),
        sample_rate,
        "PCM_16",
    )
    _write(
        directory / "branch.wav",
        _event(310.0, 0.30, sample_rate, 0.34, 103),
        sample_rate,
        "PCM_24",
    )
    _write(
        directory / "silence_pad.wav",
        np.zeros((2400, 1), dtype=np.float64),
        sample_rate,
        "PCM_16",
    )
    _write(
        directory / "music_forest.wav",
        _music(3.8, sample_rate, 104, (220.0, 277.18, 329.63), 0.12, stereo=True),
        sample_rate,
        "PCM_24",
    )
    _write_scene(
        directory,
        {
            "scene_id": "forest",
            "duration": 8.0,
            "ambience": {
                "file": "ambience.wav",
                "loop_start": 0.35,
                "loop_end": 4.65,
                "gain": 0.75,
            },
            "samples": {
                "bird": "bird.wav",
                "branch": "branch.wav",
                "silence_pad": "silence_pad.wav",
            },
            "events": [
                {"time": 1.125, "sample": "bird", "gain": 0.65},
                {"time": 3.5, "sample": "branch", "gain": 0.55},
                {"time": 6.25, "sample": "bird", "gain": 0.42},
                {"time": 7.7, "sample": "silence_pad", "gain": 1.0},
            ],
            "music_layers": [
                {
                    "file": "music_forest.wav",
                    "start": 0.5,
                    "loops": 2,
                    "gain": 0.38,
                }
            ],
            "dialogue_intervals": [
                {"start": 2.0, "end": 4.4, "duck_depth_db": 10.0},
                {"start": 6.0, "end": 7.1, "duck_depth_db": 6.0},
            ],
            "camera_cuts": [0.75, 3.75, 5.5],
        },
    )


def _scene_city(root: Path) -> None:
    directory = root / "city"
    sample_rate = 44100
    _write(
        directory / "ambience_44k.wav",
        _ambience(4.4, sample_rate, 201, stereo=True),
        sample_rate,
        "PCM_24",
    )
    _write(
        directory / "horn_16.wav",
        _event(420.0, 0.65, 32000, 0.33, 202),
        32000,
        "PCM_16",
    )
    _write(
        directory / "door.wav",
        _event(95.0, 0.25, sample_rate, 0.36, 203),
        sample_rate,
        "PCM_16",
    )
    _write(
        directory / "music_city.wav",
        _music(3.2, 48000, 204, (110.0, 164.81, 220.0), 0.10, stereo=True),
        48000,
        "PCM_16",
    )
    _write(
        directory / "music_pulse.wav",
        _music(2.4, sample_rate, 205, (55.0, 82.41), 0.08, stereo=True),
        sample_rate,
        "PCM_24",
    )
    _write_scene(
        directory,
        {
            "scene_id": "city",
            "duration": 7.5,
            "ambience": {
                "file": "ambience_44k.wav",
                "loop_start": 0.2,
                "loop_end": 4.2,
                "gain": 0.8,
            },
            "samples": {"horn": "horn_16.wav", "door": "door.wav"},
            "events": [
                {"time": 0.95, "sample": "horn", "gain": 0.5},
                {"time": 2.4, "sample": "door", "gain": 0.6},
                {"time": 5.1, "sample": "horn", "gain": 0.32},
            ],
            "music_layers": [
                {
                    "file": "music_city.wav",
                    "start": 0.25,
                    "loops": 2,
                    "gain": 0.32,
                },
                {
                    "file": "music_pulse.wav",
                    "start": 2.1,
                    "loops": 2,
                    "gain": 0.25,
                },
            ],
            "dialogue_intervals": [
                {"start": 1.4, "end": 3.3, "duck_depth_db": 8.0},
                {"start": 4.7, "end": 6.6, "duck_depth_db": 12.0},
            ],
            "camera_cuts": [1.2, 3.9, 5.9],
        },
    )


def _scene_orchestra(root: Path) -> None:
    directory = root / "orchestra"
    sample_rate = 96000
    _write(
        directory / "hall_96k.wav",
        _ambience(4.0, sample_rate, 301, stereo=True),
        sample_rate,
        "PCM_24",
    )
    _write(
        directory / "timpani.wav",
        _event(62.0, 0.75, 48000, 0.45, 302),
        48000,
        "PCM_16",
    )
    _write(
        directory / "strings.wav",
        _music(4.5, 44100, 303, (130.81, 196.0, 261.63), 0.11, stereo=True),
        44100,
        "PCM_16",
    )
    _write_scene(
        directory,
        {
            "scene_id": "orchestra",
            "duration": 9.0,
            "ambience": {
                "file": "hall_96k.wav",
                "loop_start": 0.25,
                "loop_end": 3.75,
                "gain": 0.7,
            },
            "samples": {"timpani": "timpani.wav"},
            "events": [
                {"time": 1.6, "sample": "timpani", "gain": 0.58},
                {"time": 4.85, "sample": "timpani", "gain": 0.48},
                {"time": 7.7, "sample": "timpani", "gain": 0.36},
            ],
            "music_layers": [
                {
                    "file": "strings.wav",
                    "start": 0.75,
                    "loops": 2,
                    "gain": 0.35,
                }
            ],
            "dialogue_intervals": [
                {"start": 3.0, "end": 5.8, "duck_depth_db": 9.0},
            ],
            "camera_cuts": [2.25, 5.0, 7.25],
        },
    )


def generate(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _scene_forest(root)
    _scene_city(root)
    _scene_orchestra(root)
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic WAV and scene JSON samples."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="samples",
        help="destination directory (default: samples)",
    )
    args = parser.parse_args(argv)
    root = generate(Path(args.output_dir))
    print(f"generated 3 deterministic scenes under {root}")
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        print(f"  {scene.name}/scene.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
