"""Deterministic built-in scene and WAV sample generator."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from .compression import CompressorConfig
from .errors import ParameterError, RenderRuntimeError
from .geometry import quat_yaw
from .render import render_scene
from .scene import load_scene


SAMPLE_RATE = 48000
DURATION_SECONDS = 8.0


def _fade(samples: np.ndarray, sample_rate: int, seconds: float = 0.025) -> None:
    count = min(len(samples) // 2, int(seconds * sample_rate))
    if count > 0:
        ramp = np.linspace(0.0, 1.0, count)
        samples[:count] *= ramp
        samples[-count:] *= ramp[::-1]


def _dialogue_clip() -> np.ndarray:
    duration = 5.0
    t = np.arange(int(duration * SAMPLE_RATE)) / float(SAMPLE_RATE)
    syllables = 0.35 + 0.65 * np.sin(2.0 * math.pi * 1.9 * t) ** 2
    wobble = 1.0 + 0.08 * np.sin(2.0 * math.pi * 4.3 * t)
    carrier = (
        0.24 * np.sin(2.0 * math.pi * 173.0 * t * wobble)
        + 0.12 * np.sin(2.0 * math.pi * 346.0 * t)
        + 0.06 * np.sin(2.0 * math.pi * 692.0 * t)
    )
    result = syllables * carrier
    _fade(result, SAMPLE_RATE)
    return np.clip(result, -0.8, 0.8)


def _ambience_clip() -> np.ndarray:
    duration = 2.0
    t = np.arange(int(duration * SAMPLE_RATE)) / float(SAMPLE_RATE)
    rng = np.random.default_rng(20261117)
    result = np.zeros_like(t)
    harmonics = rng.integers(20, 240, size=18)
    for index, harmonic in enumerate(harmonics):
        amplitude = 0.018 / (1.0 + index * 0.12)
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        result += amplitude * np.sin(
            2.0 * math.pi * float(harmonic) * t / duration + phase
        )
    return np.clip(result, -0.5, 0.5)


def _flyby_clip() -> np.ndarray:
    duration = 4.0
    t = np.arange(int(duration * SAMPLE_RATE)) / float(SAMPLE_RATE)
    phase = 2.0 * math.pi * (180.0 * t + 12.0 * t * t)
    result = (
        0.30 * np.sin(phase)
        + 0.10 * np.sin(2.0 * phase)
        + 0.04 * np.sin(3.0 * phase)
    )
    _fade(result, SAMPLE_RATE, 0.04)
    return np.clip(result, -0.8, 0.8)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.asarray(samples, dtype=np.float64),
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )


def generate_demo(directory: Path) -> Path:
    directory = directory.resolve()
    assets = directory / "assets"
    _write_wav(assets / "dialogue.wav", _dialogue_clip())
    _write_wav(assets / "ambience_loop.wav", _ambience_clip())
    _write_wav(assets / "flyby.wav", _flyby_clip())

    head_keys = []
    for index in range(17):
        time_seconds = index * 0.5
        angle = (math.pi / 2.0) * (time_seconds / DURATION_SECONDS)
        quaternion = quat_yaw(angle)
        head_keys.append(
            {
                "time": int(round(time_seconds * SAMPLE_RATE)),
                "quaternion": [float(value) for value in quaternion],
            }
        )
    scene = {
        "format": "spatial-scene-v1",
        "sample_rate": SAMPLE_RATE,
        "frame_length": 256,
        "duration_samples": int(DURATION_SECONDS * SAMPLE_RATE),
        "speed_of_sound": 343.0,
        "reference_distance": 1.0,
        "head_radius": 0.0875,
        "occlusion_base_cutoff_hz": 18000.0,
        "head": {
            "position": [0.0, 0.0, 0.0],
            "quaternion_keys": head_keys,
        },
        "sources": [
            {
                "name": "dialogue",
                "file": "assets/dialogue.wav",
                "gain": 1.0,
                "loop": False,
                "start_sample": 12000,
                "position_keys": [
                    {"time": 0, "position": [-1.15, 0.0, 1.9]},
                    {
                        "time": int(DURATION_SECONDS * SAMPLE_RATE),
                        "position": [-1.15, 0.0, 1.9],
                    },
                ],
            },
            {
                "name": "ambience",
                "file": "assets/ambience_loop.wav",
                "gain": 0.55,
                "loop": True,
                "start_sample": 0,
                "position_keys": [
                    {"time": 0, "position": [3.2, 0.4, -2.0]},
                    {
                        "time": int(DURATION_SECONDS * SAMPLE_RATE),
                        "position": [3.2, 0.4, -2.0],
                    },
                ],
            },
            {
                "name": "flyby",
                "file": "assets/flyby.wav",
                "gain": 0.9,
                "loop": False,
                "start_sample": 0,
                "position_keys": [
                    {"time": 0, "position": [0.0, 0.25, 9.0]},
                    {
                        "time": int(DURATION_SECONDS * SAMPLE_RATE),
                        "position": [0.0, 0.25, 2.0],
                    },
                ],
            },
        ],
        "occluders": [
            {
                "name": "screen",
                "min": [-4.0, -2.0, 3.6],
                "max": [4.0, 2.0, 4.2],
                "attenuation": 0.55,
                "cutoff_hz": 5200.0,
            }
        ],
    }
    scene_path = directory / "scene.json"
    directory.mkdir(parents=True, exist_ok=True)
    with scene_path.open("w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return scene_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic demo scene and optionally render it."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="demo_scene",
        help="directory for generated JSON/WAV files (default: demo_scene)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional stereo WAV render path",
    )
    parser.add_argument(
        "--normalize-lufs",
        type=float,
        default=None,
        help="optional integrated loudness target",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="enable master compression",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        scene_path = generate_demo(Path(args.directory))
        print("generated scene: %s" % scene_path)
        if args.output is not None:
            output = Path(args.output)
            scene = load_scene(scene_path)
            result = render_scene(
                scene,
                output,
                normalize_lufs=args.normalize_lufs,
                compressor_config=(
                    CompressorConfig() if args.compress else None
                ),
                stderr=sys.stderr,
            )
            print("rendered WAV: %s" % output)
            print(
                "duration=%.3f s peak=%.2f dBFS"
                % (result.duration_seconds, result.peak_dbfs)
            )
        return 0
    except ParameterError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 1
    except RenderRuntimeError as exc:
        sys.stderr.write("runtime error: %s\n" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
