from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def generate(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 48_000
    duration = 2.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    left = (0.12 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
    right = (0.08 * np.sin(2 * np.pi * 880.0 * time)).astype(np.float32)
    left_path = output_dir / "voice-left.wav"
    right_path = output_dir / "voice-right.wav"
    sf.write(left_path, left, sample_rate, subtype="FLOAT")
    sf.write(right_path, right, sample_rate, subtype="FLOAT")

    scene = {
        "duration_seconds": duration,
        "frame_length": 1024,
        "objects": [
            {
                "clips": [
                    {"loops": 1, "path": left_path.name, "start": 0.0},
                ],
                "id": "left-motion",
                "keyframes": [
                    {
                        "azimuth": 0.0,
                        "distance": 1.0,
                        "elevation": 0.0,
                        "gain": 1.0,
                        "size": 0.25,
                        "time": 0.0,
                    },
                    {
                        "azimuth": 90.0,
                        "distance": 2.0,
                        "elevation": 15.0,
                        "gain": 0.5,
                        "size": 0.75,
                        "time": duration,
                    },
                ],
            },
            {
                "clips": [
                    {"loops": 2, "path": right_path.name, "start": 0.25},
                ],
                "id": "right-loop",
                "keyframes": [
                    {
                        "azimuth": 180.0,
                        "distance": 1.0,
                        "elevation": 0.0,
                        "gain": 1.0,
                        "size": 0.5,
                        "time": 0.0,
                    },
                    {
                        "azimuth": 270.0,
                        "distance": 3.0,
                        "elevation": -10.0,
                        "gain": 0.8,
                        "size": 0.5,
                        "time": duration,
                    },
                ],
            },
        ],
        "sample_rate": sample_rate,
    }
    scene_path = output_dir / "scene.json"
    scene_path.write_text(
        json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return scene_path, left_path, right_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic demo inputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/demo"),
        help="directory for generated WAV files and scene JSON",
    )
    args = parser.parse_args()
    scene, left, right = generate(args.output_dir)
    print(json.dumps({"scene": str(scene), "clips": [str(left), str(right)]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
