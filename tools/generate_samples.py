"""Generate deterministic local sample media and project files.

Uses only FFmpeg's built-in ``testsrc`` and ``sine`` sources; no network
access and no external media are required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mediamixer.ffmpeg_utils import ffmpeg_exe

WIDTH = 640
HEIGHT = 360
FPS = 30
DURATION_SECONDS = 4


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(
            "ffmpeg failed: "
            + result.stderr.decode("utf-8", "replace").strip()[-500:]
        )


def generate(samples_dir: Path) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_exe()

    video = samples_dir / "video.mp4"
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={DURATION_SECONDS}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(video),
        ]
    )

    specs = [
        ("track_dialog.wav", 440, 3.5),
        ("track_music.wav", 220, 5.0),
        ("track_fx.wav", 880, 2.0),
    ]
    for name, frequency, duration in specs:
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:duration={duration}:sample_rate=48000",
                "-ac",
                "2",
                str(samples_dir / name),
            ]
        )

    project = {
        "video_path": "video.mp4",
        "output_path": "output.mp4",
        "audio_tracks": [
            {
                "path": "track_dialog.wav",
                "offset_ms": 0,
                "volume_envelope": [1.0, 0.8, 0.5, 0.9],
            },
            {
                "path": "track_music.wav",
                "offset_ms": 500,
                "volume_envelope": [0.4, 0.4, 0.2],
            },
            {
                "path": "track_fx.wav",
                "offset_ms": -300,
                "volume_envelope": [0.9, 0.6, 0.3, 0.1],
            },
        ],
        "frame_markers": [
            {"frame": 15, "drift_ms": -5},
            {"frame": 45, "drift_ms": 8},
        ],
    }
    project_path = samples_dir / "project.json"
    project_path.write_text(
        json.dumps(project, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic samples.")
    parser.add_argument(
        "--out",
        default="samples",
        help="output directory for generated media and project JSON",
    )
    args = parser.parse_args()
    path = generate(Path(args.out).resolve())
    print(f"sample project written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
