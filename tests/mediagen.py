"""Generate small deterministic media into temporary test directories."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mediamixer.ffmpeg_utils import ffmpeg_exe


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed: "
            + result.stderr.decode("utf-8", "replace").strip()[-400:]
        )


def make_video(
    directory: Path,
    name: str = "video.mp4",
    size: str = "320x240",
    fps: int = 24,
    duration: float = 2.0,
    source: str = "testsrc",
) -> Path:
    path = directory / name
    if source.startswith("color"):
        source_spec = f"{source}:s={size}:r={fps}:d={duration}"
    else:
        source_spec = f"{source}=size={size}:rate={fps}:duration={duration}"
    _run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source_spec,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            str(path),
        ]
    )
    return path


def make_sine(
    directory: Path,
    name: str,
    frequency: int = 440,
    duration: float = 2.0,
    amplitude: float | None = None,
) -> Path:
    path = directory / name
    source = (
        f"sine=frequency={frequency}:duration={duration}:sample_rate=48000"
        if amplitude is None
        else (
            f"aevalsrc={amplitude}|{amplitude}:s=48000:d={duration}"
        )
    )
    _run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-ac",
            "2",
            str(path),
        ]
    )
    return path
