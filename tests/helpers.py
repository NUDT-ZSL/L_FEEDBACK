"""Shared helpers for the test suite."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mediamixer.ffmpeg_utils import ffprobe_exe, ffmpeg_exe


def ffprobe_json(path: Path) -> dict:
    result = subprocess.run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,nb_frames,sample_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


def extract_frame(path: Path, frame_number: int, fps: int = 30) -> bytes:
    """Extract one frame as raw RGB24 bytes."""
    result = subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{frame_number})",
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


def write_project(directory: Path, payload: dict, name: str = "project.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_project(video: str, tracks: list[dict], output: str = "out.mp4",
                 markers: list[dict] | None = None) -> dict:
    return {
        "video_path": video,
        "output_path": output,
        "audio_tracks": tracks,
        "frame_markers": markers or [],
    }


def make_track(name: str, offset_ms: int = 0,
               envelope: list[float] | None = None) -> dict:
    return {
        "path": name,
        "offset_ms": offset_ms,
        "volume_envelope": envelope if envelope is not None else [1.0, 1.0],
    }
