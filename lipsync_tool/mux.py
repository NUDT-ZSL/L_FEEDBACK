"""MP4 remuxing around the FFmpeg command line.

The video stream is copied bit-for-bit (``-c:v copy``) while the stretched
audio is encoded to AAC.  ``-fflags +genpts`` regenerates presentation
timestamps so the remuxed container stays continuous.  Any failure raises
:class:`MuxError`; the pipeline then falls back to WAV + JSON output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


class MuxError(RuntimeError):
    """Raised when an MP4 remux operation cannot complete."""


def ffmpeg_available() -> bool:
    """Return True when both ffmpeg and ffprobe resolve on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def probe_streams(path: str | Path) -> List[Dict[str, str]]:
    """Return ffprobe's per-stream codec/type summary."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,duration",
        "-of",
        "json",
        str(path),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise MuxError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout or "{}")
    return payload.get("streams", [])


@dataclass
class MuxResult:
    output_path: Path
    video_codec: str
    audio_codec: str
    duration_seconds: float
    command: List[str]


def remux_mp4(
    video_path: str | Path,
    audio_wav_path: str | Path,
    output_path: str | Path,
    simulate_failure: bool = False,
) -> MuxResult:
    """Copy the video stream and AAC-encode the audio into an MP4 container."""
    video_path = Path(video_path)
    audio_wav_path = Path(audio_wav_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ffmpeg_available():
        raise MuxError("ffmpeg/ffprobe not found on PATH")
    if not video_path.is_file():
        raise MuxError(f"video input missing: {video_path}")
    if not audio_wav_path.is_file():
        raise MuxError(f"audio input missing: {audio_wav_path}")

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostdin",
        "-fflags",
        "+genpts",
        "-i",
        str(video_path),
        "-i",
        str(audio_wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-async",
        "1",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    if simulate_failure:
        raise MuxError("simulate_mux_failure is enabled in configuration")

    result = _run(command)
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink()
        raise MuxError(
            f"ffmpeg remux failed (exit {result.returncode}): {result.stderr.strip()[-800:]}"
        )

    streams = probe_streams(output_path)
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams or not audio_streams:
        output_path.unlink(missing_ok=True)
        raise MuxError("remuxed MP4 is missing a video or audio stream")

    duration = 0.0
    for stream in streams:
        try:
            duration = max(duration, float(stream.get("duration") or 0.0))
        except (TypeError, ValueError):
            continue
    return MuxResult(
        output_path=output_path,
        video_codec=video_streams[0].get("codec_name", "unknown"),
        audio_codec=audio_streams[0].get("codec_name", "unknown"),
        duration_seconds=duration,
        command=command,
    )
