"""Thin, deterministic wrappers around the ffmpeg and ffprobe executables.

The executable directory is taken from the ``FFMPEG_PATH`` environment
variable when set; otherwise the tools are resolved from ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .errors import FFmpegError

SAMPLE_RATE = 48000


def _tool_dir() -> Path | None:
    raw = os.environ.get("FFMPEG_PATH")
    if not raw:
        return None
    path = Path(raw)
    # Allow either the bin directory or the ffmpeg executable itself.
    if path.is_file():
        return path.parent
    return path


def _resolve(tool: str) -> str:
    directory = _tool_dir()
    if directory is not None:
        candidate = directory / tool
        if candidate.exists():
            return str(candidate)
        candidate_exe = directory / f"{tool}.exe"
        if candidate_exe.exists():
            return str(candidate_exe)
        raise FFmpegError(
            f"executable '{tool}' not found in FFMPEG_PATH directory: {directory}"
        )
    found = shutil.which(tool)
    if found:
        return found
    raise FFmpegError(
        f"executable '{tool}' not found; put it on PATH or set FFMPEG_PATH"
    )


def ffmpeg_exe() -> str:
    return _resolve("ffmpeg")


def ffprobe_exe() -> str:
    return _resolve("ffprobe")


def _run(cmd: list[str], stdin_data: bytes | None = None) -> bytes:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(input=stdin_data)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        tail = detail[-1] if detail else "unknown ffmpeg failure"
        raise FFmpegError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}): {tail}")
    return stdout


def probe_video(path: str | Path) -> dict[str, object]:
    """Return width, height, frame rate (Fraction) and total frame count."""
    cmd = [
        ffprobe_exe(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    import json

    data = json.loads(_run(cmd).decode("utf-8"))
    streams = data.get("streams") or []
    if not streams:
        raise FFmpegError(f"no video stream found in {path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])

    rate_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    num, den = rate_raw.split("/")
    rate = Fraction(int(num), int(den))
    if rate <= 0:
        raise FFmpegError(f"invalid frame rate '{rate_raw}' in {path}")

    nb_frames = stream.get("nb_frames")
    if nb_frames is None or int(nb_frames) <= 0:
        nb_frames = _count_frames(path)
    frame_count = int(nb_frames)
    if frame_count <= 0:
        raise FFmpegError(f"could not determine frame count for {path}")

    return {
        "width": width,
        "height": height,
        "frame_rate": rate,
        "frame_count": frame_count,
    }


def _count_frames(path: str | Path) -> int:
    cmd = [
        ffprobe_exe(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    import json

    data = json.loads(_run(cmd).decode("utf-8"))
    stream = (data.get("streams") or [{}])[0]
    return int(stream.get("nb_read_frames", 0))


def frame_timestamp_us(frame_index: int, frame_rate: Fraction) -> int:
    """Display timestamp in microseconds using integer-only arithmetic."""
    return frame_index * 1_000_000 * frame_rate.denominator // frame_rate.numerator


def decode_video_rgb(path: str | Path):
    """Spawn ffmpeg decoding video to raw RGB24 on stdout.

    Returns ``(process, stdout)``; the caller must drain stdout and wait.
    """
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def decode_audio_pcm(path: str | Path) -> bytes:
    """Decode any audio file to 48 kHz stereo float32 little-endian PCM."""
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    return _run(cmd)


def encode_intermediate_video(
    width: int,
    height: int,
    frame_rate: Fraction,
    frame_count: int,
    output_path: str | Path,
) -> subprocess.Popen:
    """Spawn an encoder consuming raw RGB24 frames from stdin (H.264, no audio)."""
    rate_text = f"{frame_rate.numerator}/{frame_rate.denominator}"
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        rate_text,
        "-i",
        "pipe:0",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def mux_final(
    intermediate_video: str | Path,
    pcm_bytes: bytes,
    output_path: str | Path,
) -> None:
    """Mux the persisted H.264 video with PCM audio (AAC 192k) into MP4."""
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(intermediate_video),
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-i",
        "pipe:0",
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
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run(cmd, stdin_data=pcm_bytes)
