"""FFmpeg-based high-quality resampling wrapper."""
from __future__ import annotations

import shutil
import subprocess

from .wavio import WavData, read_wav


class FFmpegError(Exception):
    """Raised when FFmpeg is missing or fails."""


def ensure_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FFmpegError(
            "ffmpeg executable not found on PATH; install FFmpeg >= 5.0 first"
        )
    try:
        result = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFmpegError(f"cannot invoke ffmpeg: {exc}") from exc
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    version = _parse_major_version(first_line)
    if version is not None and version < 5:
        raise FFmpegError(f"FFmpeg >= 5.0 required, found {first_line}")
    return executable


def _parse_major_version(first_line: str):
    # e.g. "ffmpeg version 7.0.2-static ..."
    parts = first_line.split()
    if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
        token = parts[2]
        digits = token.split(".")[0]
        if digits.isdigit():
            return int(digits)
    return None


def resample_to_48k(input_path: str, output_path: str, ffmpeg_bin: str,
                    target_rate: int = 48000) -> WavData:
    """Resample a WAV to `target_rate` Hz with anti-aliasing.

    Uses libswresample through the aresample filter.  The high-res phase
    shift (`filter_size=256`, `phase_shift=10`) enables the long polyphase
    anti-aliasing kernel; the intermediate file is 24-bit PCM, matching the
    final delivery bit depth.
    """
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", f"aresample={target_rate}:resampler=swr:filter_size=256:"
               "phase_shift=10:dither_method=triangular",
        "-c:a", "pcm_s24le",
        output_path,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise FFmpegError(f"ffmpeg invocation failed: {exc}") from exc
    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg resampling failed (exit {completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    data = read_wav(output_path)
    # The intermediate file is float; expose the physical target rate and keep
    # the source bit-depth metadata handled by the caller.
    return data
