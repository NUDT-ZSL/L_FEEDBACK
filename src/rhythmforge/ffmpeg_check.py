"""Optional FFmpeg container validation (never required for core building)."""

from __future__ import annotations

import shutil
import subprocess


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_wav(path: str) -> dict:
    """Decode a WAV through ffmpeg to prove the container is well-formed."""
    if not ffmpeg_available():
        return {"ffmpeg": "unavailable", "validated": False, "path": path}
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path, "-f", "null", "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return {
        "ffmpeg": shutil.which("ffmpeg"),
        "validated": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "path": path,
    }
