"""Shared test helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_tool(name: str) -> str | None:
    env_name = "FFMPEG_BIN" if name == "ffmpeg" else "FFPROBE_BIN"
    configured = os.environ.get(env_name)
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which(name)
    if found:
        return found
    candidates = (
        Path("/tmp/ffmpeg6") / name,
        Path("/home/liang/tools/ffmpeg-static") / name,
        Path("/home/liang/tools") / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def find_tools() -> tuple[str, str] | None:
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    if ffmpeg is None or ffprobe is None:
        return None
    return ffmpeg, ffprobe
