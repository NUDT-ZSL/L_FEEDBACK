from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError


@dataclass
class FFmpegTools:
    ffmpeg: str
    ffprobe: str
    version: str


def _resolve(name: str, explicit: str | None = None) -> str | None:
    env_name = "FFMPEG_BIN" if name == "ffmpeg" else "FFPROBE_BIN"
    candidate = explicit or os.environ.get(env_name)
    if candidate:
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(candidate)
        if found:
            return found
        return None
    return shutil.which(name)


def resolve_ffmpeg(ffmpeg: str | None = None, ffprobe: str | None = None) -> FFmpegTools:
    ffmpeg_path = _resolve("ffmpeg", ffmpeg)
    ffprobe_path = _resolve("ffprobe", ffprobe)
    if not ffmpeg_path:
        raise BusinessError(
            "E_TRANSCODE",
            "ffmpeg_check",
            "ffmpeg executable was not found on PATH",
            {"env": "FFMPEG_BIN"},
        )
    if not ffprobe_path:
        raise BusinessError(
            "E_TRANSCODE",
            "ffmpeg_check",
            "ffprobe executable was not found on PATH",
            {"env": "FFPROBE_BIN"},
        )
    try:
        completed = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BusinessError("E_TRANSCODE", "ffmpeg_check", f"Unable to execute ffmpeg: {exc}") from exc
    text = completed.stdout.decode("utf-8", errors="replace")
    match = re.search(r"ffmpeg version\s+(?:n)?(\d+)\.", text)
    if completed.returncode != 0 or not match:
        raise BusinessError(
            "E_TRANSCODE",
            "ffmpeg_check",
            "ffmpeg -version failed",
            {"returncode": completed.returncode, "stderr": completed.stderr.decode(errors="replace")[:500]},
        )
    major = int(match.group(1))
    if major < 6:
        raise BusinessError(
            "E_TRANSCODE",
            "ffmpeg_check",
            "FFmpeg 6.0 or newer is required",
            {"detected_major": major},
        )
    return FFmpegTools(ffmpeg_path, ffprobe_path, text.splitlines()[0])


def run_ffmpeg(
    ffmpeg: str,
    args: list[str],
    stage: str,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BusinessError(
            "E_TRANSCODE",
            stage,
            f"FFmpeg execution failed: {exc}",
            {"command": command},
        ) from exc
    if completed.returncode != 0:
        raise BusinessError(
            "E_TRANSCODE",
            stage,
            "FFmpeg returned a non-zero exit status",
            {
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-3000:],
                "command": command,
            },
        )
    return completed


def probe_json(ffprobe: str, path: str | Path, extra: list[str] | None = None) -> dict[str, Any]:
    command = [ffprobe, "-v", "error", "-of", "json", *(extra or []), str(path)]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BusinessError("E_TRANSCODE", "ffprobe", f"Unable to execute ffprobe: {exc}") from exc
    if completed.returncode != 0:
        raise BusinessError(
            "E_TRANSCODE",
            "ffprobe",
            "ffprobe failed",
            {
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-2000:],
                "path": str(path),
            },
        )
    try:
        return json.loads(completed.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise BusinessError("E_TRANSCODE", "ffprobe", "ffprobe returned invalid JSON") from exc


def transcode_base_ts(
    ffmpeg: str,
    input_wav: str | Path,
    output_ts: str | Path,
    duration_seconds: float,
    channels: int,
) -> None:
    output_ts = Path(output_ts)
    output_ts.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=720x576:r=25:d={duration_seconds:.6f}",
        "-i",
        str(input_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "mpeg2video",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "700k",
        "-maxrate",
        "900k",
        "-bufsize",
        "1800k",
        "-g",
        "12",
        "-bf",
        "0",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        str(channels),
        "-metadata:s:a:0",
        "language=eng",
        "-mpegts_pmt_start_pid",
        "0x1000",
        "-mpegts_start_pid",
        "0x0101",
        "-pat_period",
        "0.1",
        "-sdt_period",
        "0.5",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-shortest",
        "-f",
        "mpegts",
        str(output_ts),
    ]
    run_ffmpeg(ffmpeg, args, "ffmpeg_transcode", timeout=max(300, int(duration_seconds * 20)))
