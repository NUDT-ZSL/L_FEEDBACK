"""Optional reference timeline extraction from local video files via FFmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess

from .subtitles import SubtitleEvent

PACKET_GAP_MS = 500.0


class FFmpegUnavailable(RuntimeError):
    pass


def _run_ffprobe(video_path: str) -> list[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "json",
        video_path,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed")
    payload = json.loads(completed.stdout or "{}")
    timestamps: list[float] = []
    for packet in payload.get("packets", []):
        value = packet.get("pts_time")
        if value is None or value == "N/A":
            continue
        timestamps.append(float(value) * 1000.0)
    return timestamps


def _run_ffmpeg(video_path: str) -> list[float]:
    # Fallback used when ffprobe is missing but ffmpeg is on PATH.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_path,
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip()[-500:] or "ffmpeg failed")
    timestamps: list[float] = []
    for line in completed.stderr.splitlines():
        marker = "pts_time:"
        if marker in line:
            fragment = line.split(marker, 1)[1].split()[0]
            try:
                timestamps.append(float(fragment) * 1000.0)
            except ValueError:
                continue
    return timestamps


def extract_reference_events(video_path: str) -> list[SubtitleEvent]:
    """Extract audio packet timestamps and cluster them into reference cues."""

    if shutil.which("ffprobe"):
        timestamps = _run_ffprobe(video_path)
    elif shutil.which("ffmpeg"):
        timestamps = _run_ffmpeg(video_path)
    else:
        raise FFmpegUnavailable("neither ffprobe nor ffmpeg was found on PATH")

    timestamps.sort()
    events: list[SubtitleEvent] = []
    if not timestamps:
        return events

    cluster_start = timestamps[0]
    previous = timestamps[0]
    for stamp in timestamps[1:]:
        if stamp - previous > PACKET_GAP_MS:
            events.append(
                SubtitleEvent(
                    index=len(events),
                    start=cluster_start,
                    end=max(previous, cluster_start),
                    text="",
                )
            )
            cluster_start = stamp
        previous = stamp
    events.append(
        SubtitleEvent(
            index=len(events),
            start=cluster_start,
            end=max(previous, cluster_start),
            text="",
        )
    )
    return events
