"""Small ffprobe wrappers that return structured data without raw diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import PROBE_TIMEOUT_SECONDS
from .process import run_command


class ProbeError(Exception):
    pass


class ProbeTimeout(ProbeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    streams: tuple[Mapping[str, Any], ...]
    format: Mapping[str, Any]

    @property
    def video_streams(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(s for s in self.streams if s.get("codec_type") == "video")

    @property
    def audio_streams(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(s for s in self.streams if s.get("codec_type") == "audio")

    @property
    def subtitle_streams(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(s for s in self.streams if s.get("codec_type") == "subtitle")

    @property
    def video(self) -> Mapping[str, Any] | None:
        streams = self.video_streams
        return streams[0] if streams else None

    @property
    def audio(self) -> Mapping[str, Any] | None:
        streams = self.audio_streams
        return streams[0] if streams else None


def resolve_tool(env_name: str, executable: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        found = shutil.which(configured)
        if found:
            return found
        return None

    found = shutil.which(executable)
    if found:
        return found

    suffix = ".exe" if os.name == "nt" else ""
    local_candidates = (
        Path(sys.executable).resolve().parent / f"{executable}{suffix}",
        Path(__file__).resolve().parent.parent / "bin" / f"{executable}{suffix}",
    )
    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate)
    return None


_PROBE_ENTRIES = (
    "format=start_time,duration,format_name:"
    "stream=index,codec_type,codec_name,profile,level,pix_fmt,width,height,"
    "r_frame_rate,avg_frame_rate,time_base,nb_frames,duration,start_time,"
    "channels,channel_layout,sample_rate,disposition"
)


def _probe_json(
    ffprobe: str,
    path: Path,
    *,
    timeout: float,
    extra_args: list[str] | None = None,
) -> Mapping[str, Any]:
    command = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        *([] if extra_args is None else extra_args),
        "-show_entries",
        _PROBE_ENTRIES,
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command, timeout=timeout)
    if result.timed_out:
        raise ProbeTimeout("ffprobe timed out")
    if result.returncode != 0:
        raise ProbeError("ffprobe failed")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProbeError("ffprobe returned no stream data")
    return payload


def probe_media(
    ffprobe: str,
    path: Path,
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> MediaInfo:
    payload = _probe_json(ffprobe, path, timeout=timeout)
    streams = payload.get("streams")
    format_info = payload.get("format")
    if not isinstance(streams, list) or not streams:
        raise ProbeError("media has no streams")
    if not isinstance(format_info, dict):
        format_info = {}
    typed_streams = tuple(s for s in streams if isinstance(s, dict))
    return MediaInfo(streams=typed_streams, format=format_info)


def probe_output_video(
    ffprobe: str,
    path: Path,
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> MediaInfo:
    return probe_media(ffprobe, path, timeout=timeout)


def keyframe_times(
    ffprobe: str,
    path: Path,
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> list[float]:
    command = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=key_frame,pts_time,best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command, timeout=timeout)
    if result.timed_out:
        raise ProbeTimeout("keyframe probe timed out")
    if result.returncode != 0:
        raise ProbeError("keyframe probe failed")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProbeError("keyframe probe returned invalid JSON") from exc
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ProbeError("keyframe probe returned no frames")
    times: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict) or str(frame.get("key_frame")) != "1":
            continue
        raw = frame.get("pts_time") or frame.get("best_effort_timestamp_time")
        try:
            times.append(float(raw))
        except (TypeError, ValueError):
            continue
    return times


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
