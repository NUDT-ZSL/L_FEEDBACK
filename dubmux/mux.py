"""Local ffmpeg/ffprobe helpers. No network access is ever performed."""

from __future__ import annotations

import json
import subprocess

from .errors import DubError, E_FFMPEG_FAILURE


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise DubError(E_FFMPEG_FAILURE, f"failed to run {cmd[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise DubError(
            E_FFMPEG_FAILURE,
            f"{cmd[0]} exited {proc.returncode}: {proc.stderr.strip()[-400:]}",
        )
    return proc


def probe_duration(path: str) -> float:
    proc = _run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", path,
        ]
    )
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise DubError(E_FFMPEG_FAILURE, f"cannot probe duration of {path}") from exc


def probe_stream_durations(path: str) -> dict:
    """Return {'video': seconds, 'audio': seconds} for the first streams."""
    proc = _run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,duration", "-of", "json", path,
        ]
    )
    try:
        streams = json.loads(proc.stdout)["streams"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise DubError(E_FFMPEG_FAILURE, f"cannot probe streams of {path}") from exc
    out: dict[str, float] = {}
    for s in streams:
        kind = s.get("codec_type")
        if kind in ("video", "audio") and kind not in out and "duration" in s:
            out[kind] = float(s["duration"])
    return out


def extract_audio(video_path: str, wav_path: str, sample_rate: int) -> None:
    """Decode the video's audio track to mono PCM WAV at a fixed rate."""
    _run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            wav_path,
        ]
    )


def mux(video_path: str, wav_path: str, srt_path: str, out_path: str) -> None:
    """Mux without re-encoding the video; audio -> AAC, subtitles -> mov_text."""
    _run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_path, "-i", wav_path, "-i", srt_path,
            "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
            "-movflags", "+faststart",
            out_path,
        ]
    )
