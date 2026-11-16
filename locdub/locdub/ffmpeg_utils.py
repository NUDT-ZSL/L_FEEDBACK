"""Thin wrappers around FFmpeg / FFprobe via the ffmpeg-python package.

Complex filter graphs are expressed with ffmpeg-python and executed with
``quiet`` logging while errors are captured and re-raised with useful
messages. Every command is built from explicit arguments, so output is
deterministic for a given input.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import ffmpeg


class FFmpegError(Exception):
    """Raised when an FFmpeg / FFprobe invocation fails."""


def _run(stream: Any) -> None:
    try:
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
    except ffmpeg.Error as exc:  # pragma: no cover - exercised in e2e tests
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise FFmpegError(f"ffmpeg failed: {stderr.strip()[-2000:]}") from exc


def probe_video(path: Path) -> Dict[str, Any]:
    """Return duration (ms), video stream info and optional audio stream info."""
    if not path.exists():
        raise FFmpegError(f"source video not found: {path}")
    try:
        raw = ffmpeg.probe(str(path))
    except (ffmpeg.Error, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise FFmpegError(f"ffprobe failed on {path.name}: {exc}") from exc

    format_info = raw.get("format", {})
    try:
        duration_seconds = float(format_info["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FFmpegError(f"cannot determine duration of {path.name}") from exc

    video_stream = next(
        (stream for stream in raw.get("streams", [])
         if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise FFmpegError(f"source {path.name} contains no video stream")
    audio_stream = next(
        (stream for stream in raw.get("streams", [])
         if stream.get("codec_type") == "audio"),
        None,
    )
    return {
        "duration_ms": int(round(duration_seconds * 1000.0)),
        "video_stream": video_stream,
        "has_audio": audio_stream is not None,
    }


def extract_background(video: Path, out_wav: Path,
                       sample_rate: int, channels: int) -> None:
    """Extract (or synthesise) the background track as a normalised WAV.

    When the source has no audio stream a silent WAV of identical duration is
    generated so downstream mixing always has a background bed.
    """
    probe = probe_video(video)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if probe["has_audio"]:
        stream = (
            ffmpeg
            .input(str(video))
            .output(
                str(out_wav),
                ac=channels,
                ar=sample_rate,
                acodec="pcm_s16le",
                loglevel="error",
            )
        )
        _run(stream)
    else:
        duration_seconds = probe["duration_ms"] / 1000.0
        stream = (
            ffmpeg
            .input("anullsrc=channel_layout={layout}:sample_rate={rate}".format(
                layout="stereo" if channels == 2 else "mono",
                rate=sample_rate,
            ), f="lavfi", t=duration_seconds)
            .output(str(out_wav), acodec="pcm_s16le", loglevel="error")
        )
        _run(stream)


def mux_mixdown(video: Path, mix_wav: Path, out_mp4: Path,
                duration_ms: int) -> None:
    """Combine the original video with a prepared mixed audio WAV.

    Video is copied (no re-encode, timestamps preserved) and audio encoded as
    AAC. ``-shortest`` is deliberately avoided: both inputs share the same
    length, and explicit duration keeps A/V timestamps aligned.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    video_input = ffmpeg.input(str(video))
    audio_input = ffmpeg.input(str(mix_wav))
    stream = (
        ffmpeg
        .output(
            video_input.video,
            audio_input.audio,
            str(out_mp4),
            vcodec="copy",
            acodec="aac",
            **{"b:a": "192k"},
            t=duration_ms / 1000.0,
            movflags="+faststart",
            loglevel="error",
        )
    )
    _run(stream)


def make_solid_video(out_path: Path, duration_ms: int,
                     size: str = "640x360", frame_rate: int = 25,
                     color: str = "navy") -> None:
    """Deterministic solid-colour test video with a quiet tone bed."""
    duration_seconds = duration_ms / 1000.0
    video = ffmpeg.input(
        f"color=c={color}:s={size}:r={frame_rate}:d={duration_seconds}",
        f="lavfi",
    )
    audio = ffmpeg.input(
        f"sine=frequency=220:sample_rate=48000:duration={duration_seconds}",
        f="lavfi",
    )
    stream = ffmpeg.output(
        video,
        audio,
        str(out_path),
        vcodec="libx264",
        pix_fmt="yuv420p",
        acodec="aac",
        **{"b:a": "128k"},
        t=duration_seconds,
        loglevel="error",
    )
    _run(stream)


def wav_duration_ms(path: Path) -> int:
    try:
        info = ffmpeg.probe(str(path))
    except (ffmpeg.Error, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise FFmpegError(f"ffprobe failed on {path.name}: {exc}") from exc
    audio = next(
        (stream for stream in info.get("streams", [])
         if stream.get("codec_type") == "audio"),
        None,
    )
    if audio is None:
        raise FFmpegError(f"no audio stream in {path.name}")
    return int(round(float(info["format"]["duration"]) * 1000.0))


def normalise_wav(src: Path, dst: Path, sample_rate: int,
                  channels: int) -> None:
    """Resample / re-channel a WAV to the mix-down format (PCM16)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    stream = (
        ffmpeg
        .input(str(src))
        .output(
            str(dst),
            ar=sample_rate,
            ac=channels,
            acodec="pcm_s16le",
            loglevel="error",
        )
    )
    _run(stream)


def build_failed_mp4(video: Path, out_mp4: Path) -> None:
    """Fallback mux used when no mix-down could be produced (keeps video)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    stream = (
        ffmpeg
        .input(str(video))
        .output(str(out_mp4), vcodec="copy", acodec="aac",
                loglevel="error")
    )
    _run(stream)
