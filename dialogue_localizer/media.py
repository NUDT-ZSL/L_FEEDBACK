"""FFmpeg-backed probing, decoding, resampling and channel alignment."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .errors import EXIT_AUDIO, BusinessError
from .loudness import SAMPLE_RATE


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    duration_sec: float | None


@dataclass(frozen=True)
class ToolPaths:
    ffmpeg: str
    ffprobe: str | None


def _subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    return kwargs


def find_tools(
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> ToolPaths:
    if ffmpeg:
        ffmpeg_path = (
            str(Path(ffmpeg).resolve())
            if Path(ffmpeg).is_file()
            else shutil.which(ffmpeg)
        )
    else:
        ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise BusinessError(
            code="FFMPEG_NOT_FOUND",
            message="ffmpeg was not found in PATH",
            exit_code=EXIT_AUDIO,
        )
    if ffprobe:
        ffprobe_path = (
            str(Path(ffprobe).resolve())
            if Path(ffprobe).is_file()
            else shutil.which(ffprobe)
        )
        if not ffprobe_path:
            raise BusinessError(
                code="FFPROBE_NOT_FOUND",
                message="the requested ffprobe executable was not found",
                exit_code=EXIT_AUDIO,
                file=ffprobe,
            )
    else:
        ffprobe_path = shutil.which("ffprobe")
    return ToolPaths(ffmpeg=ffmpeg_path, ffprobe=ffprobe_path)


def _audio_error(
    code: str,
    message: str,
    path: str,
    *,
    details: dict[str, Any] | None = None,
) -> BusinessError:
    return BusinessError(
        code=code,
        message=f"{message}: {path}",
        exit_code=EXIT_AUDIO,
        file=path,
        details=details or {},
    )


def _probe_with_ffprobe(path: str, ffprobe: str) -> AudioInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "json",
        path,
    ]
    completed = subprocess.run(command, **_subprocess_kwargs())
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            "ffprobe could not identify an audio stream",
            path,
            details={"stderr": stderr[-2000:]},
        )
    try:
        payload = json.loads(
            completed.stdout.decode("utf-8", errors="replace")
        )
        streams = payload.get("streams") or []
        if not streams:
            raise ValueError("no audio stream")
        stream = streams[0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration_value = (payload.get("format") or {}).get("duration")
        duration = (
            float(duration_value)
            if duration_value is not None
            else None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            f"invalid ffprobe audio metadata ({exc})",
            path,
        ) from exc
    if sample_rate <= 0 or channels <= 0:
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            "ffprobe returned invalid sample rate or channel count",
            path,
        )
    return AudioInfo(
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration,
    )


def _probe_with_soundfile(path: str) -> AudioInfo:
    try:
        info = sf.info(path)
    except (RuntimeError, OSError) as exc:
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            f"could not decode audio ({exc})",
            path,
        ) from exc
    duration = (
        float(info.frames) / float(info.samplerate)
        if info.samplerate
        else None
    )
    return AudioInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        duration_sec=duration,
    )


def probe_audio(path: str, tools: ToolPaths) -> AudioInfo:
    resolved = str(Path(path).resolve())
    if not Path(resolved).is_file():
        raise _audio_error(
            "AUDIO_FILE_MISSING",
            "audio file is missing",
            resolved,
        )
    if tools.ffprobe:
        return _probe_with_ffprobe(resolved, tools.ffprobe)
    return _probe_with_soundfile(resolved)


def validate_source_format(info: AudioInfo, path: str, expected_rate: int, expected_channels: int) -> None:
    if info.sample_rate != expected_rate:
        raise _audio_error(
            "SOURCE_SAMPLE_RATE_MISMATCH",
            "source dialogue sample rate does not match the manifest",
            path,
            details={
                "expected_sample_rate": expected_rate,
                "actual_sample_rate": info.sample_rate,
                "role": "source_dialogue",
            },
        )
    if info.channels != expected_channels:
        raise _audio_error(
            "SOURCE_CHANNEL_MISMATCH",
            "source dialogue channel count does not match the manifest",
            path,
            details={
                "expected_channels": expected_channels,
                "actual_channels": info.channels,
                "role": "source_dialogue",
            },
        )


def validate_target_format(
    info: AudioInfo,
    path: str,
    expected_rate: int,
    expected_channels: int,
) -> None:
    if expected_rate and info.sample_rate != expected_rate:
        raise _audio_error(
            "SAMPLE_RATE_MISMATCH",
            "target dialogue sample rate does not match the manifest",
            path,
            details={
                "expected_sample_rate": expected_rate,
                "actual_sample_rate": info.sample_rate,
                "role": "target_dialogue",
            },
        )
    if expected_channels and info.channels != expected_channels:
        raise _audio_error(
            "CHANNEL_MISMATCH",
            "target dialogue channel count does not match the manifest",
            path,
            details={
                "expected_channels": expected_channels,
                "actual_channels": info.channels,
                "role": "target_dialogue",
            },
        )


def decode_audio_canonical(path: str, tools: ToolPaths) -> np.ndarray:
    """Decode any ffmpeg-supported file to 48 kHz, stereo float32."""

    resolved = str(Path(path).resolve())
    command = [
        tools.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        resolved,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, **_subprocess_kwargs())
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            "ffmpeg failed to decode or resample the audio",
            resolved,
            details={"stderr": stderr[-2000:]},
        )
    raw = completed.stdout
    if not raw or len(raw) % (2 * np.dtype("<f4").itemsize) != 0:
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            "ffmpeg returned incomplete or empty PCM data",
            resolved,
            details={"decoded_bytes": len(raw)},
        )
    audio = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    audio = audio.reshape(-1, 2).copy()
    if audio.shape[0] == 0:
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            "decoded audio contains no samples",
            resolved,
        )
    if not np.all(np.isfinite(audio)):
        raise _audio_error(
            "AUDIO_DECODE_FAILED",
            "decoded audio contains non-finite samples",
            resolved,
        )
    return audio


def seconds_to_samples(seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))

