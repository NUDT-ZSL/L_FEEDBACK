"""FFmpeg/ffprobe wrappers for probing and decoding source audio.

Decoding always produces packed little-endian float32 PCM at the source's
native sample rate and native channel count. Resampling and channel layout
conversion are performed in Python so the processing chain stays deterministic
and testable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CineMixerError


def _validate_wav_container(path: Path) -> None:
    """Detect truncated PCM WAV files before FFmpeg silently tolerates them.

    A deterministic truncation cuts both the ``data`` chunk body and its
    header, so FFmpeg/ffprobe report the shortened duration as if it were
    legitimate. Comparing the RIFF/data chunk sizes against the actual file
    size exposes this class of corruption for PCM sources.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return  # non-WAV or unrecognized container; rely on FFmpeg
            file_size = path.stat().st_size
            riff_declared = int.from_bytes(header[4:8], "little")
            if riff_declared + 8 != file_size:
                raise CineMixerError(
                    "E_TRACK_LOAD_FAIL",
                    (
                        f"source {path} is corrupt: RIFF size "
                        f"{riff_declared + 8} != file size {file_size}"
                    ),
                    4,
                )
            while True:
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id = chunk_header[:4]
                chunk_size = int.from_bytes(chunk_header[4:8], "little")
                if chunk_id == b"data":
                    data_end = handle.tell() + chunk_size
                    if data_end > file_size:
                        raise CineMixerError(
                            "E_TRACK_LOAD_FAIL",
                            (
                                f"source {path} is corrupt/truncated: data chunk "
                                f"ends at {data_end} but file is {file_size} bytes"
                            ),
                            4,
                        )
                    return
                # Chunks are word-aligned (padded to an even byte count).
                handle.seek(chunk_size + (chunk_size & 1), 1)
    except CineMixerError:
        raise
    except OSError as exc:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"cannot read source {path}: {exc}", 4) from exc


@dataclass(frozen=True)
class SourceInfo:
    sample_rate: int
    channels: int
    duration_seconds: float
    samples: int


def ensure_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise CineMixerError(
                "E_FFMPEG_MISSING",
                f"required binary {binary} was not found on PATH",
                7,
            )


def probe_source(path: str | Path) -> SourceInfo:
    """Probe a media file. Raises E_TRACK_LOAD_FAIL on any failure."""
    ensure_ffmpeg()
    source = Path(path)
    if not source.is_file():
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"source file missing: {source}", 4)
    if source.suffix.lower() == ".wav":
        _validate_wav_container(source)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"ffprobe failed for {source}: {exc}", 4) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or ["unknown ffprobe error"]
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"ffprobe error for {source}: {detail[0]}", 4)

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"ffprobe returned invalid JSON for {source}", 4) from exc

    streams = payload.get("streams") or []
    if not streams:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"no audio stream found in {source}", 4)
    stream = streams[0]
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"incomplete audio metadata in {source}", 4) from exc

    duration_text = stream.get("duration")
    if duration_text in (None, "N/A"):
        duration_text = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_text)
    except (TypeError, ValueError):
        duration = 0.0
    if not np.isfinite(duration) or duration <= 0.0:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"invalid duration metadata in {source}", 4)

    samples = int(round(duration * sample_rate))
    if samples <= 0:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"source {source} contains no audio samples", 4)
    return SourceInfo(sample_rate, channels, duration, samples)


def decode_source(path: str | Path, info: SourceInfo) -> np.ndarray:
    """Decode the full source into an ``(samples, channels)`` float32 array."""
    ensure_ffmpeg()
    source = Path(path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-err_detect",
        "explode",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(info.channels),
        "-ar",
        str(info.sample_rate),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=600
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"ffmpeg failed for {source}: {exc}", 4) from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        message = detail[0] if detail else "ffmpeg decode error"
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"decode error for {source}: {message}", 4)

    raw = completed.stdout
    frame_size = 4 * info.channels
    if len(raw) < frame_size or len(raw) % frame_size != 0:
        raise CineMixerError(
            "E_TRACK_LOAD_FAIL",
            f"decoded payload for {source} is truncated or corrupt ({len(raw)} bytes)",
            4,
        )
    audio = np.frombuffer(raw, dtype="<f4").astype(np.float64, copy=False)
    audio = audio.reshape(-1, info.channels)
    if audio.shape[0] == 0:
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"source {source} decoded to zero samples", 4)
    if not np.all(np.isfinite(audio)):
        raise CineMixerError("E_TRACK_LOAD_FAIL", f"source {source} contains non-finite samples", 4)
    # FFmpeg tolerates truncated payloads: compare the decoded length against
    # the duration advertised in the container header so mid-data truncation
    # (the deterministic "corrupt sample" scenario) is still rejected.
    shortfall = info.samples - audio.shape[0]
    tolerance = max(1, int(round(info.sample_rate * 0.01)))
    if shortfall > tolerance:
        raise CineMixerError(
            "E_TRACK_LOAD_FAIL",
            (
                f"source {source} is corrupt/truncated: header declares "
                f"{info.samples} frames but only {audio.shape[0]} decoded"
            ),
            4,
        )
    return audio


def load_source(path: str | Path) -> tuple[np.ndarray, SourceInfo]:
    """Probe and decode a source in one call."""
    info = probe_source(path)
    audio = decode_source(path, info)
    return audio, info
