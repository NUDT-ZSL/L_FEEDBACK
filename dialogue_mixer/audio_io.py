"""WAV output and FFmpeg-backed audio decoding.

The tool deliberately keeps its Python dependencies small.  FFmpeg is used to
decode, downmix and resample arbitrary source audio to the project sample rate.
The mix itself is written as a 32-bit IEEE-float mono WAV, which avoids integer
quantization changing the measured peak.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class AudioDecodeError(RuntimeError):
    """Raised when the source audio cannot be decoded."""


def _ffmpeg_creation_flags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Decode *path* to mono float64 at *sample_rate* using FFmpeg."""

    if not shutil.which("ffmpeg"):
        raise AudioDecodeError("ffmpeg was not found in PATH")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            creationflags=_ffmpeg_creation_flags(),
        )
    except OSError as exc:
        raise AudioDecodeError("failed to start ffmpeg: {0}".format(exc)) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(stderr or "ffmpeg returned exit code {0}".format(result.returncode))

    if len(result.stdout) % 4:
        raise AudioDecodeError("ffmpeg returned a truncated float32 stream")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64, copy=True)
    if samples.size == 0:
        raise AudioDecodeError("decoded audio is empty")
    if not np.all(np.isfinite(samples)):
        raise AudioDecodeError("decoded audio contains NaN or infinity")
    return samples


def probe_wav_basic(path: Path) -> Optional[Tuple[int, int, int]]:
    """Return (sample_rate, channels, bits) for ordinary PCM WAV files."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                int(wav_file.getframerate()),
                int(wav_file.getnchannels()),
                int(wav_file.getsampwidth() * 8),
            )
    except (wave.Error, OSError, EOFError):
        return None


def _wav_float32_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(samples, dtype="<f4").reshape(-1)
    data = mono.tobytes(order="C")
    fact = struct.pack("<I", int(mono.size))
    fmt = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        1,  # mono
        int(sample_rate),
        int(sample_rate * 4),
        4,
        32,
    )

    chunks = b"WAVE"
    chunks += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"fact" + struct.pack("<I", len(fact)) + fact
    chunks += b"data" + struct.pack("<I", len(data)) + data
    riff_size = len(chunks)
    return b"RIFF" + struct.pack("<I", riff_size) + chunks


def write_wav_float32(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Atomically write a mono 32-bit float WAV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _wav_float32_bytes(samples, sample_rate)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def decode_wav_float32(path: Path) -> Tuple[np.ndarray, int]:
    """Small reader used by tests and callers that want the produced mix."""

    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioDecodeError("not a RIFF/WAVE file")

    position = 12
    sample_rate = None
    channels = None
    bits = None
    format_tag = None
    data = None
    while position + 8 <= len(raw):
        chunk_id = raw[position : position + 4]
        chunk_size = struct.unpack("<I", raw[position + 4 : position + 8])[0]
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(raw):
            raise AudioDecodeError("truncated WAV chunk")
        payload = raw[payload_start:payload_end]
        if chunk_id == b"fmt ":
            if len(payload) < 16:
                raise AudioDecodeError("short WAV fmt chunk")
            format_tag, channels, sample_rate, _, _, bits = struct.unpack(
                "<HHIIHH", payload[:16]
            )
        elif chunk_id == b"data":
            data = payload
        position = payload_end + (chunk_size & 1)

    if format_tag != 3 or bits != 32 or channels != 1:
        raise AudioDecodeError("expected mono 32-bit float WAV")
    if data is None or sample_rate is None:
        raise AudioDecodeError("WAV is missing data or fmt information")
    samples = np.frombuffer(data, dtype="<f4").astype(np.float64, copy=True)
    return samples, int(sample_rate)

