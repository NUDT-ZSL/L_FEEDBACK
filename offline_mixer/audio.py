"""FFmpeg-backed WAV probing, decoding, and 24-bit PCM writing."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import OutputError, ToolchainError
from .manifest import SUPPORTED_SAMPLE_RATE


@dataclass(frozen=True)
class AudioProbe:
    channels: int
    sample_rate: int | None
    codec_name: str | None
    channel_layout: str | None
    duration_s: float | None


def find_tool(name: str, *, required: bool = True) -> str | None:
    path = shutil.which(name)
    if path is None and required:
        raise ToolchainError(
            "MIX1005",
            f"{name} executable was not found on PATH; Windows setup must expose it as '{name}.exe'",
        )
    return path


def check_toolchain() -> tuple[str, str | None]:
    ffmpeg = find_tool("ffmpeg", required=True)
    ffprobe = find_tool("ffprobe", required=False)
    assert ffmpeg is not None
    return ffmpeg, ffprobe


def _ffprobe_json(path: Path, ffprobe: str) -> dict[str, Any] | None:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,sample_rate,codec_name,channel_layout",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


_FFMPEG_CHANNEL_PATTERNS = {
    "mono": 1,
    "stereo": 2,
    "2.1": 3,
    "3.0": 3,
    "4.0": 4,
    "quad": 4,
    "5.0": 5,
    "5.1": 6,
    "6.1": 7,
    "7.1": 8,
}


def _probe_with_ffmpeg_stderr(path: Path, ffmpeg: str) -> AudioProbe | None:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    for line in completed.stderr.splitlines():
        if "Audio:" not in line:
            continue
        match = re.search(
            r"(\d+)\s+Hz,\s+(mono|stereo|2\.1|3\.0|4\.0|quad|5\.0|5\.1|6\.1|7\.1)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            sample_rate = int(match.group(1))
            channels = _FFMPEG_CHANNEL_PATTERNS[match.group(2).lower()]
            codec_match = re.search(r"Audio:\s+([^\s,]+)", line)
            return AudioProbe(
                channels=channels,
                sample_rate=sample_rate,
                codec_name=codec_match.group(1) if codec_match else None,
                channel_layout=match.group(2).lower(),
                duration_s=None,
            )
    return None


def probe_audio(path: str | Path, ffprobe: str | None = None) -> AudioProbe:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))
    if not audio_path.is_file():
        raise OSError(f"not a regular file: {audio_path}")

    wave_error: Exception | None = None
    try:
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            if channels <= 0:
                raise ValueError("WAV declares zero channels")
            return AudioProbe(
                channels=channels,
                sample_rate=sample_rate,
                codec_name=f"pcm_s{wav.getsampwidth() * 8}le",
                channel_layout={1: "mono", 2: "stereo", 6: "5.1"}.get(
                    channels, None
                ),
                duration_s=(frames / sample_rate) if sample_rate > 0 else None,
            )
    except Exception as exc:  # wave raises several exception classes
        wave_error = exc

    if ffprobe:
        payload = _ffprobe_json(audio_path, ffprobe)
        if payload:
            streams = payload.get("streams") or []
            if streams:
                stream = streams[0]
                try:
                    channels = int(stream["channels"])
                except (KeyError, TypeError, ValueError):
                    channels = 0
                if channels > 0:
                    sample_rate_raw = stream.get("sample_rate")
                    sample_rate = (
                        int(sample_rate_raw)
                        if sample_rate_raw not in (None, "", "N/A")
                        else None
                    )
                    duration_raw = (payload.get("format") or {}).get("duration")
                    duration = (
                        float(duration_raw)
                        if duration_raw not in (None, "", "N/A")
                        else None
                    )
                    return AudioProbe(
                        channels=channels,
                        sample_rate=sample_rate,
                        codec_name=stream.get("codec_name"),
                        channel_layout=stream.get("channel_layout"),
                        duration_s=duration,
                    )

    ffmpeg = find_tool("ffmpeg", required=True)
    fallback = _probe_with_ffmpeg_stderr(audio_path, ffmpeg)
    if fallback is not None:
        return fallback
    raise OSError(
        f"unsupported or corrupt audio container ({wave_error})"
    )


def decode_audio(
    path: str | Path,
    *,
    ffmpeg: str,
    ffprobe: str | None = None,
) -> tuple[np.ndarray, AudioProbe]:
    audio_path = Path(path)
    probe = probe_audio(audio_path, ffprobe=ffprobe)
    channels = probe.channels
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(channels),
        "-ar",
        str(SUPPORTED_SAMPLE_RATE),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"ffmpeg decode failed: {detail[-500:] or 'unknown error'}")
    raw = completed.stdout
    if not raw:
        raise OSError("ffmpeg decoded zero audio samples")
    if len(raw) % (4 * channels) != 0:
        raise OSError(
            "ffmpeg returned an incomplete interleaved float32 buffer "
            f"({len(raw)} bytes for {channels} channels)"
        )
    signal = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(-1, channels)
    if signal.shape[0] == 0:
        raise OSError("decoded audio is empty")
    if not np.all(np.isfinite(signal)):
        raise OSError("decoded audio contains NaN or infinity")
    return signal, probe


def _verify_encoded_wav(
    path: Path,
    *,
    channels: int,
    ffprobe: str | None,
) -> None:
    if ffprobe:
        payload = _ffprobe_json(path, ffprobe)
        if payload:
            streams = payload.get("streams") or []
            if streams:
                stream = streams[0]
                codec = stream.get("codec_name")
                sample_rate = stream.get("sample_rate")
                actual_channels = stream.get("channels")
                if codec != "pcm_s24le":
                    raise OSError(f"expected pcm_s24le, ffprobe reports {codec!r}")
                if str(sample_rate) != str(SUPPORTED_SAMPLE_RATE):
                    raise OSError(
                        f"expected {SUPPORTED_SAMPLE_RATE} Hz, ffprobe reports {sample_rate}"
                    )
                if str(actual_channels) != str(channels):
                    raise OSError(
                        f"expected {channels} channels, ffprobe reports {actual_channels}"
                    )
                return
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 3:
                raise OSError(
                    f"expected 24-bit PCM, WAV reports {wav.getsampwidth() * 8}-bit"
                )
            if wav.getframerate() != SUPPORTED_SAMPLE_RATE:
                raise OSError(
                    f"expected {SUPPORTED_SAMPLE_RATE} Hz, WAV reports {wav.getframerate()}"
                )
            if wav.getnchannels() != channels:
                raise OSError(
                    f"expected {channels} channels, WAV reports {wav.getnchannels()}"
                )
            return
    except Exception:
        pass

    # Python 3.11's wave module does not understand WAVE_FORMAT_EXTENSIBLE,
    # which FFmpeg uses for 5.1. Validate the RIFF fmt chunk directly.
    try:
        header = path.read_bytes()[:128]
        if len(header) < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise OSError("not a RIFF/WAVE file")
        position = 12
        fmt: bytes | None = None
        while position + 8 <= len(header):
            chunk_id = header[position : position + 4]
            chunk_size = int.from_bytes(
                header[position + 4 : position + 8], "little"
            )
            data_start = position + 8
            if chunk_id == b"fmt ":
                fmt = header[data_start : data_start + chunk_size]
                break
            position = data_start + chunk_size + (chunk_size & 1)
        if fmt is None or len(fmt) < 16:
            raise OSError("missing WAV fmt chunk")
        format_tag = int.from_bytes(fmt[0:2], "little")
        actual_channels = int.from_bytes(fmt[2:4], "little")
        actual_rate = int.from_bytes(fmt[4:8], "little")
        bits = int.from_bytes(fmt[14:16], "little")
        if format_tag == 0xFFFE and len(fmt) >= 20:
            valid_bits = int.from_bytes(fmt[18:20], "little")
            if valid_bits:
                bits = valid_bits
        if format_tag not in {1, 0xFFFE}:
            raise OSError(f"unexpected WAV format tag {format_tag}")
        if bits != 24:
            raise OSError(f"expected 24-bit PCM, fmt chunk reports {bits}-bit")
        if actual_rate != SUPPORTED_SAMPLE_RATE:
            raise OSError(
                f"expected {SUPPORTED_SAMPLE_RATE} Hz, fmt chunk reports {actual_rate}"
            )
        if actual_channels != channels:
            raise OSError(
                f"expected {channels} channels, fmt chunk reports {actual_channels}"
            )
    except Exception as exc:
        raise OSError(f"output WAV verification failed: {exc}") from exc


def write_wav_24(
    path: str | Path,
    signal: np.ndarray,
    *,
    sample_rate: int,
    channels: int,
    channel_layout: str,
    ffmpeg: str,
    ffprobe: str | None,
) -> None:
    if sample_rate != SUPPORTED_SAMPLE_RATE:
        raise OutputError("MIX3001", "output sample rate must be 48000 Hz")
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != channels:
        raise OutputError(
            "MIX3001",
            f"output signal must have shape (samples, {channels})",
        )
    if array.shape[0] == 0:
        raise OutputError("MIX3001", "refusing to write an empty output WAV")
    if not np.all(np.isfinite(array)):
        raise OutputError("MIX3001", "output signal contains NaN or infinity")
    if float(np.max(np.abs(array))) > 1.0:
        raise OutputError(
            "MIX3001",
            "output signal exceeds full scale before 24-bit quantization",
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".tmp.wav",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        interleaved = np.ascontiguousarray(array, dtype="<f4")
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-channel_layout",
            channel_layout,
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_s24le",
            "-ar",
            str(sample_rate),
            "-channel_layout",
            channel_layout,
            "-map_metadata",
            "-1",
            "-y",
            str(temporary),
        ]
        completed = subprocess.run(
            command,
            input=interleaved.tobytes(order="C"),
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise OSError(f"ffmpeg PCM writer failed: {detail[-500:] or 'unknown error'}")
        _verify_encoded_wav(temporary, channels=channels, ffprobe=ffprobe)
        os.replace(temporary, destination)
    except OSError as exc:
        raise OutputError("MIX3001", f"cannot write {destination}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except OSError as exc:
        raise OutputError("MIX3001", f"cannot write metadata {destination}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
