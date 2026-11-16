from __future__ import annotations

import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError
from .io_utils import db_to_linear, linear_to_db


SAMPLE_RATE = 48_000
BIT_DEPTH = 16


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int
    source_channels: int

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


@dataclass(frozen=True)
class SilenceTrim:
    threshold_dbfs: float
    margin_ms: float
    start_frame: int
    end_frame: int
    original_frames: int
    fully_silent: bool

    @property
    def trimmed_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def lead_frames(self) -> int:
        return self.start_frame

    @property
    def trail_frames(self) -> int:
        return self.original_frames - self.end_frame

    @property
    def total_silence_frames(self) -> int:
        return self.lead_frames + self.trail_frames


def load_mono_wav(
    path: str | Path,
    *,
    entry_id: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> LoadedAudio:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            compression = handle.getcomptype()
            frame_count = handle.getnframes()
            payload = handle.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise BusinessError(
            code="E_AUDIO_READ",
            message=f"Unable to read WAV audio: {exc}",
            entry_id=entry_id,
            path=str(source),
        ) from exc

    if compression != "NONE":
        raise BusinessError(
            code="E_AUDIO_FORMAT",
            message="WAV compression must be uncompressed PCM.",
            entry_id=entry_id,
            path=str(source),
            details={"compression": compression},
        )
    if channels != 1 or sample_width != 2 or rate != sample_rate:
        raise BusinessError(
            code="E_AUDIO_FORMAT",
            message=(
                f"Expected 48 kHz, 16-bit PCM, mono audio; got "
                f"{rate} Hz, {sample_width * 8}-bit, {channels} channel(s)."
            ),
            entry_id=entry_id,
            path=str(source),
            details={
                "required": {
                    "sample_rate_hz": sample_rate,
                    "bit_depth": 16,
                    "channels": 1,
                },
                "actual": {
                    "sample_rate_hz": rate,
                    "bit_depth": sample_width * 8,
                    "channels": channels,
                },
            },
        )
    if frame_count <= 0:
        raise BusinessError(
            code="E_AUDIO_FORMAT",
            message="WAV file contains no audio frames.",
            entry_id=entry_id,
            path=str(source),
        )
    expected_bytes = frame_count * 2
    if len(payload) < expected_bytes:
        raise BusinessError(
            code="E_AUDIO_READ",
            message="WAV data chunk is truncated.",
            entry_id=entry_id,
            path=str(source),
        )
    samples = np.frombuffer(payload[:expected_bytes], dtype="<i2").astype(np.float64)
    samples /= 32768.0
    return LoadedAudio(samples=samples, sample_rate=rate, source_channels=channels)


def trim_silence(
    samples: np.ndarray,
    *,
    sample_rate: int,
    threshold_dbfs: float,
    margin_ms: float,
) -> tuple[np.ndarray, SilenceTrim]:
    threshold = db_to_linear(threshold_dbfs)
    absolute = np.abs(samples)
    active = np.flatnonzero(absolute >= threshold)
    if active.size == 0:
        metadata = SilenceTrim(
            threshold_dbfs=threshold_dbfs,
            margin_ms=margin_ms,
            start_frame=0,
            end_frame=0,
            original_frames=int(samples.shape[0]),
            fully_silent=True,
        )
        return samples[:0].copy(), metadata

    margin_frames = max(0, int(round(margin_ms * sample_rate / 1000.0)))
    start = max(0, int(active[0]) - margin_frames)
    end = min(int(samples.shape[0]), int(active[-1]) + 1 + margin_frames)
    metadata = SilenceTrim(
        threshold_dbfs=threshold_dbfs,
        margin_ms=margin_ms,
        start_frame=start,
        end_frame=end,
        original_frames=int(samples.shape[0]),
        fully_silent=False,
    )
    return samples[start:end].copy(), metadata


def _no_window_creation_flags() -> int:
    if os.name == "nt":
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return 0


def time_stretch(
    samples: np.ndarray,
    *,
    ratio: float,
    sample_rate: int,
    ffmpeg_bin: str,
    entry_id: str | None = None,
    audio_path: str | None = None,
) -> np.ndarray:
    if not math.isfinite(ratio) or ratio <= 0:
        raise BusinessError(
            code="E_STRETCH_FAILED",
            message="Time-stretch ratio must be finite and greater than zero.",
            entry_id=entry_id,
            path=audio_path,
            details={"ratio": ratio},
        )
    pcm = np.ascontiguousarray(
        np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")
    )
    request = pcm.tobytes()
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-af",
        f"atempo={ratio:.9f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            creationflags=_no_window_creation_flags(),
        )
    except FileNotFoundError as exc:
        raise BusinessError(
            code="E_FFMPEG_NOT_FOUND",
            message=(
                "ffmpeg was not found. Install FFmpeg 6.x or newer and ensure "
                "ffmpeg is on PATH, or pass --ffmpeg-bin."
            ),
            entry_id=entry_id,
            path=audio_path,
        ) from exc
    except OSError as exc:
        raise BusinessError(
            code="E_STRETCH_FAILED",
            message=f"Unable to launch ffmpeg: {exc}",
            entry_id=entry_id,
            path=audio_path,
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BusinessError(
            code="E_STRETCH_FAILED",
            message="ffmpeg atempo time-stretch failed.",
            entry_id=entry_id,
            path=audio_path,
            details={
                "ratio": ratio,
                "ffmpeg_returncode": completed.returncode,
                "ffmpeg_stderr": stderr[-2000:],
            },
        )
    usable = len(completed.stdout) - (len(completed.stdout) % 2)
    if usable <= 0:
        raise BusinessError(
            code="E_STRETCH_FAILED",
            message="ffmpeg returned no audio samples.",
            entry_id=entry_id,
            path=audio_path,
            details={"ratio": ratio},
        )
    result = np.frombuffer(completed.stdout[:usable], dtype="<i2").astype(np.float64)
    result /= 32768.0
    return result


def peak_dbfs(signal: np.ndarray) -> float:
    if signal.size == 0:
        return -math.inf
    peak = float(np.max(np.abs(signal)))
    value = linear_to_db(peak)
    return -math.inf if value is None else value


def rms_dbfs(signal: np.ndarray) -> float:
    if signal.size == 0:
        return -math.inf
    mean_square = float(np.mean(np.square(signal, dtype=np.float64)))
    value = linear_to_db(math.sqrt(max(0.0, mean_square)))
    return -math.inf if value is None else value


def float_to_pcm16(signal: np.ndarray) -> np.ndarray:
    clipped = np.clip(signal, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2")


def pcm16_bytes(signal: np.ndarray) -> bytes:
    if signal.ndim == 1:
        return float_to_pcm16(signal).tobytes()
    return np.ascontiguousarray(float_to_pcm16(signal), dtype="<i2").tobytes()


def wav_info(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            return {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate_hz": handle.getframerate(),
                "frames": handle.getnframes(),
                "duration_s": handle.getnframes() / float(handle.getframerate()),
                "compression": handle.getcomptype(),
            }
    except (OSError, EOFError, wave.Error) as exc:
        raise BusinessError(
            code="E_AUDIO_READ",
            message=f"Unable to inspect WAV output: {exc}",
            path=str(path),
        ) from exc

