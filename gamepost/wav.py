from __future__ import annotations

import hashlib
import os
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import PipelineError


@dataclass(frozen=True)
class AudioData:
    path: Path
    sample_rate: int
    channels: int
    pcm: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.pcm.shape[0])

    @property
    def duration_ms(self) -> float:
        return self.frames * 1000.0 / self.sample_rate

    def as_float(self) -> np.ndarray:
        return self.pcm.astype(np.float64) / 32768.0

    @property
    def original_peak(self) -> int:
        if self.pcm.size == 0:
            return 0
        return int(np.max(np.abs(self.pcm.astype(np.int32))))

    def describe(self) -> dict:
        return {
            "path": str(self.path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration_ms": round(self.duration_ms, 6),
            "bit_depth": 16,
            "original_peak": self.original_peak,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wav(path: str | Path, expected_sample_rate: int = 48_000) -> AudioData:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise ValueError("compressed WAV is not supported")
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            width = source.getsampwidth()
            frames = source.getnframes()
            payload = source.readframes(frames)
    except Exception as exc:
        raise PipelineError(
            f"Cannot read WAV input: {path}",
            code="INPUT_WAV_INVALID",
            details={"path": str(path), "reason": str(exc)},
        ) from exc

    if sample_rate != expected_sample_rate:
        raise PipelineError(
            f"WAV sample rate must be {expected_sample_rate} Hz: {path}",
            code="INPUT_WAV_INVALID",
            details={"path": str(path), "sample_rate": sample_rate},
        )
    if channels not in (1, 2):
        raise PipelineError(
            f"WAV must be mono or stereo: {path}",
            code="INPUT_WAV_INVALID",
            details={"path": str(path), "channels": channels},
        )
    if width != 2:
        raise PipelineError(
            f"WAV must be 16-bit PCM: {path}",
            code="INPUT_WAV_INVALID",
            details={"path": str(path), "sample_width": width},
        )
    if frames <= 0:
        raise PipelineError(
            f"WAV must contain at least one sample frame: {path}",
            code="INPUT_WAV_INVALID",
            details={"path": str(path), "frames": frames},
        )
    expected_bytes = frames * channels * 2
    if len(payload) != expected_bytes:
        raise PipelineError(
            f"WAV payload length does not match its header: {path}",
            code="INPUT_WAV_INVALID",
            details={
                "path": str(path),
                "expected_bytes": expected_bytes,
                "actual_bytes": len(payload),
            },
        )
    pcm = np.frombuffer(payload, dtype="<i2").reshape(frames, channels).copy()
    return AudioData(path=path, sample_rate=sample_rate, channels=channels, pcm=pcm)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    values = np.asarray(samples, dtype=np.float64)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("non-finite sample in output")
    quantized = np.rint(np.clip(values, -1.0, 1.0) * 32768.0)
    quantized = np.clip(quantized, -32768.0, 32767.0).astype("<i2", copy=False)
    return quantized.tobytes(order="C")


def _wav_header(frames: int, sample_rate: int, channels: int) -> bytes:
    block_align = channels * 2
    data_bytes = frames * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
        b"data",
        data_bytes,
    )


def initialize_partial_wav(
    path: str | Path,
    frames: int,
    sample_rate: int,
    channels: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _wav_header(frames, sample_rate, channels)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.truncate(44 + frames * channels * 2)
        handle.flush()
        os.fsync(handle.fileno())


def write_wav_chunk(
    path: str | Path,
    start_frame: int,
    samples: np.ndarray,
    *,
    channels: int,
) -> bytes:
    data = pcm16_bytes(samples)
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape[1] != channels:
        raise ValueError("chunk channel count mismatch")
    expected = values.shape[0] * channels * 2
    if len(data) != expected:
        raise ValueError("chunk byte count mismatch")
    offset = 44 + start_frame * channels * 2
    with open(path, "r+b") as handle:
        handle.seek(offset)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return data


def validate_wav_header(
    path: str | Path,
    frames: int,
    sample_rate: int,
    channels: int,
) -> bool:
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            header = handle.read(44)
    except OSError:
        return False
    return header == _wav_header(frames, sample_rate, channels)

