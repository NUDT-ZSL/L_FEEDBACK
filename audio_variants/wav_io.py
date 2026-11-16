from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class WavError(ValueError):
    """Raised when a WAV file cannot be decoded or uses unsupported formatting."""


SUPPORTED_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "FLOAT"}
SUPPORTED_FORMATS = {"WAV", "WAVEX"}


@dataclass(frozen=True)
class WavInfo:
    format: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int

    @property
    def output_subtype(self) -> str:
        return "FLOAT" if self.subtype == "FLOAT" else self.subtype


def inspect_wav(path: Path) -> WavInfo:
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as exc:
        raise WavError(f"cannot decode WAV file: {exc}") from exc
    if info.format not in SUPPORTED_FORMATS:
        raise WavError(f"unsupported container {info.format!r}; expected WAV")
    if info.subtype not in SUPPORTED_SUBTYPES:
        raise WavError(
            f"unsupported bit depth/encoding {info.subtype!r}; "
            "supported: 16/24/32-bit PCM and 32-bit float"
        )
    if info.channels not in (1, 2):
        raise WavError(
            f"unsupported channel count {info.channels}; only mono and stereo are supported"
        )
    if info.frames <= 0:
        raise WavError("WAV file contains no audio frames")
    return WavInfo(
        format=info.format,
        subtype=info.subtype,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
    )


def read_wav(path: Path) -> tuple[np.ndarray, WavInfo]:
    info = inspect_wav(path)
    try:
        data, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True, frames=info.frames
        )
    except (RuntimeError, OSError) as exc:
        raise WavError(f"cannot decode WAV file: {exc}") from exc
    if int(sample_rate) != info.sample_rate:
        raise WavError("decoded sample rate does not match file header")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != info.channels:
        raise WavError("decoded channel layout does not match file header")
    if not np.all(np.isfinite(data)):
        raise WavError("WAV file contains NaN or infinite samples")
    return data, info


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    if subtype not in SUPPORTED_SUBTYPES:
        raise WavError(f"unsupported output subtype: {subtype}")
    data = np.asarray(audio, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise WavError("refusing to write non-finite audio samples")
    if data.size and float(np.max(np.abs(data))) > 0.999999:
        raise WavError("refusing to write potentially clipped audio samples")
    try:
        sf.write(
            str(path),
            data,
            int(sample_rate),
            format="WAV",
            subtype=subtype,
        )
    except (RuntimeError, OSError) as exc:
        raise WavError(f"cannot write WAV file: {exc}") from exc
    _clear_peak_chunk_timestamp(path)


def _clear_peak_chunk_timestamp(path: Path) -> None:
    """Remove wall-clock metadata so identical generations are byte-identical."""
    try:
        with path.open("r+b") as handle:
            header = handle.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return
            while True:
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                payload_position = handle.tell()
                if chunk_id == b"PEAK" and chunk_size >= 8:
                    handle.seek(payload_position + 4)
                    handle.write(b"\x00\x00\x00\x00")
                    return
                handle.seek(payload_position + chunk_size + (chunk_size & 1))
    except OSError as exc:
        raise WavError(f"cannot finalize WAV metadata: {exc}") from exc
