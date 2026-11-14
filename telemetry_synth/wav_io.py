"""PCM16 WAV writing and RIFF header verification."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from . import BITS_PER_SAMPLE, CHANNELS, SAMPLE_RATE
from .errors import SynthError
from .util import pcm16_bytes


def write_pcm16_wav(
    path: Path,
    expected_samples: int,
    blocks: Iterable[np.ndarray],
) -> int:
    """Write float blocks as stereo PCM16 and return the written frame count."""

    written = 0
    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(CHANNELS)
            handle.setsampwidth(BITS_PER_SAMPLE // 8)
            handle.setframerate(SAMPLE_RATE)
            handle.setnframes(expected_samples)
            for block in blocks:
                if block.ndim != 2 or block.shape[1] != CHANNELS:
                    raise SynthError(
                        "E_ALIGNMENT",
                        f"audio block must have shape (n, {CHANNELS}), got {block.shape}",
                    )
                handle.writeframesraw(pcm16_bytes(block))
                written += block.shape[0]
    except SynthError:
        raise
    except (OSError, wave.Error, ValueError) as exc:
        raise SynthError("E_WAV_IO", f"failed writing WAV {path}: {exc}") from exc
    return written


def _header_error(path: Path, message: str) -> None:
    raise SynthError("E_WAV_HEADER", f"{path}: {message}")


def validate_pcm16_wav(
    path: Path,
    expected_samples: int,
) -> dict[str, int | str]:
    """Parse RIFF chunks and verify every required PCM header field."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12:
                _header_error(path, "file is shorter than a RIFF header")
            riff, riff_size, wave_tag = struct.unpack("<4sI4s", header)
            if riff != b"RIFF":
                _header_error(path, "missing RIFF signature")
            if wave_tag != b"WAVE":
                _header_error(path, "missing WAVE signature")
            if riff_size != file_size - 8:
                _header_error(
                    path,
                    f"RIFF size {riff_size} does not match file size {file_size}",
                )

            fmt: tuple[int, int, int, int, int, int] | None = None
            data_size: int | None = None
            offset = 12
            while offset + 8 <= file_size:
                handle.seek(offset)
                chunk_header = handle.read(8)
                if len(chunk_header) != 8:
                    _header_error(path, f"truncated chunk header at offset {offset}")
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                payload_offset = offset + 8
                if payload_offset + chunk_size > file_size:
                    _header_error(
                        path,
                        f"chunk {chunk_id!r} extends beyond the end of file",
                    )
                if chunk_id == b"fmt ":
                    if chunk_size < 16:
                        _header_error(path, "fmt chunk is too short")
                    fmt_payload = handle.read(16)
                    fmt = struct.unpack("<HHIIHH", fmt_payload)
                elif chunk_id == b"data":
                    data_size = chunk_size
                offset = payload_offset + chunk_size + (chunk_size & 1)

        if fmt is None:
            _header_error(path, "missing fmt chunk")
        if data_size is None:
            _header_error(path, "missing data chunk")

        audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
        expected_data_size = expected_samples * CHANNELS * (BITS_PER_SAMPLE // 8)
        if audio_format != 1:
            _header_error(path, f"audio format must be PCM (1), got {audio_format}")
        if channels != CHANNELS:
            _header_error(path, f"channel count must be {CHANNELS}, got {channels}")
        if sample_rate != SAMPLE_RATE:
            _header_error(
                path, f"sample rate must be {SAMPLE_RATE}, got {sample_rate}"
            )
        if bits != BITS_PER_SAMPLE:
            _header_error(path, f"bit depth must be {BITS_PER_SAMPLE}, got {bits}")
        if block_align != CHANNELS * (BITS_PER_SAMPLE // 8):
            _header_error(path, f"invalid block alignment {block_align}")
        if byte_rate != SAMPLE_RATE * block_align:
            _header_error(path, f"invalid byte rate {byte_rate}")
        if data_size != expected_data_size:
            _header_error(
                path,
                f"data chunk has {data_size} bytes; expected {expected_data_size}",
            )

        with wave.open(str(path), "rb") as handle:
            actual_frames = handle.getnframes()
            if actual_frames != expected_samples:
                _header_error(
                    path,
                    f"WAV frame count is {actual_frames}, expected {expected_samples}",
                )
            if handle.getframerate() != SAMPLE_RATE:
                _header_error(path, "wave module reported an unexpected sample rate")
    except SynthError:
        raise
    except (OSError, EOFError, wave.Error, struct.error) as exc:
        raise SynthError("E_WAV_IO", f"cannot inspect WAV {path}: {exc}") from exc

    return {
        "chunk_size_bytes": int(riff_size),
        "data_size_bytes": int(data_size),
        "frame_count": int(expected_samples),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "bits_per_sample": int(bits),
        "block_align": int(block_align),
        "byte_rate": int(byte_rate),
    }

