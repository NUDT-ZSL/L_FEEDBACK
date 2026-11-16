"""Minimal dependency-free PCM WAV reader and writer."""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .errors import LipAlignError


PathLike = Union[str, os.PathLike]


@dataclass
class WavAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])


def _wav_error(message: str, code: str = "E_WAV_CORRUPT") -> LipAlignError:
    return LipAlignError(code, message, exit_code=2)


def read_wav(path: PathLike) -> WavAudio:
    """Read a 16-bit PCM WAV file with strict structural validation.

    Extra RIFF chunks (LIST, JUNK, fact, ...) are accepted.  Files with a
    truncated data chunk, inconsistent block alignment, or unsupported sample
    format are rejected with a business error.
    """

    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise LipAlignError(
            "E_INPUT", "Cannot read WAV file: {}".format(file_path), exit_code=2
        ) from exc

    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise _wav_error("Not a RIFF/WAVE file: {}".format(file_path))

    riff_size = struct.unpack_from("<I", raw, 4)[0]
    if riff_size < 4 or riff_size + 8 > len(raw):
        raise _wav_error("RIFF size exceeds file size: {}".format(file_path))

    pos = 12
    fmt: Optional[bytes] = None
    data: Optional[bytes] = None
    file_end = min(len(raw), riff_size + 8)
    while pos + 8 <= file_end:
        chunk_id = raw[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        payload_start = pos + 8
        payload_end = payload_start + chunk_size
        if payload_end > file_end:
            raise _wav_error(
                "Truncated {} chunk in {}".format(
                    chunk_id.decode("ascii", errors="replace"), file_path
                )
            )
        if chunk_id == b"fmt " and fmt is None:
            fmt = raw[payload_start:payload_end]
        elif chunk_id == b"data" and data is None:
            data = raw[payload_start:payload_end]
        pos = payload_end + (chunk_size & 1)

    if fmt is None:
        raise _wav_error("Missing fmt chunk: {}".format(file_path))
    if data is None:
        raise _wav_error("Missing data chunk: {}".format(file_path))
    if len(fmt) < 16:
        raise _wav_error("Short fmt chunk: {}".format(file_path))

    (
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt, 0)

    if channels < 1:
        raise _wav_error("Invalid channel count: {}".format(file_path))
    if sample_rate < 1:
        raise _wav_error("Invalid sample rate: {}".format(file_path))
    if bits_per_sample != 16:
        raise _wav_error(
            "Only 16-bit PCM is supported, got {}-bit: {}".format(
                bits_per_sample, file_path
            ),
            code="E_WAV_FORMAT",
        )

    effective_tag = format_tag
    if format_tag == 0xFFFE:
        if len(fmt) < 40:
            raise _wav_error("Short WAVE_FORMAT_EXTENSIBLE fmt chunk: {}".format(file_path))
        valid_bits = struct.unpack_from("<H", fmt, 18)[0]
        subformat = fmt[24:40]
        pcm_guid_prefix = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        if valid_bits not in (0, 16) or subformat != pcm_guid_prefix:
            raise _wav_error(
                "Only PCM subformat is supported: {}".format(file_path),
                code="E_WAV_FORMAT",
            )
        effective_tag = 1

    if effective_tag != 1:
        raise _wav_error(
            "Only integer PCM WAV files are supported: {}".format(file_path),
            code="E_WAV_FORMAT",
        )

    expected_align = channels * 2
    expected_byte_rate = sample_rate * expected_align
    if block_align != expected_align or byte_rate != expected_byte_rate:
        raise _wav_error(
            "Inconsistent WAV byte rate/block alignment: {}".format(file_path)
        )
    if len(data) % block_align != 0:
        raise _wav_error("Data chunk is not sample-aligned: {}".format(file_path))

    samples = np.frombuffer(data, dtype="<i2").reshape(-1, channels).copy()
    return WavAudio(samples=samples, sample_rate=int(sample_rate), channels=int(channels))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, str(path))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def pcm16_to_bytes(samples: np.ndarray) -> bytes:
    """Serialize an interleaved int16 array to canonical little-endian PCM."""

    array = np.asarray(samples)
    if array.dtype != np.int16:
        raise ValueError("pcm16_to_bytes requires int16 input")
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return np.ascontiguousarray(array, dtype="<i2").tobytes(order="C")


def write_pcm16_wav(
    path: PathLike,
    samples: np.ndarray,
    sample_rate: int,
    channels: Optional[int] = None,
    atomic: bool = True,
) -> None:
    """Write interleaved 16-bit PCM as a standard RIFF/WAVE file."""

    array = np.asarray(samples)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("samples must be a 1D or 2D array")
    if array.dtype != np.int16:
        raise ValueError("samples must have dtype int16")
    actual_channels = int(array.shape[1])
    if channels is not None and channels != actual_channels:
        raise ValueError("channel count does not match sample array")
    if actual_channels < 1 or actual_channels > 65535:
        raise ValueError("invalid channel count")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")

    data = pcm16_to_bytes(array)
    block_align = actual_channels * 2
    byte_rate = int(sample_rate) * block_align
    fmt_chunk = struct.pack(
        "<HHIIHH",
        1,
        actual_channels,
        int(sample_rate),
        byte_rate,
        block_align,
        16,
    )
    chunks = [
        b"WAVE",
        b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk,
        b"data" + struct.pack("<I", len(data)) + data,
    ]
    if len(data) & 1:
        chunks.append(b"\x00")
    body = b"".join(chunks)
    riff = b"RIFF" + struct.pack("<I", len(body)) + body
    out_path = Path(path)
    if atomic:
        _atomic_write_bytes(out_path, riff)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(riff)


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int16(samples: np.ndarray) -> str:
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return hashlib.sha256(pcm16_to_bytes(arr)).hexdigest()

