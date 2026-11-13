"""Minimal, deterministic RIFF/WAVE reader and writer.

Only the Python standard library and NumPy's array operations are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import struct
from typing import Any

import numpy as np

from .util import round_half_even_array


class WavError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class WavHeader:
    path: str
    format_tag: int
    sample_rate: int
    channels: int
    bit_depth: int
    valid_bits: int
    block_align: int
    data_offset: int
    data_size: int
    file_size: int
    encoding: str

    @property
    def frame_count(self) -> int:
        if self.block_align <= 0:
            return 0
        return self.data_size // self.block_align


def _read_exact(handle: Any, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise WavError("SOURCE_READ_ERROR", "truncated RIFF chunk header")
    return data


def read_wav_header(path: str) -> WavHeader:
    """Read and validate only the RIFF metadata; the data payload is not read."""
    try:
        file_size = os.path.getsize(path)
    except OSError as exc:
        raise WavError("SOURCE_READ_ERROR", f"cannot stat source: {exc}") from exc
    with open(path, "rb") as handle:
        riff = _read_exact(handle, 12)
        riff_id, riff_size, wave_id = struct.unpack("<4sI4s", riff)
        if riff_id != b"RIFF" or wave_id != b"WAVE":
            raise WavError("SOURCE_READ_ERROR", "not a little-endian RIFF/WAVE file")
        if riff_size + 8 > file_size:
            # A short RIFF payload can still be a complete header followed by a
            # truncated data chunk. Full reads report that condition later.
            pass
        fmt: tuple[int, int, int, int, int] | None = None
        data_offset = -1
        data_size = 0
        while True:
            header = handle.read(8)
            if len(header) == 0:
                break
            if len(header) != 8:
                raise WavError("SOURCE_READ_ERROR", "truncated RIFF chunk header")
            chunk_id, chunk_size = struct.unpack("<4sI", header)
            payload_pos = handle.tell()
            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise WavError("SOURCE_READ_ERROR", "fmt chunk is shorter than 16 bytes")
                raw = _read_exact(handle, min(chunk_size, 64))
                tag, channels, sample_rate, _byte_rate, block_align, bits = struct.unpack("<HHIIHH", raw[:16])
                valid_bits = bits
                if tag == 0xFFFE:
                    if len(raw) < 40:
                        raise WavError("SOURCE_READ_ERROR", "extensible fmt chunk is incomplete")
                    valid_bits = struct.unpack("<H", raw[18:20])[0] or bits
                    sub_tag = struct.unpack("<H", raw[24:26])[0]
                    tag = sub_tag
                fmt = (tag, channels, sample_rate, block_align, bits, valid_bits)
            elif chunk_id == b"data":
                data_offset = payload_pos
                data_size = chunk_size
            next_pos = payload_pos + chunk_size + (chunk_size & 1)
            if next_pos > file_size:
                # Still return a parsed header so the full read can distinguish
                # truncation from unsupported format.
                handle.seek(file_size)
            else:
                handle.seek(next_pos)
            if fmt is not None and data_offset >= 0:
                break
    if fmt is None:
        raise WavError("SOURCE_READ_ERROR", "missing fmt chunk")
    if data_offset < 0:
        raise WavError("SOURCE_READ_ERROR", "missing data chunk")
    tag, channels, sample_rate, block_align, bits, valid_bits = fmt
    if tag == 1:
        if bits not in (16, 24, 32):
            raise WavError("SOURCE_READ_ERROR", f"unsupported PCM bit depth {bits}")
        encoding = f"pcm{bits}"
    elif tag == 3:
        if bits != 32:
            raise WavError("SOURCE_READ_ERROR", f"unsupported float bit depth {bits}")
        encoding = "float32"
    else:
        raise WavError("SOURCE_READ_ERROR", f"unsupported WAVE format tag {tag}")
    if channels not in (1, 2):
        raise WavError("SOURCE_READ_ERROR", f"unsupported channel count {channels}")
    if sample_rate <= 0 or block_align <= 0:
        raise WavError("SOURCE_READ_ERROR", "invalid sample rate or block alignment")
    expected_align = channels * ((bits + 7) // 8)
    if block_align != expected_align:
        raise WavError("SOURCE_READ_ERROR", "block alignment is inconsistent with format")
    return WavHeader(
        path=path,
        format_tag=tag,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bits,
        valid_bits=valid_bits,
        block_align=block_align,
        data_offset=data_offset,
        data_size=data_size,
        file_size=file_size,
        encoding=encoding,
    )


def read_wav(path: str) -> tuple[np.ndarray, WavHeader]:
    """Read all sample frames as float64 in the range [-1, 1)."""
    header = read_wav_header(path)
    if header.data_offset + header.data_size > header.file_size:
        raise WavError(
            "SOURCE_READ_ERROR",
            f"truncated data chunk: declares {header.data_size} bytes, only "
            f"{max(0, header.file_size - header.data_offset)} available",
        )
    if header.data_size % header.block_align != 0:
        raise WavError("SOURCE_READ_ERROR", "data chunk does not contain whole frames")
    with open(path, "rb") as handle:
        handle.seek(header.data_offset)
        raw = handle.read(header.data_size)
    if len(raw) != header.data_size:
        raise WavError("SOURCE_READ_ERROR", "data chunk is truncated")
    channels = header.channels
    if header.encoding == "float32":
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    elif header.encoding == "pcm16":
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif header.encoding == "pcm24":
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float64) / 8388608.0
    else:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    if samples.size % channels != 0:
        raise WavError("SOURCE_READ_ERROR", "sample count is not divisible by channels")
    if not np.all(np.isfinite(samples)):
        raise WavError("SOURCE_READ_ERROR", "source contains NaN or infinite samples")
    return samples.reshape(-1, channels), header


_FMT_PCM = 1
_FMT_FLOAT = 3


def _encode_samples(samples: np.ndarray, encoding: str) -> tuple[bytes, int, int]:
    data = np.asarray(samples, dtype=np.float64)
    channels = int(data.shape[1])
    if encoding == "float32":
        payload = data.astype("<f4", copy=False).tobytes(order="C")
        return payload, 32, _FMT_FLOAT
    if encoding == "pcm16":
        bits = 16
        maximum = 32767
    elif encoding == "pcm24":
        bits = 24
        maximum = 8388607
    elif encoding == "pcm32":
        bits = 32
        maximum = 2147483647
    else:
        raise WavError("UNSUPPORTED_ENCODING", f"unsupported output encoding {encoding}")
    scaled = data * float(maximum)
    scaled = np.clip(scaled, -float(maximum + 1), float(maximum))
    rounded = round_half_even_array(scaled)
    if bits == 16:
        payload = rounded.astype("<i2").tobytes(order="C")
    elif bits == 24:
        flat = rounded.reshape(-1).astype(np.int64)
        low = (flat & 0xFF).astype(np.uint8)
        middle = ((flat >> 8) & 0xFF).astype(np.uint8)
        high = ((flat >> 16) & 0xFF).astype(np.uint8)
        payload = np.stack((low, middle, high), axis=1).tobytes(order="C")
    else:
        payload = rounded.astype("<i4").tobytes(order="C")
    return payload, bits, _FMT_PCM


def write_wav(path: str, samples: np.ndarray, sample_rate: int, encoding: str) -> dict[str, Any]:
    """Write an 8/16/24/32-bit PCM or 32-bit float WAV file."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2:
        raise WavError("INTERNAL_ERROR", "audio samples must be a 2-D array")
    if data.shape[0] == 0:
        raise WavError("INTERNAL_ERROR", "cannot write an empty WAV file")
    if data.shape[1] not in (1, 2):
        raise WavError("INTERNAL_ERROR", "only mono and stereo output are supported")
    if not np.all(np.isfinite(data)):
        raise WavError("INTERNAL_ERROR", "audio contains non-finite samples")
    payload, bits, format_tag = _encode_samples(data, encoding)
    channels = int(data.shape[1])
    block_align = channels * ((bits + 7) // 8)
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    riff = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(riff)
    return {
        "bit_depth": bits,
        "encoding": encoding,
        "samples": int(data.shape[0]),
        "sample_rate": int(sample_rate),
        "channels": channels,
        "bytes": len(riff),
    }
