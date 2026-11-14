"""Canonical PCM WAV decoding and deterministic 16/24-bit encoding."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class WavFormatError(Exception):
    pass


@dataclass(frozen=True)
class WavData:
    sample_rate: int
    channels: int
    bit_depth: int
    sample_count: int
    samples: np.ndarray
    file_sha256: str

    @property
    def mono_samples(self) -> np.ndarray:
        if self.channels != 1:
            raise WavFormatError("audio is not mono")
        return self.samples[:, 0]


def _shape_channels(values: np.ndarray, channels: int) -> np.ndarray:
    if channels > 1:
        values = values.reshape(-1, channels)
    else:
        values = values.reshape(-1, 1)
    return np.ascontiguousarray(values, dtype=np.float64)


def _decode_integer_pcm(
    raw: bytes, sample_width: int, channels: int
) -> np.ndarray:
    if sample_width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(unsigned >= 0x800000, unsigned - 0x1000000, unsigned)
        values = signed.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise WavFormatError(f"unsupported PCM sample width: {sample_width} bytes")
    return _shape_channels(values, channels)


def _decode_float_pcm(
    raw: bytes, sample_width: int, channels: int
) -> np.ndarray:
    if sample_width == 4:
        values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    elif sample_width == 8:
        values = np.frombuffer(raw, dtype="<f8").astype(np.float64)
    else:
        raise WavFormatError(
            f"unsupported IEEE float sample width: {sample_width} bytes"
        )
    if not np.all(np.isfinite(values)):
        raise WavFormatError("input WAV contains non-finite float samples")
    return _shape_channels(values, channels)


def _iter_riff_chunks(raw_file: bytes):
    if len(raw_file) < 12 or raw_file[:4] != b"RIFF" or raw_file[8:12] != b"WAVE":
        raise WavFormatError("file is not a RIFF/WAVE file")
    position = 12
    while position + 8 <= len(raw_file):
        chunk_id = raw_file[position : position + 4]
        chunk_size = struct.unpack_from("<I", raw_file, position + 4)[0]
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(raw_file):
            raise WavFormatError(f"truncated RIFF chunk {chunk_id!r}")
        yield chunk_id, raw_file[payload_start:payload_end]
        position = payload_end + (chunk_size & 1)


def read_wav(path: str | Path) -> WavData:
    path = Path(path)
    raw_file = path.read_bytes()
    digest = hashlib.sha256(raw_file).hexdigest()

    format_payload: bytes | None = None
    data_payloads: list[bytes] = []
    for chunk_id, payload in _iter_riff_chunks(raw_file):
        if chunk_id == b"fmt ":
            if format_payload is not None:
                raise WavFormatError("WAV contains more than one fmt chunk")
            format_payload = payload
        elif chunk_id == b"data":
            data_payloads.append(payload)
    if format_payload is None:
        raise WavFormatError("WAV is missing its fmt chunk")
    if not data_payloads:
        raise WavFormatError("WAV is missing its data chunk")
    if len(format_payload) < 16:
        raise WavFormatError("WAV fmt chunk is shorter than 16 bytes")

    (
        format_tag,
        channels,
        sample_rate,
        _byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", format_payload, 0)

    if format_tag == 0xFFFE:
        if len(format_payload) < 40:
            raise WavFormatError("extensible WAV fmt chunk is too short")
        extension_size = struct.unpack_from("<H", format_payload, 16)[0]
        if extension_size < 22:
            raise WavFormatError("extensible WAV fmt extension is too short")
        valid_bits = struct.unpack_from("<H", format_payload, 18)[0]
        subformat_tag = struct.unpack_from("<H", format_payload, 24)[0]
        format_tag = subformat_tag
        if valid_bits not in (0, bits_per_sample):
            raise WavFormatError(
                "WAV valid bits do not match its container width"
            )

    if format_tag not in (1, 3):
        raise WavFormatError(f"unsupported WAV format tag: {format_tag}")
    if channels < 1:
        raise WavFormatError("WAV has no channels")
    if sample_rate < 1:
        raise WavFormatError("WAV sample rate must be positive")
    if block_align < channels or block_align % channels != 0:
        raise WavFormatError("WAV block alignment is invalid")
    sample_width = block_align // channels
    if format_tag == 1 and bits_per_sample != sample_width * 8:
        raise WavFormatError(
            "integer WAV bits per sample do not match its container width"
        )
    if format_tag == 3 and bits_per_sample != sample_width * 8:
        raise WavFormatError(
            "float WAV bits per sample do not match its container width"
        )

    raw_pcm = b"".join(data_payloads)
    if len(raw_pcm) % block_align != 0:
        raise WavFormatError(
            "WAV data length is not a whole number of sample frames"
        )
    sample_count = len(raw_pcm) // block_align
    if sample_count < 1:
        raise WavFormatError("WAV contains no audio samples")

    if format_tag == 1:
        samples = _decode_integer_pcm(raw_pcm, sample_width, channels)
    else:
        samples = _decode_float_pcm(raw_pcm, sample_width, channels)
    return WavData(
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bits_per_sample,
        sample_count=sample_count,
        samples=samples,
        file_sha256=digest,
    )


def quantize_int(samples: np.ndarray, bits: int) -> np.ndarray:
    if bits not in (16, 24):
        raise ValueError("quantization must be 16 or 24 bits")
    scale = 1 << (bits - 1)
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    scaled = clipped * float(scale)
    rounded = np.empty_like(scaled)
    nonnegative = scaled >= 0.0
    rounded[nonnegative] = np.floor(scaled[nonnegative] + 0.5)
    rounded[~nonnegative] = np.ceil(scaled[~nonnegative] - 0.5)
    return np.clip(rounded, -scale, scale - 1).astype(np.int64, copy=False)


def _encode_16(values: np.ndarray) -> bytes:
    return values.astype("<i2", copy=False).tobytes(order="C")


def _encode_24(values: np.ndarray) -> bytes:
    unsigned = values.astype("<i4", copy=False).view("<u4") & np.uint32(0xFFFFFF)
    packed = unsigned.view(np.uint8).reshape(-1, 4)[:, :3]
    return np.ascontiguousarray(packed).tobytes(order="C")


def encode_wav(
    samples: np.ndarray,
    *,
    quantization_bits: int,
    sample_rate: int = 48000,
) -> tuple[bytes, bytes]:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("encoder expects a one-dimensional mono signal")
    values = quantize_int(audio, quantization_bits)
    if quantization_bits == 16:
        pcm = _encode_16(values)
    else:
        pcm = _encode_24(values)

    sample_width = quantization_bits // 8
    channels = 1
    block_align = channels * sample_width
    byte_rate = sample_rate * block_align
    pad = b"\x00" if len(pcm) % 2 else b""
    data_size = len(pcm)
    riff_size = 4 + (8 + 16) + (8 + data_size + len(pad))
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        quantization_bits,
        b"data",
        data_size,
    )
    return header + pcm + pad, pcm

