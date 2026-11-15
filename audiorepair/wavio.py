"""Strict RIFF/WAVE PCM-16 reader and deterministic WAVE writer."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .errors import AUDIO_UNSUPPORTED_FORMAT, RepairError

SUPPORTED_SAMPLE_RATES = (44100, 48000)
SUPPORTED_CHANNELS = (1, 2)
SUPPORTED_BITS = 16
SUPPORTED_FORMAT_TAG = 1  # PCM


@dataclass(frozen=True)
class WavAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int
    bits_per_sample: int

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


def _unsupported(message: str) -> RepairError:
    return RepairError(AUDIO_UNSUPPORTED_FORMAT, message)


def parse_wav_bytes(payload: bytes) -> WavAudio:
    """Parse a canonical or chunked 16-bit PCM RIFF/WAVE file."""
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise _unsupported("not a RIFF/WAVE file")

    fmt = None
    data = None
    position = 12
    while position + 8 <= len(payload):
        chunk_id = payload[position : position + 4]
        chunk_size = struct.unpack_from("<I", payload, position + 4)[0]
        chunk_start = position + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(payload):
            raise _unsupported("truncated WAVE chunk")
        chunk = payload[chunk_start:chunk_end]
        if chunk_id == b"fmt " and fmt is None:
            fmt = chunk
        elif chunk_id == b"data" and data is None:
            data = chunk
        position = chunk_end + (chunk_size & 1)

    if fmt is None or data is None:
        raise _unsupported("WAVE requires both fmt and data chunks")
    if len(fmt) < 16:
        raise _unsupported("fmt chunk is shorter than PCM minimum")

    format_tag, channels, sample_rate, _byte_rate, block_align, bits = (
        struct.unpack_from("<HHIIHH", fmt, 0)
    )
    if format_tag != SUPPORTED_FORMAT_TAG:
        raise _unsupported(
            "only WAVE_FORMAT_PCM is supported (format tag {})".format(format_tag)
        )
    if bits != SUPPORTED_BITS:
        raise _unsupported(
            "only 16-bit PCM is supported (got {}-bit)".format(bits)
        )
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise _unsupported(
            "unsupported sample rate {} (allowed: 44100, 48000)".format(sample_rate)
        )
    if channels not in SUPPORTED_CHANNELS:
        raise _unsupported(
            "unsupported channel count {} (allowed: 1, 2)".format(channels)
        )

    expected_align = channels * (bits // 8)
    if block_align != expected_align:
        raise _unsupported("invalid block alignment for PCM-16")
    if len(data) % expected_align:
        raise _unsupported("data chunk is not sample-frame aligned")

    pcm = np.frombuffer(data, dtype="<i2")
    samples = pcm.reshape(-1, channels).astype(np.float64) / 32768.0
    return WavAudio(
        samples=samples,
        sample_rate=int(sample_rate),
        channels=int(channels),
        bits_per_sample=int(bits),
    )


def read_wav(path: str) -> WavAudio:
    with open(path, "rb") as handle:
        return parse_wav_bytes(handle.read())


def encode_wav(samples: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Quantise float64 audio to canonical 16-bit PCM WAVE bytes."""
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise _unsupported("unsupported output sample rate {}".format(sample_rate))
    if channels not in SUPPORTED_CHANNELS:
        raise _unsupported("unsupported output channel count {}".format(channels))
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] != channels:
        raise _unsupported("sample matrix does not match output channel count")

    # Symmetric scaling maps -32768..32767 to -1.0..(32767/32768).
    clipped = np.clip(audio, -1.0, 32767.0 / 32768.0)
    pcm = np.rint(clipped * 32768.0).astype("<i2")
    interleaved = np.ascontiguousarray(pcm).tobytes(order="C")
    block_align = channels * 2
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH",
        SUPPORTED_FORMAT_TAG,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        SUPPORTED_BITS,
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(interleaved)) + interleaved
    if len(interleaved) & 1:
        chunks += b"\x00"
    riff_size = 4 + len(chunks)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + chunks


def decode_wav_bytes(payload: bytes) -> Tuple[np.ndarray, int, int]:
    audio = parse_wav_bytes(payload)
    return audio.samples, audio.sample_rate, audio.channels

