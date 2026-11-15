"""Small deterministic RIFF/WAVE reader and writer.

The reader intentionally supports only uncompressed integer PCM at the rates
and widths required by the task. This keeps format diagnostics explicit and
avoids FFmpeg command-line differences during deterministic renders.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import BusinessError


_PCM_GUID_TAIL = bytes.fromhex("000000001000800000aa00389b71")


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    frames: int


def _unsupported(message: str) -> BusinessError:
    return BusinessError("E_FORMAT_UNSUPPORTED", message)


def read_wav(path: Path) -> tuple[np.ndarray, WavInfo]:
    """Read a PCM WAV file into float64 samples shaped (frames, channels)."""

    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise BusinessError("E_ASSET_MISSING", f"asset not found: {path}") from exc
    except OSError as exc:
        raise BusinessError("E_ASSET_MISSING", f"cannot read asset: {path}: {exc}") from exc

    if len(payload) < 12 or payload[0:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise _unsupported(f"not a RIFF/WAVE file: {path}")

    declared_riff_size = struct.unpack_from("<I", payload, 4)[0]
    if declared_riff_size + 8 > len(payload):
        raise _unsupported(f"truncated RIFF container: {path}")

    fmt: tuple[int, int, int, int, int, int, int] | None = None
    data: bytes | None = None
    offset = 12
    limit = min(len(payload), declared_riff_size + 8)
    while offset + 8 <= limit:
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        body_start = offset + 8
        body_end = body_start + chunk_size
        if body_end > limit:
            raise _unsupported(f"truncated WAV chunk: {path}")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise _unsupported(f"invalid WAV fmt chunk: {path}")
            audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from(
                "<HHIIHH", payload, body_start
            )
            base_format = audio_format
            if audio_format == 0xFFFE:
                if chunk_size < 40:
                    raise _unsupported(f"truncated WAVE_FORMAT_EXTENSIBLE: {path}")
                valid_bits = struct.unpack_from("<H", payload, body_start + 18)[0]
                subtype = payload[body_start + 24 : body_start + 40]
                if subtype[:4] != b"\x01\x00\x00\x00" or subtype[4:] != _PCM_GUID_TAIL:
                    raise _unsupported(f"non-PCM extensible WAV is unsupported: {path}")
                base_format = 1
                if valid_bits not in (0, bits):
                    raise _unsupported(
                        f"valid bits {valid_bits} differ from container bits {bits}: {path}"
                    )
            fmt = (base_format, channels, sample_rate, block_align, bits, 0, 0)
        elif chunk_id == b"data":
            if data is None:
                data = payload[body_start:body_end]
        offset = body_end + (chunk_size & 1)

    if fmt is None:
        raise _unsupported(f"WAV has no fmt chunk: {path}")
    if data is None:
        raise _unsupported(f"WAV has no data chunk: {path}")

    audio_format, channels, sample_rate, block_align, bits, _, _ = fmt
    if audio_format != 1:
        raise _unsupported(f"only integer PCM is supported: {path}")
    if sample_rate != 48000:
        raise _unsupported(f"expected 48000 Hz, got {sample_rate} Hz: {path}")
    if bits != 24:
        raise _unsupported(f"expected 24-bit PCM, got {bits}-bit: {path}")
    if channels not in (1, 2):
        raise _unsupported(f"expected mono or stereo, got {channels} channels: {path}")
    if block_align != channels * 3:
        raise _unsupported(f"invalid PCM block alignment: {path}")
    if len(data) % block_align:
        raise _unsupported(f"partial PCM frame in data chunk: {path}")

    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    values = (
        raw[:, 0].astype(np.int32)
        | (raw[:, 1].astype(np.int32) << 8)
        | (raw[:, 2].astype(np.int32) << 16)
    )
    values = np.where(values & 0x800000, values - 0x1000000, values)
    samples = values.astype(np.float64).reshape(-1, channels) / 8388608.0
    frames = samples.shape[0]
    return samples, WavInfo(sample_rate, channels, bits, frames)


def _pcm_payload(samples: np.ndarray, bit_depth: int) -> bytes:
    if samples.ndim != 2:
        raise ValueError("samples must be shaped (frames, channels)")
    if bit_depth == 16:
        scale = 32768.0
        lower = -32768
        upper = 32767
        dtype = "<i2"
    elif bit_depth == 24:
        scale = 8388608.0
        lower = -8388608
        upper = 8388607
        dtype = None
    else:
        raise ValueError(f"unsupported export bit depth: {bit_depth}")

    quantized = np.rint(samples * scale).astype(np.int64)
    quantized = np.clip(quantized, lower, upper)
    if bit_depth == 16:
        return quantized.astype(dtype).tobytes()

    unsigned = (quantized & 0xFFFFFF).astype(np.uint32).reshape(-1)
    out = np.empty((unsigned.size, 3), dtype=np.uint8)
    out[:, 0] = unsigned & 0xFF
    out[:, 1] = (unsigned >> 8) & 0xFF
    out[:, 2] = (unsigned >> 16) & 0xFF
    return out.tobytes()


def wav_bytes(samples: np.ndarray, sample_rate: int, bit_depth: int) -> bytes:
    """Build a canonical little-endian RIFF/WAVE PCM byte stream."""

    if samples.ndim != 2:
        raise ValueError("samples must be shaped (frames, channels)")
    channels = samples.shape[1]
    payload = _pcm_payload(samples, bit_depth)
    block_align = channels * (bit_depth // 8)
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bit_depth,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(payload))
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", len(fmt)),
            fmt,
            b"data",
            struct.pack("<I", len(payload)),
            payload,
        )
    )


def write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    bit_depth: int,
) -> None:
    path.write_bytes(wav_bytes(samples, sample_rate, bit_depth))

