"""Small deterministic RIFF/WAVE reader, encoder, and verifier.

Only uncompressed PCM 16/24/32-bit and IEEE float32 are supported.  The
reader also understands WAVE_FORMAT_EXTENSIBLE containers because source
assets exported by common DAWs often use them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .errors import SkillSFXError


_PCM = 1
_FLOAT = 3
_EXTENSIBLE = 0xFFFE


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    valid_bits: int
    sample_format: str
    audio_format: int
    block_align: int
    byte_rate: int
    frame_count: int
    data_size: int
    file_size: int
    riff_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_header(raw: bytes, path: Path) -> tuple[WavInfo, int, int]:
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise SkillSFXError(
            "SFX-E2002",
            "file is not a RIFF/WAVE stream",
            {"source_path": str(path)},
        )
    riff_size = struct.unpack_from("<I", raw, 4)[0]
    if riff_size + 8 > len(raw):
        raise SkillSFXError(
            "SFX-E2002",
            "RIFF container length exceeds file length",
            {
                "source_path": str(path),
                "riff_size": riff_size,
                "file_size": len(raw),
            },
        )

    fmt_payload: bytes | None = None
    data_offset = -1
    data_size = -1
    cursor = 12
    limit = min(len(raw), riff_size + 8)
    while cursor + 8 <= limit:
        chunk_id = raw[cursor : cursor + 4]
        chunk_size = struct.unpack_from("<I", raw, cursor + 4)[0]
        payload_at = cursor + 8
        payload_end = payload_at + chunk_size
        if payload_end > len(raw):
            raise SkillSFXError(
                "SFX-E2002",
                "WAV chunk extends past end of file",
                {
                    "source_path": str(path),
                    "chunk_id": chunk_id.decode("latin-1", errors="replace"),
                    "chunk_size": chunk_size,
                },
            )
        if chunk_id == b"fmt ":
            fmt_payload = raw[payload_at:payload_end]
        elif chunk_id == b"data":
            data_offset = payload_at
            data_size = chunk_size
        cursor = payload_end + (chunk_size & 1)

    if fmt_payload is None or len(fmt_payload) < 16:
        raise SkillSFXError(
            "SFX-E2002",
            "WAV is missing a complete fmt chunk",
            {"source_path": str(path)},
        )
    if data_offset < 0:
        raise SkillSFXError(
            "SFX-E2002",
            "WAV is missing a data chunk",
            {"source_path": str(path)},
        )

    (
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt_payload, 0)
    valid_bits = bits_per_sample
    if audio_format == _EXTENSIBLE:
        if len(fmt_payload) < 40:
            raise SkillSFXError(
                "SFX-E2002",
                "extensible WAV fmt chunk is too short",
                {"source_path": str(path)},
            )
        valid_bits = struct.unpack_from("<H", fmt_payload, 18)[0]
        subformat = struct.unpack_from("<H", fmt_payload, 24)[0]
        audio_format = subformat

    if audio_format == _PCM:
        if bits_per_sample not in (16, 24, 32):
            raise SkillSFXError(
                "SFX-E2002",
                "unsupported PCM bit depth",
                {
                    "source_path": str(path),
                    "bits_per_sample": bits_per_sample,
                },
            )
        sample_format = "pcm"
        expected_align = channels * (bits_per_sample // 8)
    elif audio_format == _FLOAT:
        if bits_per_sample != 32:
            raise SkillSFXError(
                "SFX-E2002",
                "only 32-bit IEEE float WAV is supported",
                {
                    "source_path": str(path),
                    "bits_per_sample": bits_per_sample,
                },
            )
        sample_format = "float"
        expected_align = channels * 4
    else:
        raise SkillSFXError(
            "SFX-E2002",
            "unsupported WAV audio format",
            {"source_path": str(path), "audio_format": audio_format},
        )

    if channels < 1 or channels > 2:
        raise SkillSFXError(
            "SFX-E2002",
            "source WAV must be mono or stereo",
            {"source_path": str(path), "channels": channels},
        )
    if sample_rate <= 0 or block_align != expected_align:
        raise SkillSFXError(
            "SFX-E2002",
            "invalid WAV sample rate or block alignment",
            {
                "source_path": str(path),
                "sample_rate": sample_rate,
                "block_align": block_align,
                "expected_block_align": expected_align,
            },
        )
    if byte_rate != sample_rate * block_align:
        raise SkillSFXError(
            "SFX-E2002",
            "WAV byte rate is inconsistent with sample rate and block alignment",
            {
                "source_path": str(path),
                "byte_rate": byte_rate,
                "expected_byte_rate": sample_rate * block_align,
            },
        )
    if data_size % block_align != 0:
        raise SkillSFXError(
            "SFX-E2002",
            "WAV data chunk is not frame aligned",
            {
                "source_path": str(path),
                "data_size": data_size,
                "block_align": block_align,
            },
        )

    info = WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
        valid_bits=valid_bits if valid_bits else bits_per_sample,
        sample_format=sample_format,
        audio_format=audio_format,
        block_align=block_align,
        byte_rate=byte_rate,
        frame_count=data_size // block_align,
        data_size=data_size,
        file_size=len(raw),
        riff_size=riff_size,
    )
    return info, data_offset, data_size


def _decode_data(
    data: bytes,
    info: WavInfo,
    source_path: str | Path,
) -> np.ndarray:
    if info.sample_format == "float":
        samples = np.frombuffer(data, dtype="<f4").astype(np.float64)
    elif info.bits_per_sample == 16:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    elif info.bits_per_sample == 24:
        packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values >= 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float64) / 8388608.0
    else:
        samples = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0

    if not np.all(np.isfinite(samples)):
        raise SkillSFXError(
            "SFX-E2002",
            "source WAV contains NaN or infinite samples",
            {"source_path": str(source_path)},
        )
    return samples.reshape(-1, info.channels)


def read_wav(path: str | Path) -> tuple[np.ndarray, WavInfo]:
    resolved = Path(path)
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise SkillSFXError(
            "SFX-E2001",
            "source WAV could not be read",
            {"source_path": str(resolved), "reason": str(exc)},
        ) from exc
    info, offset, size = _parse_header(raw, resolved)
    audio = _decode_data(raw[offset : offset + size], info, resolved)
    return audio, info


def _quantize(audio: np.ndarray, bit_depth: int) -> bytes:
    flat = audio.reshape(-1)
    if bit_depth == 16:
        values = np.rint(flat * 32768.0)
        values = np.clip(values, -32768, 32767).astype("<i2")
        return values.tobytes()
    if bit_depth == 24:
        values = np.rint(flat * 8388608.0)
        values = np.clip(values, -8388608, 8388607).astype("<i4")
        packed = values.view(np.uint8).reshape(-1, 4)
        return packed[:, :3].tobytes()
    if bit_depth == 32:
        values = np.rint(flat * 2147483648.0)
        values = np.clip(values, -2147483648, 2147483647).astype("<i4")
        return values.tobytes()
    raise ValueError(f"unsupported PCM bit depth: {bit_depth}")


def encode_wav(audio: np.ndarray, sample_rate: int, bit_depth: int | str) -> bytes:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise SkillSFXError(
            "SFX-E4001",
            "encoder expected a 1-D or 2-D sample array",
            {"shape": list(samples.shape)},
        )
    if not np.all(np.isfinite(samples)):
        raise SkillSFXError(
            "SFX-E4001",
            "refusing to encode non-finite audio",
            {"shape": list(samples.shape)},
        )

    channels = int(samples.shape[1])
    if channels not in (1, 2):
        raise SkillSFXError(
            "SFX-E4001",
            "output WAV must be mono or stereo",
            {"channels": channels},
        )
    if sample_rate not in (44100, 48000):
        raise SkillSFXError(
            "SFX-E4001",
            "output sample rate must be 44100 or 48000",
            {"sample_rate": sample_rate},
        )

    if bit_depth == "float32":
        if np.max(np.abs(samples), initial=0.0) >= 1.0:
            raise SkillSFXError(
                "SFX-E4002",
                "float32 output contains a full-scale sample",
                {"peak": float(np.max(np.abs(samples), initial=0.0))},
            )
        data = samples.astype("<f4").tobytes()
        audio_format = _FLOAT
        bits = 32
    elif bit_depth in (16, 24, 32):
        data = _quantize(samples, int(bit_depth))
        audio_format = _PCM
        bits = int(bit_depth)
    else:
        raise SkillSFXError(
            "SFX-E4001",
            "unsupported output bit depth",
            {"bit_depth": bit_depth},
        )

    block_align = channels * (bits // 8)
    byte_rate = sample_rate * block_align
    padding = b"\x00" if len(data) & 1 else b""
    fmt = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
    )
    chunks = (
        b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
        + padding
    )
    riff_size = 4 + len(chunks)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + chunks


def verify_wav(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    bit_depth: int | str | None = None,
    frame_count: int | None = None,
) -> tuple[np.ndarray, WavInfo]:
    resolved = Path(path)
    raw = resolved.read_bytes()
    info, data_offset, data_size = _parse_header(raw, resolved)
    if info.riff_size + 8 != info.file_size:
        raise SkillSFXError(
            "SFX-E4001",
            "post-write WAV RIFF length does not match file length",
            {
                "output_path": str(resolved),
                "riff_size": info.riff_size,
                "file_size": info.file_size,
            },
        )
    expected_bits = 32 if bit_depth == "float32" else bit_depth
    expected_format = "float" if bit_depth == "float32" else "pcm"
    checks = {
        "sample_rate": (info.sample_rate, sample_rate),
        "channels": (info.channels, channels),
        "bits_per_sample": (info.bits_per_sample, expected_bits),
        "sample_format": (info.sample_format, expected_format),
        "frame_count": (info.frame_count, frame_count),
    }
    for field, (actual, expected) in checks.items():
        if expected is not None and actual != expected:
            raise SkillSFXError(
                "SFX-E4001",
                "post-write WAV header verification failed",
                {
                    "output_path": str(resolved),
                    "field": field,
                    "actual": actual,
                    "expected": expected,
                },
            )
    expected_data_size = info.frame_count * info.block_align
    if data_size != expected_data_size or info.data_size != data_size:
        raise SkillSFXError(
            "SFX-E4001",
            "post-write WAV data length and frame count disagree",
            {
                "output_path": str(resolved),
                "data_size": data_size,
                "block_align": info.block_align,
                "frame_count": info.frame_count,
                "expected_data_size": expected_data_size,
            },
        )
    if data_offset + data_size > len(raw):
        raise SkillSFXError(
            "SFX-E4001",
            "post-write WAV data chunk extends beyond the file",
            {"output_path": str(resolved)},
        )
    audio = _decode_data(raw[data_offset : data_offset + data_size], info, resolved)
    return audio, info


def encoded_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def db_from_amplitude(value: float) -> float:
    if value <= 0.0:
        return -math.inf
    return 20.0 * math.log10(value)
