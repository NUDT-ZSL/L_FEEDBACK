"""Versioned binary PCM pack construction and verification."""

from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BakeConfig
from .constants import (
    PACK_FORMAT_VERSION,
    PACK_HEADER_SIZE,
    PACK_HEADER_STRUCT,
    PACK_MAGIC,
    SAMPLE_FORMAT,
)
from .util import sha256_bytes, sha256_file

FORMAT_CODE_PCM_S16LE = 1
_HEADER = struct.Struct(PACK_HEADER_STRUCT)


@dataclass(frozen=True)
class ChunkPayload:
    entry_id: str
    locale: str
    sample_count: int
    data: bytes


@dataclass(frozen=True)
class BuiltPack:
    data: bytes
    chunks: list[dict[str, Any]]
    sha256: str


def _header(config: BakeConfig, chunk_count: int, timeline_samples: int) -> bytes:
    packed = _HEADER.pack(
        PACK_MAGIC,
        PACK_FORMAT_VERSION,
        config.sample_rate,
        config.channels,
        FORMAT_CODE_PCM_S16LE,
        chunk_count,
        PACK_HEADER_SIZE,
        timeline_samples,
    )
    if len(packed) > PACK_HEADER_SIZE:
        raise ValueError("pack header struct exceeds header size")
    return packed + b"\x00" * (PACK_HEADER_SIZE - len(packed))


def build_pack(
    payloads: list[ChunkPayload],
    config: BakeConfig,
    timeline_samples: int,
) -> BuiltPack:
    buffer = bytearray(_header(config, len(payloads), timeline_samples))
    chunks: list[dict[str, Any]] = []
    alignment = config.chunk_alignment
    for index, payload in enumerate(payloads):
        expected_bytes = payload.sample_count * 2
        if len(payload.data) != expected_bytes:
            raise ValueError(
                f"{payload.entry_id}: PCM length {len(payload.data)} != {expected_bytes}"
            )
        cursor = len(buffer)
        offset = ((cursor + alignment - 1) // alignment) * alignment
        if offset > cursor:
            buffer.extend(b"\x00" * (offset - cursor))
        buffer.extend(payload.data)
        chunks.append(
            {
                "chunk_id": f"chunk-{index:06d}",
                "entry_id": payload.entry_id,
                "locale": payload.locale,
                "offset": offset,
                "length": len(payload.data),
                "sample_count": payload.sample_count,
                "sha256": sha256_bytes(payload.data),
            }
        )
    data = bytes(buffer)
    return BuiltPack(data=data, chunks=chunks, sha256=sha256_bytes(data))


def parse_header(data: bytes) -> dict[str, int | str]:
    if len(data) < PACK_HEADER_SIZE:
        raise ValueError("pack is shorter than its header")
    (
        magic,
        format_version,
        sample_rate,
        channels,
        format_code,
        chunk_count,
        header_size,
        timeline_samples,
    ) = _HEADER.unpack_from(data, 0)
    if magic != PACK_MAGIC:
        raise ValueError("pack magic does not match")
    if format_version != PACK_FORMAT_VERSION:
        raise ValueError("unsupported pack format version")
    if header_size != PACK_HEADER_SIZE:
        raise ValueError("unsupported pack header size")
    if format_code != FORMAT_CODE_PCM_S16LE:
        raise ValueError("unsupported pack sample format")
    return {
        "magic": magic.decode("ascii"),
        "format_version": format_version,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_format": SAMPLE_FORMAT,
        "chunk_count": chunk_count,
        "header_size": header_size,
        "timeline_samples": timeline_samples,
    }


def validate_chunk(
    data: bytes,
    metadata: dict[str, Any],
    *,
    expected_sample_count: int | None = None,
) -> bytes:
    offset = metadata.get("offset")
    length = metadata.get("length")
    sample_count = metadata.get("sample_count")
    if not isinstance(offset, int) or not isinstance(length, int):
        raise TypeError("chunk offset/length must be integers")
    if offset < PACK_HEADER_SIZE or length < 0 or offset + length > len(data):
        raise ValueError("chunk lies outside pack")
    if length % 2:
        raise ValueError("chunk byte length is not sample aligned")
    if not isinstance(sample_count, int) or sample_count * 2 != length:
        raise ValueError("chunk sample_count does not match its byte length")
    if expected_sample_count is not None and sample_count != expected_sample_count:
        raise ValueError("chunk sample_count does not match the expected slot duration")
    payload = data[offset : offset + length]
    if metadata.get("sha256") != sha256_bytes(payload):
        raise ValueError("chunk hash does not match")
    return payload


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def ensure_pack_file(path: Path, data: bytes, expected_sha256: str) -> bool:
    """Return True when an identical complete pack already exists."""

    if (
        path.is_file()
        and path.stat().st_size == len(data)
        and sha256_file(path) == expected_sha256
    ):
        return True
    write_bytes_atomic(path, data)
    return False


def verify_pack_file(path: Path, expected_sha256: str, expected_size: int) -> bytes:
    if not path.is_file():
        raise ValueError("pack file does not exist")
    if path.stat().st_size != expected_size:
        raise ValueError("pack file size does not match the index")
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise ValueError("pack file hash does not match the index")
    header = parse_header(data)
    if header["sample_rate"] != 48000 or header["channels"] != 1:
        raise ValueError("pack is not 48 kHz mono")
    return data
