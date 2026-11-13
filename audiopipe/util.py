"""Filesystem, JSON and WAV helpers."""

from __future__ import annotations

import json
import os
import struct
import uuid
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    atomic_write_bytes(path, encoded.encode("utf-8"))


def probe_wav(path: Path) -> dict[str, Any]:
    """Validate a deliberately simple canonical PCM16 WAV file.

    The pipeline writes exactly one fmt chunk and one data chunk. Strictly
    requiring data to end at EOF catches truncated concurrent writers.
    """

    size = path.stat().st_size
    if size < 44:
        raise ValueError(f"WAV is only {size} bytes")
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("short RIFF header")
        riff, riff_size, wave = struct.unpack("<4sI4s", header)
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        if riff_size + 8 != size:
            raise ValueError(f"RIFF size is {riff_size + 8}, actual file size is {size}")

        fmt: Any = None
        data_offset = None
        data_size = None
        offset = 12
        while offset + 8 <= size:
            handle.seek(offset)
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                raise ValueError("truncated chunk header")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            body_offset = offset + 8
            end = body_offset + chunk_size
            if end > size:
                raise ValueError(f"chunk {chunk_id!r} extends past EOF")
            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise ValueError("short fmt chunk")
                handle.seek(body_offset)
                values = struct.unpack("<HHIIHH", handle.read(16))
                fmt = {
                    "audio_format": values[0],
                    "channels": values[1],
                    "sample_rate": values[2],
                    "byte_rate": values[3],
                    "block_align": values[4],
                    "bits_per_sample": values[5],
                }
            elif chunk_id == b"data":
                data_offset = body_offset
                data_size = chunk_size
            offset = end + (chunk_size & 1)

        if fmt is None:
            raise ValueError("missing fmt chunk")
        if data_offset is None or data_size is None:
            raise ValueError("missing data chunk")
        if offset != size:
            raise ValueError("trailing or malformed chunk padding")
        if data_offset + data_size != size:
            raise ValueError("data chunk is not the final chunk at EOF")
        expected = {
            "audio_format": 1,
            "channels": 2,
            "sample_rate": 48000,
            "byte_rate": 192000,
            "block_align": 4,
            "bits_per_sample": 16,
        }
        if fmt != expected:
            raise ValueError(f"unexpected PCM format: {fmt}")
        if data_size % 4:
            raise ValueError("PCM data size is not a whole number of stereo frames")
        return {
            "file_size": size,
            "riff_size": riff_size + 8,
            "data_size": data_size,
            "sample_rate": fmt["sample_rate"],
            "channels": fmt["channels"],
            "bits_per_sample": fmt["bits_per_sample"],
        }


def write_pcm16_wav_atomic(pcm_path: Path, output_path: Path) -> dict[str, Any]:
    """Wrap a raw stereo 48 kHz s16le file in an exact canonical WAV.

    The temporary WAV is created in the final output directory and renamed
    only after fsync and strict structural validation.
    """

    data_size = pcm_path.stat().st_size
    if data_size % 4:
        raise ValueError("raw PCM size is not aligned to stereo s16 frames")
    if data_size > 0xFFFFFFFF - 36:
        raise OverflowError("WAV data exceeds the RIFF 32-bit limit")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".{output_path.stem}.part-{os.getpid()}-{uuid.uuid4().hex}.wav"
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        2,
        48000,
        192000,
        4,
        16,
        b"data",
        data_size,
    )

    try:
        with open(temp_path, "xb") as out_handle, open(pcm_path, "rb") as in_handle:
            out_handle.write(header)
            while True:
                block = in_handle.read(1024 * 1024)
                if not block:
                    break
                out_handle.write(block)
            out_handle.flush()
            os.fsync(out_handle.fileno())
        info = probe_wav(temp_path)
        if info["data_size"] != data_size:
            raise ValueError("assembled WAV data size does not match source PCM")
        os.replace(temp_path, output_path)
        fsync_directory(output_path.parent)
        return info
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def size_mb(value: int) -> float:
    return value / (1024.0 * 1024.0)
