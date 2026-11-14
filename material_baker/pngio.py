"""Deterministic PNG encoding without ancillary metadata.

OpenCV is used for decoding only. Encoding is handled here so the byte stream
is independent of image-library version metadata, locale and wall-clock time.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .errors import MaterialError
from .util import atomic_write_bytes

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS_TO_COLOR_TYPE = {1: 0, 3: 2, 4: 6}


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _to_integer(data: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth not in {8, 16}:
        raise MaterialError("E009", f"unsupported PNG bit depth: {bit_depth}")
    if np.issubdtype(data.dtype, np.floating):
        maximum = (1 << bit_depth) - 1
        data = np.rint(np.clip(data, 0.0, 1.0) * maximum).astype(
            np.uint8 if bit_depth == 8 else np.uint16
        )
    elif data.dtype == np.uint8 and bit_depth == 8:
        pass
    elif data.dtype == np.uint16 and bit_depth == 16:
        pass
    else:
        raise MaterialError(
            "E009",
            f"PNG array dtype {data.dtype} does not match {bit_depth}-bit output",
        )
    return data


def _row_bytes(row: np.ndarray, bit_depth: int) -> bytes:
    if row.ndim == 1:
        row = row[:, None]
    if bit_depth == 8:
        raw = np.ascontiguousarray(row.astype(np.uint8))
    else:
        raw = np.ascontiguousarray(row.astype(">u2"))
    return raw.tobytes()


def encode_png(data: np.ndarray, bit_depth: int = 8) -> bytes:
    integer = _to_integer(np.asarray(data), bit_depth)
    if integer.ndim == 2:
        integer = integer[:, :, None]
    if integer.ndim != 3 or integer.shape[2] not in _CHANNELS_TO_COLOR_TYPE:
        raise MaterialError("E009", "PNG needs 1, 3 or 4 channels")
    height, width, channels = integer.shape

    raw_rows = bytearray()
    for row in integer:
        raw_rows.append(0)
        raw_rows.extend(_row_bytes(row, bit_depth))

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        _CHANNELS_TO_COLOR_TYPE[channels],
        0,
        0,
        0,
    )
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows), 9))
        + _chunk(b"IEND", b"")
    )


def write_png(path: Path, data: np.ndarray, bit_depth: int = 8) -> str:
    payload = encode_png(data, bit_depth)
    atomic_write_bytes(path, payload)
    import hashlib

    return hashlib.sha256(payload).hexdigest()


class StreamingPNGWriter:
    """Stream a C-order image into one deterministic PNG IDAT."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        channels: int,
        bit_depth: int,
    ) -> None:
        if channels not in _CHANNELS_TO_COLOR_TYPE:
            raise MaterialError("E009", "PNG needs 1, 3 or 4 channels")
        if bit_depth not in {8, 16}:
            raise MaterialError("E009", f"unsupported PNG bit depth: {bit_depth}")
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.bit_depth = int(bit_depth)
        self._rows_written = 0
        self._compress = zlib.compressobj(9)
        self._temporary = path.with_name(f".{path.name}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: BinaryIO = self._temporary.open("wb")
        ihdr = struct.pack(
            ">IIBBBBB",
            self.width,
            self.height,
            self.bit_depth,
            _CHANNELS_TO_COLOR_TYPE[self.channels],
            0,
            0,
            0,
        )
        self._handle.write(PNG_SIGNATURE)
        self._handle.write(_chunk(b"IHDR", ihdr))

    def _compress_rows(self, rows: np.ndarray) -> None:
        if rows.ndim == 2:
            rows = rows[:, :, None]
        if rows.ndim != 3 or rows.shape[2] != self.channels:
            raise MaterialError(
                "E009",
                f"PNG row block must have {self.channels} channels",
            )
        integer = _to_integer(rows, self.bit_depth)
        for row in integer:
            payload = bytes([0]) + _row_bytes(row, self.bit_depth)
            self._handle.write(self._compress.compress(payload))
        self._rows_written += integer.shape[0]

    def append_rows(self, rows: np.ndarray) -> None:
        self._compress_rows(np.asarray(rows))

    def finalize(self) -> str:
        if self._handle.closed:
            raise MaterialError("E009", "PNG stream is already closed")
        if self._rows_written != self.height:
            self._handle.close()
            raise MaterialError(
                "E009",
                f"PNG stream wrote {self._rows_written} of {self.height} rows",
            )
        compressed = self._compress.flush()
        self._handle.write(_chunk(b"IDAT", compressed))
        self._handle.write(_chunk(b"IEND", b""))
        self._handle.flush()
        self._handle.close()
        self._temporary.replace(self.path)
        from .util import sha256_file

        return sha256_file(self.path)


def write_tga(path: Path, rgb: np.ndarray, bit_depth: int = 8) -> str:
    """Write an uncompressed, top-left-origin TGA used by sample fixtures."""

    data = _to_integer(np.asarray(rgb), bit_depth)
    if data.ndim != 3 or data.shape[2] not in {3, 4}:
        raise MaterialError("E009", "TGA sample writer needs RGB or RGBA")
    if bit_depth != 8:
        raise MaterialError("E009", "this fixture TGA writer supports 8-bit only")
    height, width, channels = data.shape
    descriptor = 0x20 | (8 if channels == 4 else 0)
    header = bytearray(18)
    header[2] = 2
    header[12:14] = struct.pack("<H", width)
    header[14:16] = struct.pack("<H", height)
    header[16] = 24 if channels == 3 else 32
    header[17] = descriptor
    bgr = data[:, :, :3][:, :, ::-1]
    if channels == 4:
        body = np.concatenate([bgr, data[:, :, 3:4]], axis=2).tobytes()
    else:
        body = bgr.tobytes()
    payload = bytes(header) + body
    atomic_write_bytes(path, payload)
    import hashlib

    return hashlib.sha256(payload).hexdigest()
