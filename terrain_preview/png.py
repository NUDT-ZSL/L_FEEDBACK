"""Streaming RGB/gray PNG writer used to avoid full-image staging buffers."""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


class StreamingPNG:
    """Write an 8-bit PNG one scanline at a time.

    The temporary file is atomically renamed on successful close.
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        *,
        grayscale: bool = False,
        compression_level: int = 6,
    ) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.grayscale = grayscale
        self.rows_written = 0
        self._temporary = self.path.with_name(self.path.name + f".tmp-{os.getpid()}")
        self._handle: BinaryIO | None = None
        self._compressor = zlib.compressobj(level=compression_level)
        self._pending = bytearray()

    def __enter__(self) -> "StreamingPNG":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._temporary.open("wb")
        self._handle.write(b"\x89PNG\r\n\x1a\n")
        color_type = 0 if self.grayscale else 2
        header = struct.pack(
            ">IIBBBBB", self.width, self.height, 8, color_type, 0, 0, 0
        )
        self._handle.write(_chunk(b"IHDR", header))
        return self

    def _emit(self, data: bytes) -> None:
        if not data or self._handle is None:
            return
        self._pending.extend(data)
        if len(self._pending) >= 65536:
            payload = bytes(self._pending[:65536])
            del self._pending[:65536]
            self._handle.write(_chunk(b"IDAT", payload))

    def write_row(self, row: np.ndarray) -> None:
        if self.rows_written >= self.height:
            raise ValueError("too many PNG rows")
        array = np.asarray(row)
        expected = self.width if self.grayscale else self.width * 3
        flattened = np.ascontiguousarray(array.reshape(-1))
        if flattened.size != expected:
            raise ValueError(
                f"PNG row has {flattened.size} values, expected {expected}"
            )
        if flattened.dtype != np.uint8:
            flattened = np.clip(flattened, 0, 255).astype(np.uint8)
        self._emit(self._compressor.compress(b"\x00" + flattened.tobytes()))
        self.rows_written += 1

    def write_rows(self, rows: Iterable[np.ndarray]) -> None:
        for row in rows:
            self.write_row(row)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            if self.rows_written != self.height:
                raise ValueError(
                    f"PNG has {self.rows_written} rows, expected {self.height}"
                )
            self._emit(self._compressor.flush())
            if self._pending:
                self._handle.write(_chunk(b"IDAT", bytes(self._pending)))
                self._pending.clear()
            self._handle.write(_chunk(b"IEND", b""))
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
            os.replace(self._temporary, self.path)
        except Exception:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None
            try:
                self._temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None
            try:
                self._temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return
        self.close()

