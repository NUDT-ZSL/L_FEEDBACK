from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .jsonutil import atomic_write_bytes


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum)
    )


def encode_png_rgb8(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must be an HxWx3 uint8 array")
    height, width, _ = image.shape
    compressor = zlib.compressobj(
        level=9,
        method=zlib.DEFLATED,
        wbits=15,
        memLevel=9,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )
    compressed = bytearray()
    row_stride = width * 3
    for row in range(height):
        compressed.extend(compressor.compress(b"\x00"))
        compressed.extend(
            compressor.compress(image[row].tobytes(order="C")[:row_stride])
        )
    compressed.extend(compressor.flush())

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", bytes(compressed)),
            _chunk(b"IEND", b""),
        ]
    )


def write_png_rgb8(path: Path, image: np.ndarray) -> bytes:
    data = encode_png_rgb8(image)
    atomic_write_bytes(path, data)
    return data
