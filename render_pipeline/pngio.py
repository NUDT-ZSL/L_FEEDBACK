from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(tag)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def write_rgb8_png(path: Path, rgb8: np.ndarray) -> None:
    """Write a deterministic, metadata-free 8-bit sRGB PNG.

    A private encoder is used instead of a library encoder so chunk order,
    compression level and ancillary metadata do not depend on Pillow's version.
    """

    image = np.asarray(rgb8)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 uint8 array")
    height, width, _ = image.shape
    scanlines = bytearray()
    for row in image:
        scanlines.append(0)
        scanlines.extend(row.tobytes(order="C"))
    compressed = zlib.compress(bytes(scanlines), level=9)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"sRGB", b"\x00")
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
