"""Deterministic PNG output with an explicit sRGB marker.

Pillow does not consistently emit the ``sRGB`` chunk for plain RGB images,
so the PNG IDAT payload is produced once and then re-wrapped with an
explicit ``sRGB`` chunk (rendering intent 0 = perceptual) plus ``gAMA``.
Byte-identical input produces byte-identical output.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _parse_chunks(raw: bytes) -> list[tuple[bytes, bytes]]:
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG stream")
    chunks: list[tuple[bytes, bytes]] = []
    pos = len(PNG_SIGNATURE)
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", raw[pos + 8 + length:pos + 12 + length])[0]
        if (zlib.crc32(tag + data) & 0xFFFFFFFF) != crc:
            raise ValueError("corrupt PNG chunk CRC")
        chunks.append((tag, data))
        pos += 12 + length
    return chunks


def write_srgb_png(path: str, rgb: np.ndarray) -> None:
    """Write an HxWx3 uint8 array as an sRGB-marked PNG deterministically."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 array")
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6, optimize=False)
    chunks = _parse_chunks(buffer.getvalue())

    output = bytearray(PNG_SIGNATURE)
    idat = bytearray()
    skip = {b"sRGB", b"gAMA", b"iCCP", b"cHRM"}
    for tag, data in chunks:
        if tag == b"IDAT":
            idat += data
        elif tag in skip:
            continue
        elif tag == b"IHDR":
            output += _chunk(tag, data)
            output += _chunk(b"sRGB", b"\x00")
            # gAMA: gamma exponent 1/2.2 -> integer 45455.
            output += _chunk(b"gAMA", struct.pack(">I", 45455))
        elif tag == b"IEND":
            continue
        else:
            output += _chunk(tag, data)
    output += _chunk(b"IDAT", bytes(idat))
    output += _chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(bytes(output))


def read_png_chunks(path: str) -> list[bytes]:
    """Return the list of chunk tags present in a PNG file."""
    with open(path, "rb") as handle:
        return [tag for tag, _ in _parse_chunks(handle.read())]
