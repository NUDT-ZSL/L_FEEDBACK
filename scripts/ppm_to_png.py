#!/usr/bin/env python3
"""Convert a binary PPM (P6) image to PNG using only the Python standard library."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def read_token(data: bytes, position: int) -> tuple[bytes, int]:
    while position < len(data) and chr(data[position]).isspace():
        position += 1
    start = position
    while position < len(data) and not chr(data[position]).isspace():
        position += 1
    return data[start:position], position


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"usage: {Path(sys.argv[0]).name} INPUT.ppm [OUTPUT.png]", file=sys.stderr)
        return 2

    source_path = Path(sys.argv[1])
    output_path = (
        Path(sys.argv[2]) if len(sys.argv) == 3 else source_path.with_suffix(".png")
    )
    data = source_path.read_bytes()
    position = 0
    magic, position = read_token(data, position)
    if magic != b"P6":
        print("only binary P6 PPM files are supported", file=sys.stderr)
        return 2
    width_token, position = read_token(data, position)
    height_token, position = read_token(data, position)
    maximum_token, position = read_token(data, position)
    width = int(width_token)
    height = int(height_token)
    maximum = int(maximum_token)
    if maximum != 255:
        print("only 8-bit PPM files are supported", file=sys.stderr)
        return 2
    while position < len(data) and chr(data[position]).isspace():
        position += 1

    pixels = data[position : position + width * height * 3]
    if len(pixels) != width * height * 3:
        print("PPM pixel payload is truncated", file=sys.stderr)
        return 2

    scanlines = b"".join(
        b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + chunk_type
            + payload
            + struct.pack(">I", crc)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 9))
        + chunk(b"IEND", b"")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

