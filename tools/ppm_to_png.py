#!/usr/bin/env python3
"""Convert the GPU test's binary PPM artifacts to PNG using only stdlib."""

import glob
import os
import struct
import sys
import zlib


def chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def convert(source: str) -> None:
    with open(source, "rb") as stream:
        magic = stream.readline().strip()
        width, height = map(int, stream.readline().split())
        maximum = stream.readline().strip()
        pixels = stream.read()
    if magic != b"P6" or maximum != b"255":
        raise ValueError(f"Unsupported PPM: {source}")
    row_stride = width * 3
    rows = b"".join(
        b"\x00" + pixels[y * row_stride:(y + 1) * row_stride]
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )
    destination = os.path.splitext(source)[0] + ".png"
    with open(destination, "wb") as stream:
        stream.write(png)


def main() -> int:
    pattern = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "artifacts", "*.ppm")
    files = sorted(glob.glob(pattern))
    for path in files:
        convert(path)
    print(f"Converted {len(files)} PPM file(s) to PNG")
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
