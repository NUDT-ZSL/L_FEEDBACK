#!/usr/bin/env python3
"""Arrange the cl-produced PE read-only sections for the required layout."""

from __future__ import annotations

import struct
import sys


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def patch(path: str) -> int:
    data = bytearray(open(path, "rb").read())
    if data[:2] != b"MZ":
        raise SystemExit("not a PE file")
    pe = _u32(data, 0x3C)
    if data[pe : pe + 4] != b"PE\x00\x00":
        raise SystemExit("missing PE signature")
    coff = pe + 4
    sections = struct.unpack_from("<H", data, coff + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    table = coff + 20 + optional_size
    names = {}
    for index in range(sections):
        offset = table + index * 40
        raw = bytes(data[offset : offset + 8])
        name = raw.split(b"\x00", 1)[0].decode("latin-1")
        names[name] = offset
    if ".idata" not in names:
        raise SystemExit(".idata section was not produced by the linker")
    if ".rdata" in names:
        data[names[".rdata"] : names[".rdata"] + 8] = b".rodata\x00"
    data[names[".idata"] : names[".idata"] + 8] = b".rdata\x00\x00"
    with open(path, "wb") as handle:
        handle.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(patch(sys.argv[1]))
