"""Custom binary cache for intermediate height fields and normal maps.

Layout (little-endian)::

    offset  size  field
    0       8     magic b"TBCACHE1"
    8       4     format version (uint32)
    12      4     payload kind (uint32): 1=height f32, 2=normals u8x3
    16      4     width (uint32)
    20      4     height (uint32)
    24      64    params hash (ascii hex, zero padded)
    88      ...   blocks: [u32 length][u32 crc32][payload] ... until EOF

On any version mismatch or CRC failure the cache is treated as invalid:
all cache files are deleted and the data is rebuilt from scratch. Old
offsets are never reused across versions.
"""

from __future__ import annotations

import os
import zlib

import numpy as np

MAGIC = b"TBCACHE1"
VERSION = 1
HEADER_SIZE = 88
BLOCK_SIZE = 1 << 16

KIND_HEIGHT = 1
KIND_NORMALS = 2

_KIND_DTYPES = {
    KIND_HEIGHT: (np.float32, 1),
    KIND_NORMALS: (np.uint8, 3),
}


class CacheMiss(Exception):
    """Raised when a cache entry is absent, stale, or corrupt."""


def _header(kind: int, width: int, height: int, params_hash: str) -> bytes:
    import struct
    hash_bytes = params_hash.encode("ascii").ljust(64, b"\x00")
    return (MAGIC
            + struct.pack("<IIII", VERSION, kind, width, height)
            + hash_bytes)


def write_cache(path: str, kind: int, array: np.ndarray,
                params_hash: str) -> None:
    import struct
    dtype, channels = _KIND_DTYPES[kind]
    data = np.ascontiguousarray(array, dtype=dtype)
    height, width = data.shape[0], data.shape[1]
    raw = data.tobytes()
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(_header(kind, width, height, params_hash))
        for offset in range(0, len(raw), BLOCK_SIZE):
            block = raw[offset:offset + BLOCK_SIZE]
            handle.write(struct.pack("<II", len(block),
                                     zlib.crc32(block) & 0xFFFFFFFF))
            handle.write(block)
    os.replace(tmp_path, path)


def read_cache(path: str, kind: int, params_hash: str) -> np.ndarray:
    import struct
    dtype, channels = _KIND_DTYPES[kind]
    if not os.path.isfile(path):
        raise CacheMiss("cache file absent")
    with open(path, "rb") as handle:
        header = handle.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE or header[:8] != MAGIC:
            raise CacheMiss("bad magic")
        version, fkind, width, height = struct.unpack("<IIII", header[8:24])
        if version != VERSION:
            raise CacheMiss(f"version mismatch: {version} != {VERSION}")
        if fkind != kind:
            raise CacheMiss("kind mismatch")
        stored_hash = header[24:88].rstrip(b"\x00").decode("ascii")
        if stored_hash != params_hash:
            raise CacheMiss("params hash mismatch")
        chunks = []
        while True:
            block_header = handle.read(8)
            if not block_header:
                break
            if len(block_header) != 8:
                raise CacheMiss("truncated block header")
            length, crc = struct.unpack("<II", block_header)
            block = handle.read(length)
            if len(block) != length:
                raise CacheMiss("truncated block payload")
            if zlib.crc32(block) & 0xFFFFFFFF != crc:
                raise CacheMiss("block CRC mismatch")
            chunks.append(block)
    raw = b"".join(chunks)
    array = np.frombuffer(raw, dtype=dtype)
    expected = width * height * channels
    if array.size != expected:
        raise CacheMiss("payload size mismatch")
    shape = (height, width) if channels == 1 else (height, width, channels)
    return array.reshape(shape).copy()


def clear_cache_dir(cache_dir: str) -> None:
    """Remove every cache file (used on version mismatch / corruption)."""
    if not os.path.isdir(cache_dir):
        return
    for name in os.listdir(cache_dir):
        if name.endswith(".tbc") or name.endswith(".tbc.tmp"):
            os.remove(os.path.join(cache_dir, name))
