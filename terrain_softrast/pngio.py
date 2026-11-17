from __future__ import annotations

import os
import struct
import tempfile
import zlib

import numpy as np


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _encode_png(array: np.ndarray, color_type: int, bit_depth: int) -> bytes:
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        height, width = array.shape
        channels = 1
    elif array.ndim == 3:
        height, width, channels = array.shape
    else:
        raise ValueError("PNG array must be 2D or 3D")
    if color_type == 2 and channels != 3:
        raise ValueError("RGB PNG requires three channels")
    if color_type == 0 and channels != 1:
        raise ValueError("Grayscale PNG requires one channel")
    if bit_depth == 8:
        payload = array.astype(np.uint8, copy=False).tobytes()
    elif bit_depth == 16:
        payload = array.astype(">u2", copy=False).tobytes()
    else:
        raise ValueError("only 8-bit and 16-bit PNG output is supported")
    stride = width * channels * (bit_depth // 8)
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        start = row * stride
        scanlines.extend(payload[start : start + stride])
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    idat = zlib.compress(bytes(scanlines), level=9)
    return _PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def encode_rgb8(array: np.ndarray) -> bytes:
    return _encode_png(np.asarray(array, dtype=np.uint8), color_type=2, bit_depth=8)


def encode_gray8(array: np.ndarray) -> bytes:
    return _encode_png(np.asarray(array, dtype=np.uint8), color_type=0, bit_depth=8)


def encode_gray16(array: np.ndarray) -> bytes:
    return _encode_png(np.asarray(array, dtype=np.uint16), color_type=0, bit_depth=16)


def atomic_write(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".png-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_rgb8(path: str, array: np.ndarray) -> None:
    atomic_write(path, encode_rgb8(array))


def write_gray8(path: str, array: np.ndarray) -> None:
    atomic_write(path, encode_gray8(array))


def write_gray16(path: str, array: np.ndarray) -> None:
    atomic_write(path, encode_gray16(array))
