from __future__ import annotations

import struct
import zlib
from typing import Optional

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def encode_png(array: np.ndarray, bit_depth: int) -> bytes:
    """Encode a deterministic RGB/gray PNG with filter type 0 on every row."""
    if array.dtype == np.bool_:
        raise ValueError("boolean arrays are not valid PNG pixels")
    if array.ndim == 2:
        height, width = array.shape
        channels = 1
        color_type = 0
        source = array
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        channels = 3
        color_type = 2
        source = array
    else:
        raise ValueError("PNG input must have shape (H, W) or (H, W, 3)")

    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    if bit_depth == 8:
        if source.dtype != np.uint8:
            raise ValueError("8-bit PNG input must be uint8")
        rows = [b"\x00" + source[y].tobytes(order="C") for y in range(height)]
    elif bit_depth == 16:
        if source.dtype not in (np.uint16, np.int32, np.int64):
            raise ValueError("16-bit PNG input must be uint16")
        if source.dtype != np.uint16:
            source = np.clip(source, 0, 65535).astype(np.uint16)
        big_endian = source.astype(">u2", copy=False)
        rows = [b"\x00" + big_endian[y].tobytes(order="C") for y in range(height)]
    else:
        raise ValueError("only 8-bit and 16-bit PNG output is supported")

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,
        0,
        0,
    )
    compressed = zlib.compress(b"".join(rows), level=9)
    return b"".join(
        (
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", compressed),
            _chunk(b"IEND", b""),
        )
    )


def quantize_8(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.floor(clipped * 255.0 + 0.5).astype(np.uint8)


def quantize_16(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.floor(clipped * 65535.0 + 0.5).astype(np.uint16)


def encode_rgb8(values: np.ndarray) -> bytes:
    return encode_png(quantize_8(values), 8)


def encode_gray8(values: np.ndarray) -> bytes:
    return encode_png(quantize_8(values), 8)


def encode_rgb16(values: np.ndarray) -> bytes:
    return encode_png(quantize_16(values), 16)


def encode_gray16(values: np.ndarray) -> bytes:
    return encode_png(quantize_16(values), 16)


def png_bit_depth(payload: bytes) -> Optional[int]:
    """Small inspection helper used by tests and verification scripts."""
    if not payload.startswith(PNG_SIGNATURE) or len(payload) < 33:
        return None
    if payload[12:16] != b"IHDR":
        return None
    return int(payload[24])
