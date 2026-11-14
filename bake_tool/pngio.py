from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png(array: np.ndarray) -> tuple[bytes, str]:
    array = np.asarray(array)
    if array.ndim == 2:
        if array.dtype == np.uint16:
            color_type = 0
            bit_depth = 16
            rows = array.astype(">u2", copy=False).tobytes()
            bytes_per_row = array.shape[1] * 2
        elif array.dtype == np.uint8:
            color_type = 0
            bit_depth = 8
            rows = array.tobytes()
            bytes_per_row = array.shape[1]
        else:
            raise ValueError("Grayscale PNG must be uint8 or uint16.")
    elif array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.uint8:
        color_type = 2
        bit_depth = 8
        rows = array.tobytes()
        bytes_per_row = array.shape[1] * 3
    elif array.ndim == 3 and array.shape[2] == 4 and array.dtype == np.uint8:
        color_type = 6
        bit_depth = 8
        rows = array.tobytes()
        bytes_per_row = array.shape[1] * 4
    else:
        raise ValueError("Unsupported PNG array shape/dtype.")

    height, width = array.shape[:2]
    filtered = b"".join(
        b"\x00" + rows[row * bytes_per_row : (row + 1) * bytes_per_row]
        for row in range(height)
    )
    header = struct.pack(
        ">IIBBBBB",
        int(width),
        int(height),
        bit_depth,
        color_type,
        0,
        0,
        0,
    )
    png = _SIGNATURE + _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(filtered, level=9))
    png += _chunk(b"IEND", b"")
    label = {0: "gray", 2: "rgb", 6: "rgba"}[color_type]
    if color_type == 0 and bit_depth == 16:
        label = "gray16"
    return png, label


def write_png(path: str | Path, array: np.ndarray) -> None:
    payload, _ = encode_png(array)
    Path(path).write_bytes(payload)

