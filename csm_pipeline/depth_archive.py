"""Binary depth-texture archive with an explicit, validated header.

Little-endian layout (all integers unsigned 32-bit)::

    magic          4 bytes  b"CSMD"
    version        uint32   currently 1
    cascade_count  uint32
    width          uint32
    height         uint32
    pixel_dtype    uint32   1 = float32
    reserved       uint32   must be zero
    payload        cascade_count * height * width float32 values
"""

from __future__ import annotations

import os
import struct

import numpy as np

from .errors import (
    ERR_ARCHIVE_HEADER,
    ERR_ARCHIVE_MISSING,
    ERR_ARCHIVE_TRUNCATED,
    ERR_IO,
    PipelineError,
)

MAGIC = b"CSMD"
VERSION = 1
HEADER_SIZE = 32
PIXEL_DTYPE_FLOAT32 = 1

# 4-byte magic plus seven uint32 fields = 32 bytes.
_HEADER = struct.Struct("<4s7I")
assert _HEADER.size == HEADER_SIZE


def write_depth_archive(
    path: str,
    depth_maps: np.ndarray,
) -> None:
    """Serialize a ``(cascade, height, width)`` float32 stack."""
    stack = np.asarray(depth_maps, dtype=np.float32)
    if stack.ndim != 3:
        raise PipelineError(ERR_IO, "depth archive expects a 3-D stack")
    cascade_count, height, width = stack.shape
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        int(cascade_count),
        int(width),
        int(height),
        PIXEL_DTYPE_FLOAT32,
        0,
        0,
    )
    try:
        with open(path, "wb") as handle:
            handle.write(header)
            handle.write(stack.astype("<f4", copy=False).tobytes())
    except OSError as exc:
        raise PipelineError(ERR_IO, f"failed to write depth archive: {exc}") from exc


def read_depth_archive(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise PipelineError(
            ERR_ARCHIVE_MISSING, f"depth archive does not exist: {path}"
        )
    try:
        with open(path, "rb") as handle:
            header_bytes = handle.read(HEADER_SIZE)
            if len(header_bytes) != HEADER_SIZE:
                raise PipelineError(
                    ERR_ARCHIVE_TRUNCATED,
                    f"depth archive header truncated ({len(header_bytes)} bytes)",
                )
            try:
                (
                    magic,
                    version,
                    cascade_count,
                    width,
                    height,
                    pixel_dtype,
                    reserved,
                    reserved2,
                ) = _HEADER.unpack(header_bytes)
            except struct.error as exc:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER, f"malformed archive header: {exc}"
                ) from exc

            if magic != MAGIC:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER, "bad magic bytes in depth archive"
                )
            if version != VERSION:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER,
                    f"unsupported archive version {version}, expected {VERSION}",
                )
            if pixel_dtype != PIXEL_DTYPE_FLOAT32:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER,
                    f"unsupported pixel dtype code {pixel_dtype}",
                )
            if reserved != 0:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER, "archive reserved field must be zero"
                )
            if reserved2 != 0:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER, "archive reserved2 field must be zero"
                )
            if cascade_count == 0 or width == 0 or height == 0:
                raise PipelineError(
                    ERR_ARCHIVE_HEADER,
                    "archive dimensions must all be strictly positive",
                )

            expected = cascade_count * width * height * 4
            payload = handle.read()
    except PipelineError:
        raise
    except OSError as exc:
        raise PipelineError(
            ERR_ARCHIVE_MISSING, f"cannot read depth archive: {exc}"
        ) from exc

    if len(payload) != expected:
        raise PipelineError(
            ERR_ARCHIVE_TRUNCATED,
            f"archive payload has {len(payload)} bytes, expected {expected}",
        )
    stack = np.frombuffer(payload, dtype="<f4").reshape(
        cascade_count, height, width
    )
    if not np.isfinite(stack).all():
        raise PipelineError(
            ERR_ARCHIVE_HEADER, "archive payload contains non-finite pixels"
        )
    return np.array(stack, dtype=np.float32, copy=True)

