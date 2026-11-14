"""Small, dependency-free PNG reader/writer for 8- and 16-bit material maps."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import zlib

import numpy as np

from .errors import invalid_format, io_error, missing_file


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


@dataclass(frozen=True)
class PNGImage:
    rgb: np.ndarray
    width: int
    height: int
    bit_depth: int
    color_type: int


def _read_chunks(path: Path) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise io_error(f"Cannot read PNG file {path}: {exc}") from exc

    if not data.startswith(PNG_SIGNATURE):
        raise invalid_format(f"{path} is not a PNG file")

    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise invalid_format(f"{path} contains a truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise invalid_format(f"{path} contains a truncated PNG chunk")
        payload = data[payload_start:payload_end]
        stored_crc = struct.unpack(">I", data[payload_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise invalid_format(f"{path} has an invalid PNG CRC")
        chunks.append((chunk_type, payload))
        offset = crc_end
        if chunk_type == b"IEND":
            saw_iend = True
            break

    if not saw_iend:
        raise invalid_format(f"{path} has no PNG IEND chunk")
    return data, chunks


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_rows(
    raw: bytes, offset: int, height: int, row_bytes: int, bpp: int
) -> tuple[bytes, int]:
    expected = height * (row_bytes + 1)
    if offset + expected > len(raw):
        raise invalid_format("PNG image data is shorter than its scanline layout")
    output = bytearray(height * row_bytes)
    previous = bytearray(row_bytes)

    for y in range(height):
        source = offset + y * (row_bytes + 1)
        filter_type = raw[source]
        row = bytearray(raw[source + 1 : source + 1 + row_bytes])

        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(bpp, row_bytes):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filter_type == 2:
            for i in range(row_bytes):
                row[i] = (row[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(row_bytes):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(row_bytes):
                left = row[i - bpp] if i >= bpp else 0
                upper_left = previous[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth(left, previous[i], upper_left)) & 0xFF
        else:
            raise invalid_format(f"PNG uses unsupported filter type {filter_type}")

        start = y * row_bytes
        output[start : start + row_bytes] = row
        previous = row

    return bytes(output), offset + expected


def _unfilter_scanlines(
    compressed: bytes, width: int, height: int, channels: int, bytes_per_sample: int
) -> bytes:
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise invalid_format(f"PNG image data cannot be decompressed: {exc}") from exc

    row_bytes = width * channels * bytes_per_sample
    bpp = max(1, channels * bytes_per_sample)
    output, consumed = _unfilter_rows(raw, 0, height, row_bytes, bpp)
    expected = height * (row_bytes + 1)
    if consumed != expected or len(raw) != expected:
        raise invalid_format(
            f"PNG image data length is {len(raw)}, expected {expected}"
        )
    return output


def _unfilter_adam7(
    compressed: bytes,
    width: int,
    height: int,
    channels: int,
    bytes_per_sample: int,
    bit_depth: int,
) -> np.ndarray:
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise invalid_format(f"PNG image data cannot be decompressed: {exc}") from exc

    if bit_depth == 8:
        samples = np.zeros((height, width, channels), dtype=np.uint8)
    else:
        samples = np.zeros((height, width, channels), dtype=np.uint16)
    bpp = max(1, channels * bytes_per_sample)
    offset = 0

    for x_start, y_start, x_step, y_step in _ADAM7_PASSES:
        pass_width = (width - x_start + x_step - 1) // x_step
        pass_height = (height - y_start + y_step - 1) // y_step
        if pass_width <= 0 or pass_height <= 0:
            continue
        row_bytes = pass_width * channels * bytes_per_sample
        rows, offset = _unfilter_rows(
            raw, offset, pass_height, row_bytes, bpp
        )
        if bit_depth == 8:
            pass_samples = np.frombuffer(rows, dtype=np.uint8).reshape(
                pass_height, pass_width, channels
            )
        else:
            pass_samples = np.frombuffer(rows, dtype=">u2").reshape(
                pass_height, pass_width, channels
            )
        samples[
            y_start::y_step,
            x_start::x_step,
            :,
        ] = pass_samples

    if offset != len(raw):
        raise invalid_format("Adam7 PNG image data length is inconsistent")
    return samples


def read_png_rgb(path: str | os.PathLike[str]) -> PNGImage:
    """Read a PNG and return normalized RGB values in the range [0, 1]."""

    png_path = Path(path)
    if not png_path.exists():
        raise missing_file(f"Input file not found: {png_path}")
    if not png_path.is_file():
        raise missing_file(f"Input path is not a file: {png_path}")

    _, chunks = _read_chunks(png_path)
    if not chunks or chunks[0][0] != b"IHDR":
        raise invalid_format(f"{png_path} does not start with IHDR")
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise invalid_format(f"{png_path} has an invalid IHDR chunk")

    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if width <= 0 or height <= 0:
        raise invalid_format(f"{png_path} has invalid dimensions")
    if bit_depth not in (8, 16):
        raise invalid_format(
            f"{png_path} uses {bit_depth}-bit samples; only 8/16-bit PNG is supported"
        )
    if color_type not in _CHANNELS:
        raise invalid_format(f"{png_path} uses unsupported PNG color type {color_type}")
    if compression != 0 or filter_method != 0:
        raise invalid_format(f"{png_path} uses unsupported PNG compression/filter method")
    if interlace not in (0, 1):
        raise invalid_format(f"{png_path} uses an unsupported interlace method")
    if color_type == 3 and bit_depth != 8:
        raise invalid_format(f"{png_path} has an invalid palette bit depth")

    idat = b"".join(payload for kind, payload in chunks if kind == b"IDAT")
    if not idat:
        raise invalid_format(f"{png_path} has no IDAT image data")

    palette: np.ndarray | None = None
    if color_type == 3:
        plte = next((payload for kind, payload in chunks if kind == b"PLTE"), None)
        if plte is None or len(plte) % 3 != 0 or not plte:
            raise invalid_format(f"{png_path} has no valid PLTE palette")
        palette = (
            np.frombuffer(plte, dtype=np.uint8).reshape(-1, 3).astype(np.float64)
            / 255.0
        )

    channels = _CHANNELS[color_type]
    bytes_per_sample = bit_depth // 8
    if interlace == 0:
        scanlines = _unfilter_scanlines(
            idat, width, height, channels, bytes_per_sample
        )
        if bit_depth == 8:
            samples = np.frombuffer(scanlines, dtype=np.uint8).reshape(
                height, width, channels
            )
        else:
            samples = np.frombuffer(scanlines, dtype=">u2").reshape(
                height, width, channels
            )
    else:
        samples = _unfilter_adam7(
            idat, width, height, channels, bytes_per_sample, bit_depth
        )
    if bit_depth == 8:
        normalized = samples.astype(np.float64) / 255.0
    else:
        normalized = samples.astype(np.float64) / 65535.0

    if color_type == 3:
        assert palette is not None
        indices = samples[:, :, 0].astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= palette.shape[0]):
            raise invalid_format(f"{png_path} contains an out-of-range palette index")
        rgb = palette[indices]
    elif color_type in (0, 4):
        gray = normalized[:, :, 0]
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        rgb = normalized[:, :, :3]

    return PNGImage(
        rgb=np.ascontiguousarray(rgb, dtype=np.float64),
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
    )


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def encode_rgb_png(linear_rgb: np.ndarray, bit_depth: int, gamma: float) -> bytes:
    """Encode a linear RGB float image as an 8/16-bit PNG."""

    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")
    if gamma not in (1.0, 2.2):
        raise ValueError("gamma must be 1.0 or 2.2")

    image = np.asarray(linear_rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains NaN or infinity")

    encoded = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
    maximum = 255 if bit_depth == 8 else 65535
    quantized = np.rint(encoded * maximum).astype(np.uint32)
    quantized = np.clip(quantized, 0, maximum).astype(np.uint16 if bit_depth == 16 else np.uint8)

    height, width, _ = quantized.shape
    if bit_depth == 8:
        scanlines = b"".join(
            b"\x00" + quantized[y].tobytes(order="C") for y in range(height)
        )
    else:
        scanlines = b"".join(
            b"\x00" + quantized[y].astype(">u2").tobytes(order="C")
            for y in range(height)
        )

    ihdr = struct.pack(
        ">IIBBBBB", width, height, bit_depth, 2, 0, 0, 0
    )
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _chunk(b"IEND", b"")
    )


def write_rgb_png(
    path: str | os.PathLike[str],
    linear_rgb: np.ndarray,
    bit_depth: int,
    gamma: float,
) -> None:
    destination = Path(path)
    try:
        payload = encode_rgb_png(linear_rgb, bit_depth, gamma)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(destination.name + ".tmp")
        temp_path.write_bytes(payload)
        os.replace(temp_path, destination)
    except OSError as exc:
        raise io_error(f"Cannot write PNG file {destination}: {exc}") from exc

