from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError


DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 576
PAGE_ID = 0x0001


@dataclass
class BitmapRegion:
    pixels: np.ndarray
    width: int
    height: int
    x: int
    y: int
    palette: list[tuple[int, int, int, int]]
    render_method: str


_FONT_5X7: dict[str, list[str]] = {
    " ": ["00000"] * 7,
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    ",": ["00000", "00000", "00000", "00000", "00110", "00110", "00100"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ":": ["00000", "00110", "00110", "00000", "00110", "00110", "00000"],
    "'": ["00100", "00100", "00000", "00000", "00000", "00000", "00000"],
}


def _fallback_rgba(text: str, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT) -> np.ndarray:
    scale = 4
    glyph_w = 6 * scale
    glyph_h = 9 * scale
    max_chars = max(1, (width - 40) // glyph_w)
    lines: list[str] = []
    for raw_line in text.splitlines() or [text]:
        line = raw_line.upper()
        while len(line) > max_chars:
            lines.append(line[:max_chars])
            line = line[max_chars:]
        lines.append(line)
    block_w = max((len(line) for line in lines), default=1) * glyph_w
    block_h = max(1, len(lines)) * glyph_h
    x0 = max(0, (width - block_w) // 2)
    y0 = max(20, height - block_h - 28)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for row_index, line in enumerate(lines):
        for col_index, char in enumerate(line):
            glyph = _FONT_5X7.get(char, _FONT_5X7["?"])
            gx = x0 + col_index * glyph_w
            gy = y0 + row_index * glyph_h
            for py, bits in enumerate(glyph):
                for px, bit in enumerate(bits):
                    if bit == "1":
                        y = gy + py * scale
                        x = gx + px * scale
                        if 0 <= y < height - scale and 0 <= x < width - scale:
                            rgba[y : y + scale, x : x + scale] = (255, 255, 255, 255)
    return rgba


def render_rgba(
    text: str,
    ffmpeg: str,
    work_dir: str | Path,
    width: int = DISPLAY_WIDTH,
    height: int = DISPLAY_HEIGHT,
) -> tuple[np.ndarray, str]:
    import subprocess
    import tempfile

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render_", dir=str(work_dir)) as temp_name:
        temp = Path(temp_name)
        (temp / "cue.srt").write_text(
            "1\n00:00:00,000 --> 00:00:10,000\n" + text + "\n",
            encoding="utf-8",
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black@0.0:size={width}x{height}:rate=1:duration=1,format=rgba",
            "-vf",
            "subtitles=cue.srt:alpha=1,format=rgba",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(temp),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _fallback_rgba(text, width, height), f"fallback_font ({exc})"
        expected = width * height * 4
        if completed.returncode == 0 and len(completed.stdout) == expected:
            return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(height, width, 4), "ffmpeg_libass"
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        return _fallback_rgba(text, width, height), f"fallback_font ({message[:160]})"


def _palette() -> list[tuple[int, int, int, int]]:
    # (Y, Cr/Cb offset, Cr/Cb offset, transparency)
    entries = [(0, 0, 0, 255)]
    for transparency in (0, 47, 111, 191, 239):
        entries.append((235, 0, 0, transparency))
        entries.append((16, 0, 0, transparency))
    while len(entries) < 16:
        entries.append((0, 0, 0, 255))
    return entries


def quantize_rgba(rgba: np.ndarray, render_method: str) -> BitmapRegion:
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3].astype(np.int16)
    ys, xs = np.where(alpha >= 8)
    if ys.size == 0:
        # A valid but invisible object keeps the stream structure regular.
        xs = np.array([width // 2])
        ys = np.array([height - 30])
        alpha = alpha.copy()
        alpha[ys[0], xs[0]] = 8
    margin_x, margin_y = 6, 4
    left = max(0, int(xs.min()) - margin_x)
    right = min(width, int(xs.max()) + margin_x + 1)
    top = max(0, int(ys.min()) - margin_y)
    bottom = min(height, int(ys.max()) + margin_y + 1)
    if (right - left) % 2:
        right = min(width, right + 1)
    if (bottom - top) % 2:
        bottom = min(height, bottom + 1)
    if right <= left:
        right = min(width, left + 2)
    if bottom <= top:
        bottom = min(height, top + 2)
    crop = rgba[top:bottom, left:right].copy()
    crop_alpha = crop[:, :, 3].astype(np.int32)
    luma = (
        crop[:, :, 0].astype(np.float64) * 0.299
        + crop[:, :, 1].astype(np.float64) * 0.587
        + crop[:, :, 2].astype(np.float64) * 0.114
    )
    pixels = np.zeros(crop.shape[:2], dtype=np.uint8)
    # digitize returns 0 for the faintest pixels and 4 for opaque pixels;
    # invert it so palette index 1 is fully opaque white.
    alpha_bins = 4 - np.digitize(crop_alpha, [32, 96, 160, 224])
    is_black = luma < 80
    for bin_index in range(5):
        mask = (alpha_bins == bin_index) & (crop_alpha >= 8)
        white_index = 1 + bin_index * 2
        black_index = white_index + 1
        pixels[mask & ~is_black] = white_index
        pixels[mask & is_black] = black_index
    region_h = pixels.shape[0]
    region_w = pixels.shape[1]
    x = max(0, min(width - region_w, (width - region_w) // 2))
    y = max(0, min(height - region_h, height - region_h - 24))
    return BitmapRegion(
        pixels=pixels,
        width=region_w,
        height=region_h,
        x=x,
        y=y,
        palette=_palette(),
        render_method=render_method,
    )


def make_region(text: str, ffmpeg: str, work_dir: str | Path) -> BitmapRegion:
    rgba, method = render_rgba(text, ffmpeg, work_dir)
    return quantize_rgba(rgba, method)


def _rle4(bitmap: np.ndarray) -> bytes:
    height, width = bitmap.shape
    output = bytearray()
    for row in bitmap:
        output.append(0x11)
        bit_buffer = 0
        bit_count = 4

        def put(value: int) -> None:
            nonlocal bit_buffer, bit_count
            bit_buffer |= value << bit_count
            bit_count -= 4
            if bit_count < 0:
                bit_count = 4
                output.append(bit_buffer & 0xFF)
                bit_buffer = 0

        x = 0
        while x < width:
            x1 = x + 1
            color = int(row[x])
            while x1 < width and int(row[x1]) == color:
                x1 += 1
            length = x1 - x
            if color == 0 and length == 2:
                put(0)
                put(0xD)
            elif color == 0 and 3 <= length <= 9:
                put(0)
                put(length - 2)
            elif 4 <= length <= 7:
                put(0)
                put(8 + length - 4)
                put(color)
            elif 9 <= length <= 24:
                put(0)
                put(0xE)
                put(length - 9)
                put(color)
            elif length >= 25:
                length = min(length, 280)
                value = length - 25
                put(0)
                put(0xF)
                put(value >> 4)
                put(value & 0xF)
                put(color)
            else:
                put(color)
                if color == 0:
                    put(0xC)
                length = 1
            x += length
        put(0)
        put(0)
        if bit_count != 4:
            output.append(bit_buffer & 0xFF)
        output.append(0xF0)
    return bytes(output)


def _segment(segment_type: int, page_id: int, body: bytes) -> bytes:
    return bytes([0x0F, segment_type]) + page_id.to_bytes(2, "big") + len(body).to_bytes(2, "big") + body


def _display_definition_segment() -> bytes:
    body = bytes([0x00]) + (DISPLAY_WIDTH - 1).to_bytes(2, "big") + (DISPLAY_HEIGHT - 1).to_bytes(2, "big")
    return _segment(0x14, PAGE_ID, body)


def _page_composition(region: BitmapRegion | None, version: int) -> bytes:
    body = bytearray([30, ((version & 0x0F) << 4) | (2 << 2) | 0x03])
    if region is not None:
        body.append(0)
        body.append(0xFF)
        body.extend(region.x.to_bytes(2, "big"))
        body.extend(region.y.to_bytes(2, "big"))
    return _segment(0x10, PAGE_ID, bytes(body))


def _clut_definition(version: int) -> bytes:
    body = bytearray([0, ((version & 0x0F) << 4) | 0x0F])
    for index, (y_value, cr_value, cb_value, transparency) in enumerate(_palette()):
        body.append(index)
        body.append(0x5F)
        body.append(y_value & 0xFF)
        body.append((128 + cr_value) & 0xFF)
        body.append((128 + cb_value) & 0xFF)
        body.append(transparency & 0xFF)
    return _segment(0x12, PAGE_ID, bytes(body))


def _region_composition(region: BitmapRegion, version: int) -> bytes:
    body = bytearray()
    body.append(0)
    body.append(((version & 0x0F) << 4) | 0x07)
    body.extend(region.width.to_bytes(2, "big"))
    body.extend(region.height.to_bytes(2, "big"))
    body.append(0x4B)  # compatibility=2 (4-bit CLUT), depth=2 (4 bpp), reserved
    body.append(0)  # CLUT id
    body.append(0)  # 8-bit pixel code
    body.append(0x03)  # 4-bit and 2-bit fill codes
    body.extend((0).to_bytes(2, "big"))
    body.append(0x00)  # object type/provider, horizontal high nibble
    body.append(0x00)
    body.append(0x00)
    body.append(0x00)
    return _segment(0x11, PAGE_ID, bytes(body))


def _object_data(region: BitmapRegion, version: int) -> bytes:
    top = _rle4(region.pixels[0::2, :])
    bottom = _rle4(region.pixels[1::2, :])
    body = bytearray()
    body.extend((0).to_bytes(2, "big"))
    body.append(((version & 0x0F) << 4) | 0x01)
    body.extend(len(top).to_bytes(2, "big"))
    body.extend(len(bottom).to_bytes(2, "big"))
    body.extend(top)
    body.extend(bottom)
    return _segment(0x13, PAGE_ID, bytes(body))


def build_show_segments(region: BitmapRegion, version: int) -> bytes:
    return b"".join(
        [
            _display_definition_segment(),
            _page_composition(region, version),
            _region_composition(region, version),
            _clut_definition(version),
            _object_data(region, version),
            _segment(0x80, PAGE_ID, b""),
        ]
    )


def build_clear_segments(version: int) -> bytes:
    return b"".join(
        [
            _display_definition_segment(),
            _page_composition(None, version),
            _segment(0x80, PAGE_ID, b""),
        ]
    )


def _encode_pts(pts: int) -> bytes:
    pts &= (1 << 33) - 1
    return bytes(
        [
            0x21 | (((pts >> 30) & 0x07) << 1),
            (pts >> 22) & 0xFF,
            0x01 | (((pts >> 15) & 0x7F) << 1),
            (pts >> 7) & 0xFF,
            0x01 | ((pts & 0x7F) << 1),
        ]
    )


def build_subtitle_pes(segments: bytes, pts: int) -> bytes:
    payload = b"\x20\x00" + segments
    pes_packet_length = 3 + 5 + len(payload)
    if pes_packet_length > 0xFFFF:
        raise BusinessError(
            "E_FORMAT",
            "dvb_build",
            "DVB subtitle PES exceeds 65535 bytes",
            {"payload_length": len(payload)},
        )
    return (
        b"\x00\x00\x01\xbd"
        + pes_packet_length.to_bytes(2, "big")
        + b"\x84\x80\x05"
        + _encode_pts(pts)
        + payload
    )


def parse_subtitle_pes(payload: bytes) -> dict[str, Any]:
    if len(payload) < 2 or payload[:2] != b"\x20\x00":
        raise ValueError("invalid DVB subtitle PES payload")
    position = 2
    segments: list[dict[str, Any]] = []
    while position + 6 <= len(payload):
        if payload[position] == 0xFF:
            break
        if payload[position] != 0x0F:
            raise ValueError("invalid DVB subtitle segment sync byte")
        segment_type = payload[position + 1]
        page_id = int.from_bytes(payload[position + 2 : position + 4], "big")
        length = int.from_bytes(payload[position + 4 : position + 6], "big")
        body_start = position + 6
        body_end = body_start + length
        if body_end > len(payload):
            raise ValueError("truncated DVB subtitle segment")
        segments.append(
            {
                "segment_type": segment_type,
                "page_id": page_id,
                "length": length,
                "body": payload[body_start:body_end],
            }
        )
        position = body_end
    page = next((item for item in segments if item["segment_type"] == 0x10), None)
    region_count = 0
    if page is not None:
        body = page["body"]
        if len(body) >= 2:
            region_count = (len(body) - 2) // 6
    return {
        "segment_types": [item["segment_type"] for item in segments],
        "segment_count": len(segments),
        "page_id": page["page_id"] if page else None,
        "region_count": region_count,
        "has_object": any(item["segment_type"] == 0x13 for item in segments),
        "has_end": any(item["segment_type"] == 0x80 for item in segments),
    }
