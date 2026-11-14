"""RG16F binary container with a self-describing JSON header."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import TEXTURE_FORMAT
from .errors import output_io, texture_corrupt


MAGIC = b"SKBAKETX"
CONTAINER_VERSION = 1


@dataclass
class TextureData:
    width: int
    height: int
    pixels: np.ndarray
    header: dict[str, Any]


def encode_texture_bytes(pixels: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    if pixels.ndim != 3 or pixels.shape[2] != 2:
        raise ValueError("texture pixels must have shape (height, width, 2)")
    height, width, _ = pixels.shape
    raw = np.asarray(pixels, dtype="<f2").tobytes(order="C")
    header: dict[str, Any] = {
        "format": TEXTURE_FORMAT,
        "version": CONTAINER_VERSION,
        "width": int(width),
        "height": int(height),
        "channels": 2,
        "channelOrder": ["u", "v"],
        "scalarType": "float16",
        "endianness": "little",
        "origin": "bottom-left",
        "rowOrder": "bottom-to-top",
        "dataBytes": len(raw),
        "dataSha256": hashlib.sha256(raw).hexdigest(),
    }
    header_bytes = json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    container = (
        MAGIC
        + struct.pack("<II", CONTAINER_VERSION, len(header_bytes))
        + header_bytes
        + raw
    )
    return container, header


def write_texture(path: Path, pixels: np.ndarray) -> tuple[dict[str, Any], str]:
    try:
        container, header = encode_texture_bytes(pixels)
        path.write_bytes(container)
    except OSError as exc:
        raise output_io(f"cannot write texture '{path}': {exc}") from exc
    return header, hashlib.sha256(container).hexdigest()


def read_texture(path: str | Path) -> TextureData:
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise texture_corrupt(f"cannot read texture '{source}': {exc}") from exc
    if len(data) < 16 or data[:8] != MAGIC:
        raise texture_corrupt("texture magic is invalid")
    version, header_len = struct.unpack_from("<II", data, 8)
    if version != CONTAINER_VERSION:
        raise texture_corrupt(f"unsupported texture container version {version}")
    header_start = 16
    payload_start = header_start + header_len
    if payload_start > len(data):
        raise texture_corrupt("texture header length exceeds file size")
    try:
        header = json.loads(data[header_start:payload_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise texture_corrupt(f"texture header JSON is invalid: {exc}") from exc
    if (
        header.get("format") != TEXTURE_FORMAT
        or header.get("channels") != 2
        or header.get("channelOrder") != ["u", "v"]
        or header.get("scalarType") != "float16"
        or header.get("endianness") != "little"
    ):
        raise texture_corrupt("texture header format/channel description is invalid")
    width = int(header.get("width", 0))
    height = int(header.get("height", 0))
    expected = width * height * 2 * 2
    payload = data[payload_start:]
    if width <= 0 or height <= 0 or expected != len(payload):
        raise texture_corrupt("texture payload size does not match header dimensions")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != header.get("dataSha256"):
        raise texture_corrupt("texture payload SHA-256 does not match header")
    pixels = np.frombuffer(payload, dtype="<f2").reshape(height, width, 2).astype(np.float64)
    return TextureData(width=width, height=height, pixels=pixels, header=header)

