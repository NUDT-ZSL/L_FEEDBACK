"""Minimal, deterministic RIFF/WAVE I/O.

Reading supports integer PCM 8/16/24/32-bit (chunk format 0x0001) and
WAVE_FORMAT_EXTENSIBLE (0xFFFE) whose sub-format is PCM.  IEEE float
(0x0003) and any other encoding are rejected with E_UNSUPPORTED_FORMAT so
the caller can record a business error instead of crashing.

Writing emits canonical PCM WAV (extensible when channels > 2) with a
deterministic byte layout and atomic replace.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from .errors import Code, PipelineError

_FMT_PCM = 0x0001
_FMT_FLOAT = 0x0003
_FMT_EXTENSIBLE = 0xFFFE

# KSDATAFORMAT_SUBTYPE_PCM as stored in WAVE_FORMAT_EXTENSIBLE fmt chunks.
_GUID_PCM = struct.pack("<IHH8s", 0x0001, 0x0000, 0x0010,
                        b"\x80\x00\x00\xaa\x00\x38\x9b\x71")

_INT_SCALE = {8: 128.0, 16: 32768.0, 24: 8388608.0, 32: 2147483648.0}


def _parse_fmt(body: bytes) -> dict:
    if len(body) < 16:
        raise PipelineError(Code.E_DECODE_ERROR, "fmt chunk too small")
    fmt, channels, rate, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
    if fmt == _FMT_EXTENSIBLE:
        if len(body) < 40:
            raise PipelineError(Code.E_DECODE_ERROR, "extensible fmt too small")
        sub = struct.unpack("<H", body[24:26])[0]
        valid_bits = struct.unpack("<H", body[18:20])[0]
        fmt = sub
        if valid_bits:
            bits = valid_bits
    if fmt == _FMT_FLOAT:
        raise PipelineError(Code.E_UNSUPPORTED_FORMAT,
                            f"IEEE float {bits}-bit WAV is not supported")
    if fmt != _FMT_PCM:
        raise PipelineError(Code.E_UNSUPPORTED_FORMAT,
                            f"unsupported WAV encoding tag 0x{fmt:04x}")
    if bits not in _INT_SCALE:
        raise PipelineError(Code.E_UNSUPPORTED_FORMAT,
                            f"unsupported bit depth {bits}")
    if channels < 1:
        raise PipelineError(Code.E_DECODE_ERROR, "zero channels")
    return {"channels": channels, "rate": rate, "bits": bits}


def _decode_pcm(pcm: bytes, bits: int, channels: int) -> np.ndarray:
    if bits == 8:
        v = np.frombuffer(pcm, dtype=np.uint8).astype(np.float64)
        x = (v - 128.0) / 128.0
    elif bits == 16:
        v = np.frombuffer(pcm, dtype="<i2")
        x = v.astype(np.float64) / 32768.0
    elif bits == 24:
        n = len(pcm) // 3 * 3
        a = np.frombuffer(pcm[:n], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float64) / 8388608.0
    else:  # 32
        v = np.frombuffer(pcm, dtype="<i4")
        x = v.astype(np.float64) / 2147483648.0
    if len(x) % channels:
        x = x[: len(x) - (len(x) % channels)]
    return x.reshape(-1, channels)


def read_wav(path: str):
    """Return (samples float64 (n, ch) in [-1, 1], sample_rate, channels)."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise PipelineError(Code.E_DECODE_ERROR, f"cannot read file: {exc}")
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise PipelineError(Code.E_DECODE_ERROR, "not a RIFF/WAVE file")
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8: pos + 8 + size]
        if cid == b"fmt ":
            fmt = _parse_fmt(body)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None:
        raise PipelineError(Code.E_DECODE_ERROR, "missing fmt chunk")
    if pcm is None:
        raise PipelineError(Code.E_DECODE_ERROR, "missing data chunk")
    x = _decode_pcm(pcm, fmt["bits"], fmt["channels"])
    if x.shape[0] == 0:
        raise PipelineError(Code.E_DECODE_ERROR, "empty audio data")
    return x, fmt["rate"], fmt["channels"]


def _encode_pcm(x: np.ndarray, bits: int) -> bytes:
    scale = _INT_SCALE[bits]
    v = np.clip(np.rint(x * scale), -scale, scale - 1)
    if bits == 8:
        return (v.astype(np.int16) + 128).astype(np.uint8).tobytes()
    if bits == 16:
        return v.astype("<i2").tobytes()
    if bits == 24:
        b = v.astype("<i4").reshape(-1).view(np.uint8).reshape(-1, 4)
        return b[:, :3].tobytes()
    return v.astype("<i4").tobytes()


def write_wav(path: str, x: np.ndarray, rate: int, bits: int) -> None:
    """Write (n, ch) float samples as integer PCM WAV, atomically."""
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    pcm = _encode_pcm(np.ascontiguousarray(x, dtype=np.float64), bits)
    block_align = channels * bits // 8
    byte_rate = rate * block_align
    if channels > 2:
        mask = (1 << channels) - 1 if channels <= 18 else 0x3F
        fmt = struct.pack("<HHIIHHHHI16s", _FMT_EXTENSIBLE, channels, rate,
                          byte_rate, block_align, bits, 22, bits, mask,
                          _GUID_PCM)
    else:
        fmt = struct.pack("<HHIIHH", _FMT_PCM, channels, rate, byte_rate,
                          block_align, bits)
    riff_size = 4 + (8 + len(fmt)) + (8 + len(pcm))
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    header += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    header += b"data" + struct.pack("<I", len(pcm))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(header)
        fh.write(pcm)
    os.replace(tmp, path)
