"""WAV input/output.

Reading prefers SoundFile (libsndfile); a small pure-Python PCM reader is
kept as a fallback so the engine never depends on an external audio tool.
Writing is done by this module directly: 48 kHz / 24-bit PCM with an
explicit RIFF ``fact`` chunk, giving full control over the required layout.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np

from .errors import MixerError


def read_wav(path) -> Tuple[np.ndarray, int]:
    """Read a mono PCM WAV, returning ``(float64 samples in [-1, 1], sr)``."""
    path = Path(path)
    if not path.is_file():
        raise MixerError("E_TRACK_NOT_FOUND", details={"path": str(path)})
    try:
        import soundfile as sf  # local import: optional at import time

        data, sample_rate = sf.read(str(path), dtype="float64",
                                    always_2d=False)
    except Exception:
        data, sample_rate = _read_pcm_fallback(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sample_rate)


def _read_pcm_fallback(path: Path) -> Tuple[np.ndarray, int]:
    """Minimal RIFF/FLOAT-less PCM reader (8/16/24/32-bit, mono/stereo)."""
    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise MixerError("E_TRACK_FORMAT", details={"path": str(path)})
        fmt = None
        frames = None
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", header)
            payload = handle.read(chunk_size)
            if chunk_size % 2:
                handle.read(1)
            if chunk_id == b"fmt ":
                fmt = struct.unpack("<HHIIHH", payload[:16])
            elif chunk_id == b"data":
                frames = payload
        if fmt is None or frames is None:
            raise MixerError("E_TRACK_FORMAT", details={"path": str(path)})
        tag, channels, sample_rate, _byte_rate, _block_align, bits = fmt
        if tag != 1:
            raise MixerError("E_TRACK_FORMAT",
                             details={"path": str(path), "tag": tag})
        if bits == 16:
            raw = np.frombuffer(frames, dtype="<i2").astype(np.float64)
            data = raw / 32768.0
        elif bits == 24:
            count = len(frames) // 3
            words = np.zeros(count, dtype="<i4")
            view = words.view(np.uint8).reshape(count, 4)
            view[:, :3] = np.frombuffer(
                frames[: count * 3], dtype=np.uint8).reshape(count, 3)
            sign = (view[:, 2:3] & 0x80).astype(np.int32) * 0x010101
            view[:, 3:4] = np.where(sign != 0, 255, 0).astype(np.uint8)
            data = words.astype(np.float64) / 8388608.0
        elif bits == 32:
            data = np.frombuffer(frames, dtype="<i4").astype(np.float64) \
                / 2147483648.0
        elif bits == 8:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
                    - 128.0) / 128.0
        else:
            raise MixerError("E_TRACK_FORMAT",
                             details={"bits": bits, "path": str(path)})
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, sample_rate


def _int24_bytes(values: np.ndarray) -> bytes:
    clipped = np.clip(values, -8388608, 8388607).astype("<i4")
    count = len(clipped)
    out = np.empty((count, 4), dtype=np.uint8)
    out.view("<i4")[:] = clipped.reshape(count, 1)
    return out[:, :3].tobytes()


def write_wav_24bit(path, samples: np.ndarray, sample_rate: int) -> int:
    """Write mono 48 kHz / 24-bit PCM WAV with RIFF header + ``fact`` chunk.

    Returns the number of sample frames written.
    """
    path = Path(path)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        quantized = np.round(samples * 8388608.0).astype(np.int64)
        audio_bytes = _int24_bytes(quantized)
        channels = 1
        bits_per_sample = 24
        block_align = channels * bits_per_sample // 8
        byte_rate = sample_rate * block_align
        data_size = len(audio_bytes)
        fact_payload = struct.pack("<I", len(samples))
        fmt_payload = struct.pack(
            "<HHIIHH",
            1,                 # WAVE_FORMAT_PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        with path.open("wb") as handle:
            handle.write(b"RIFF")
            body_size = (4 + 24) + (8 + len(fact_payload)) \
                + (8 + data_size)
            handle.write(struct.pack("<I", body_size))
            handle.write(b"WAVE")
            handle.write(b"fmt ")
            handle.write(struct.pack("<I", len(fmt_payload)))
            handle.write(fmt_payload)
            handle.write(b"fact")
            handle.write(struct.pack("<I", len(fact_payload)))
            handle.write(fact_payload)
            handle.write(b"data")
            handle.write(struct.pack("<I", data_size))
            handle.write(audio_bytes)
    except OSError as exc:
        raise MixerError("E_OUTPUT_IO", str(exc),
                         {"path": str(path)}) from exc
    return len(samples)


def read_wav_chunks(path) -> dict:
    """Inspect RIFF chunk layout (used by verification/tests)."""
    path = Path(path)
    chunks = []
    with path.open("rb") as handle:
        header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise MixerError("E_TRACK_FORMAT", details={"path": str(path)})
        while True:
            head = handle.read(8)
            if len(head) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", head)
            chunks.append((chunk_id.decode("ascii", "replace"),
                           int(chunk_size)))
            handle.seek(chunk_size + (chunk_size % 2), 1)
    return {"chunks": chunks, "size": path.stat().st_size}
