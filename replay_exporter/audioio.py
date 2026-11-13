"""Deterministic WAV reading/writing on top of the standard library.

The project only supports 48 kHz / 16-bit PCM. Source clips are mono and the
rendered master is stereo. A minimal RIFF parser is used (instead of trusting
``wave`` alone) so that data chunks whose byte length is not a whole number of
frames can be reported with a dedicated business error.
"""

from __future__ import annotations

import os
import struct
import wave
from dataclasses import dataclass
from typing import List, Tuple

from .errors import ErrorCode, ExporterError

SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2
CHANNELS_MONO = 1
CHANNELS_STEREO = 2
INT16_MAX = 32767
INT16_MIN = -32768


@dataclass
class WavData:
    """Decoded mono PCM source clip."""

    path: str
    sample_rate: int
    frames: List[int]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_ms(self) -> float:
        return len(self.frames) * 1000.0 / self.sample_rate


def _read_riff_chunks(path: str) -> Tuple[bytes, dict]:
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 12 or blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ExporterError(
            ErrorCode.INVALID_WAV,
            f"{path}: not a RIFF/WAVE file",
            detail={"path": path},
        )
    chunks = {}
    cursor = 12
    while cursor + 8 <= len(blob):
        chunk_id = blob[cursor : cursor + 4]
        (chunk_size,) = struct.unpack_from("<I", blob, cursor + 4)
        body_start = cursor + 8
        body_end = body_start + chunk_size
        if body_end > len(blob):
            raise ExporterError(
                ErrorCode.INVALID_WAV,
                f"{path}: truncated RIFF chunk {chunk_id!r}",
                detail={"path": path, "chunk": chunk_id.decode("ascii", "replace")},
            )
        chunks.setdefault(chunk_id, []).append((body_start, chunk_size))
        cursor = body_end + (chunk_size & 1)  # RIFF pads odd chunks with one byte
    return blob, chunks


def read_mono_wav(path: str, segment_index: int) -> WavData:
    """Read and validate a mono 16-bit PCM WAV, mapping failures to codes."""
    if not os.path.exists(path):
        raise ExporterError(
            ErrorCode.FILE_NOT_FOUND,
            f"segment {segment_index}: WAV not found: {path}",
            segment_index=segment_index,
            detail={"path": path},
        )
    try:
        blob, chunks = _read_riff_chunks(path)
    except ExporterError:
        raise
    except OSError as exc:  # pragma: no cover - defensive IO mapping
        raise ExporterError(
            ErrorCode.INVALID_WAV,
            f"{path}: unreadable WAV ({exc})",
            segment_index=segment_index,
            detail={"path": path},
        ) from exc

    if b"fmt " not in chunks or b"data" not in chunks:
        raise ExporterError(
            ErrorCode.INVALID_WAV,
            f"{path}: missing fmt/data chunk",
            segment_index=segment_index,
            detail={"path": path},
        )
    fmt_start, _ = chunks[b"fmt "][0]
    audio_format, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = (
        struct.unpack_from("<HHIIHH", blob, fmt_start)
    )
    if audio_format != 1:
        raise ExporterError(
            ErrorCode.INVALID_WAV,
            f"{path}: only PCM (format 1) supported, got {audio_format}",
            segment_index=segment_index,
            detail={"path": path, "audio_format": audio_format},
        )
    if channels != CHANNELS_MONO:
        raise ExporterError(
            ErrorCode.UNEXPECTED_CHANNELS,
            f"segment {segment_index}: source must be mono, got {channels} channel(s)",
            segment_index=segment_index,
            detail={"path": path, "channels": channels},
        )
    if bits_per_sample != 16:
        raise ExporterError(
            ErrorCode.UNEXPECTED_SAMPLE_WIDTH,
            f"segment {segment_index}: source must be 16-bit, got {bits_per_sample}",
            segment_index=segment_index,
            detail={"path": path, "bits_per_sample": bits_per_sample},
        )

    data_start, data_size = chunks[b"data"][0]
    if data_size % 2 != 0:
        offset_ms = (data_size // 2) * 1000.0 / sample_rate
        raise ExporterError(
            ErrorCode.NON_INTEGER_FRAME,
            f"segment {segment_index}: data chunk holds {data_size} bytes, "
            "not a whole number of 16-bit frames",
            segment_index=segment_index,
            offset_ms=round(offset_ms, 4),
            detail={"path": path, "data_bytes": data_size},
        )

    frame_count = data_size // 2
    frames = list(
        struct.unpack_from(f"<{frame_count}h", blob, data_start)
    ) if frame_count else []
    return WavData(path=path, sample_rate=sample_rate, frames=frames)


def assert_master_sample_rate(voice: WavData, ambience: WavData, segment_index: int) -> None:
    for clip in (voice, ambience):
        if clip.sample_rate != SAMPLE_RATE:
            raise ExporterError(
                ErrorCode.SAMPLE_RATE_MISMATCH,
                f"segment {segment_index}: sample rate {clip.sample_rate} Hz "
                f"does not match master rate {SAMPLE_RATE} Hz ({clip.path})",
                segment_index=segment_index,
                offset_ms=0.0,
                detail={
                    "path": clip.path,
                    "actual_rate": clip.sample_rate,
                    "expected_rate": SAMPLE_RATE,
                },
            )
    if voice.sample_rate != ambience.sample_rate:
        raise ExporterError(
            ErrorCode.SAMPLE_RATE_MISMATCH,
            f"segment {segment_index}: voice rate {voice.sample_rate} Hz != "
            f"ambience rate {ambience.sample_rate} Hz",
            segment_index=segment_index,
            offset_ms=0.0,
            detail={
                "voice_rate": voice.sample_rate,
                "ambience_rate": ambience.sample_rate,
            },
        )


def write_mono_wav(path: str, frames: List[int], sample_rate: int = SAMPLE_RATE) -> None:
    """Write a deterministic mono 16-bit PCM WAV (used by the sample generator)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(CHANNELS_MONO)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(frames)}h", *frames))


class StereoMasterWriter:
    """Stereo 16-bit PCM master file supporting random-access block writes.

    The WAV header is written up front sized to the full planned duration so
    that resume can simply seek to a block offset and overwrite bytes. The
    data-size field is therefore valid even while a render is in progress.
    """

    HEADER_SIZE = 44

    def __init__(self, path: str, total_frames: int, *, create_if_missing: bool = True) -> None:
        self.path = path
        self.total_frames = total_frames
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data_bytes = total_frames * CHANNELS_STEREO * SAMPLE_WIDTH
        expected_size = self.HEADER_SIZE + data_bytes
        # On resume an existing, correctly sized master must be preserved so the
        # checkpoint verification can observe torn/half-written blocks. A
        # truncated/missing container is rebuilt only after reconciliation has
        # decided which blocks must be recomputed.
        if os.path.exists(path):
            if os.path.getsize(path) != expected_size and create_if_missing:
                self.initialize_container()
            return
        if create_if_missing:
            self.initialize_container()

    def initialize_container(self) -> None:
        """Create (or recreate) the zero-filled, header-sized master file."""
        data_bytes = self.total_frames * CHANNELS_STEREO * SAMPLE_WIDTH
        expected_size = self.HEADER_SIZE + data_bytes
        if os.path.exists(self.path) and os.path.getsize(self.path) == expected_size:
            return
        header = b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE"
        header += b"fmt " + struct.pack(
            "<IHHIIHH",
            16,
            1,
            CHANNELS_STEREO,
            SAMPLE_RATE,
            SAMPLE_RATE * CHANNELS_STEREO * SAMPLE_WIDTH,
            CHANNELS_STEREO * SAMPLE_WIDTH,
            16,
        )
        header += b"data" + struct.pack("<I", data_bytes)
        with open(self.path, "wb") as handle:
            handle.write(header)
            handle.write(b"\x00" * data_bytes)

    def block_byte_offset(self, block_index: int, block_frames: int) -> int:
        return self.HEADER_SIZE + block_index * block_frames * CHANNELS_STEREO * SAMPLE_WIDTH

    def write_block(self, block_index: int, block_frames: int, pcm: bytes) -> None:
        with open(self.path, "r+b") as handle:
            handle.seek(self.block_byte_offset(block_index, block_frames))
            handle.write(pcm)

    def read_block(self, block_index: int, block_frames: int, frame_count: int) -> bytes:
        byte_count = frame_count * CHANNELS_STEREO * SAMPLE_WIDTH
        with open(self.path, "rb") as handle:
            handle.seek(self.block_byte_offset(block_index, block_frames))
            return handle.read(byte_count)

    def data_length_bytes(self) -> int:
        return self.total_frames * CHANNELS_STEREO * SAMPLE_WIDTH
