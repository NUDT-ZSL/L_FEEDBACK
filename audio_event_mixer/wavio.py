"""WAV I/O for integer PCM and IEEE float, with streaming and atomic output."""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Union

import numpy as np

from .errors import BusinessError, ErrorCode

_PCM = 1
_IEEE_FLOAT = 3
_EXTENSIBLE = 0xFFFE
_KSDATAFORMAT_SUFFIX = bytes.fromhex("00001000800000aa00389b71")


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    bits_per_sample: int
    format_tag: int
    encoding: str
    data_offset: int
    data_size: int
    block_align: int
    frames: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / float(self.sample_rate)

    def as_dict(self) -> dict:
        return {
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bits_per_sample,
            "frames": self.frames,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class WavData:
    samples: np.ndarray
    info: WavInfo


def _unsupported(message: str, path: Optional[Path] = None) -> BusinessError:
    details = {"path": str(path)} if path is not None else None
    return BusinessError(
        ErrorCode.UNSUPPORTED_FORMAT, message, details=details
    )


def _not_found(path: Path, message: str = "audio file does not exist") -> BusinessError:
    return BusinessError(
        ErrorCode.FILE_NOT_FOUND,
        message,
        details={"path": str(path)},
    )


def inspect_wav(path: Union[str, os.PathLike[str]]) -> WavInfo:
    """Parse the RIFF header without loading sample data."""

    wav_path = Path(path).expanduser()
    if not wav_path.is_file():
        raise _not_found(wav_path)

    try:
        file_size = wav_path.stat().st_size
        with wav_path.open("rb") as fh:
            header = fh.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise _unsupported("not a RIFF/WAVE file", wav_path)

            fmt_chunk: Optional[bytes] = None
            data_offset: Optional[int] = None
            data_size: Optional[int] = None
            pos = 12
            while pos + 8 <= file_size:
                fh.seek(pos)
                chunk_header = fh.read(8)
                if len(chunk_header) != 8:
                    break
                chunk_id = chunk_header[:4]
                (chunk_size,) = struct.unpack("<I", chunk_header[4:8])
                chunk_data_pos = pos + 8
                if chunk_id == b"fmt " and fmt_chunk is None:
                    available = max(0, min(chunk_size, file_size - chunk_data_pos))
                    fmt_chunk = fh.read(available)
                elif chunk_id == b"data" and data_offset is None:
                    data_offset = chunk_data_pos
                    available = max(0, min(chunk_size, file_size - chunk_data_pos))
                    data_size = available
                pos = chunk_data_pos + chunk_size + (chunk_size & 1)
                if fmt_chunk is not None and data_offset is not None:
                    break
    except BusinessError:
        raise
    except OSError as exc:
        raise BusinessError(
            ErrorCode.FILE_NOT_FOUND,
            f"cannot open audio file: {exc}",
            details={"path": str(wav_path)},
        ) from exc

    if fmt_chunk is None or data_offset is None or data_size is None:
        raise _unsupported("WAV is missing a fmt or data chunk", wav_path)
    if len(fmt_chunk) < 16:
        raise _unsupported("WAV fmt chunk is truncated", wav_path)

    (
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt_chunk, 0)

    if format_tag == _EXTENSIBLE:
        if len(fmt_chunk) < 40:
            raise _unsupported("truncated WAVE_FORMAT_EXTENSIBLE header", wav_path)
        cb_size = struct.unpack_from("<H", fmt_chunk, 16)[0]
        if cb_size < 22:
            raise _unsupported("invalid WAVE_FORMAT_EXTENSIBLE header", wav_path)
        valid_bits = struct.unpack_from("<H", fmt_chunk, 18)[0]
        subformat = fmt_chunk[24:40]
        if len(subformat) != 16 or subformat[4:] != _KSDATAFORMAT_SUFFIX:
            raise _unsupported("unknown WAVE_FORMAT_EXTENSIBLE subformat", wav_path)
        (format_tag,) = struct.unpack_from("<I", subformat, 0)
        if valid_bits not in (16, 24, 32) or valid_bits != bits_per_sample:
            raise _unsupported(
                f"unsupported extensible valid bits: {valid_bits}", wav_path
            )

    if channels <= 0 or sample_rate <= 0 or bits_per_sample <= 0:
        raise _unsupported("invalid channel count, sample rate, or bit depth", wav_path)

    if format_tag == _PCM:
        if bits_per_sample not in (16, 24, 32):
            raise _unsupported(
                f"unsupported PCM bit depth: {bits_per_sample}", wav_path
            )
        encoding = f"pcm_s{bits_per_sample}le"
    elif format_tag == _IEEE_FLOAT:
        if bits_per_sample != 32:
            raise _unsupported(
                f"unsupported float bit depth: {bits_per_sample}", wav_path
            )
        encoding = "float_s32le"
    else:
        raise _unsupported(
            f"unsupported WAV encoding (format tag {format_tag})", wav_path
        )

    expected_align = channels * (bits_per_sample // 8)
    if block_align != expected_align:
        raise _unsupported(
            f"invalid block alignment {block_align}; expected {expected_align}",
            wav_path,
        )
    if byte_rate != sample_rate * block_align:
        raise _unsupported(
            f"invalid byte rate {byte_rate}; expected {sample_rate * block_align}",
            wav_path,
        )
    if data_size % block_align != 0:
        raise _unsupported("data chunk is not sample-aligned", wav_path)

    return WavInfo(
        path=wav_path.resolve(),
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
        format_tag=format_tag,
        encoding=encoding,
        data_offset=data_offset,
        data_size=data_size,
        block_align=block_align,
        frames=data_size // block_align,
    )


def _decode_samples(raw: bytes, info: WavInfo) -> np.ndarray:
    if not raw:
        return np.empty((0, info.channels), dtype=np.float64)

    if info.encoding == "pcm_s16le":
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        values /= 32768.0
    elif info.encoding == "pcm_s24le":
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = (
            packed[:, 0]
            | (packed[:, 1] << 8)
            | (packed[:, 2] << 16)
        )
        values = (values ^ 0x800000) - 0x800000
        values = values.astype(np.float64) / 8388608.0
    elif info.encoding == "pcm_s32le":
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        values /= 2147483648.0
    elif info.encoding == "float_s32le":
        values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise _unsupported("float WAV contains NaN or infinity", info.path)
    else:
        raise _unsupported(f"unsupported encoding: {info.encoding}", info.path)

    frame_count = values.size // info.channels
    return values.reshape(frame_count, info.channels)


class WavReader:
    """Seekable, chunked WAV reader.

    The standard-library wave module handles ordinary integer PCM files.
    IEEE-float and WAVE_FORMAT_EXTENSIBLE files are read through the same
    validated RIFF metadata using direct little-endian NumPy decoding.
    """

    def __init__(self, path: Union[str, os.PathLike[str]]):
        self.info = inspect_wav(path)
        self._wave: Optional[wave.Wave_read] = None
        self._fh: Optional[BinaryIO] = None

        if self.info.format_tag == _PCM:
            try:
                candidate = wave.open(str(self.info.path), "rb")
                valid = (
                    candidate.getcomptype() == "NONE"
                    and candidate.getnchannels() == self.info.channels
                    and candidate.getframerate() == self.info.sample_rate
                    and candidate.getsampwidth() * 8 == self.info.bits_per_sample
                    and candidate.getnframes() == self.info.frames
                )
                if valid:
                    self._wave = candidate
                else:
                    candidate.close()
            except (wave.Error, EOFError, OSError):
                self._wave = None

        if self._wave is None:
            try:
                self._fh = self.info.path.open("rb")
            except OSError as exc:
                raise BusinessError(
                    ErrorCode.FILE_NOT_FOUND,
                    f"cannot open audio file: {exc}",
                    details={"path": str(self.info.path)},
                ) from exc

    @property
    def frames(self) -> int:
        return self.info.frames

    @property
    def channels(self) -> int:
        return self.info.channels

    def read_frames(self, start: int, count: int) -> np.ndarray:
        start = max(0, int(start))
        count = min(max(0, int(count)), max(0, self.frames - start))
        if count == 0:
            return np.empty((0, self.channels), dtype=np.float64)

        try:
            if self._wave is not None:
                self._wave.setpos(start)
                raw = self._wave.readframes(count)
            else:
                assert self._fh is not None
                self._fh.seek(self.info.data_offset + start * self.info.block_align)
                raw = self._fh.read(count * self.info.block_align)
        except (OSError, wave.Error) as exc:
            raise _unsupported(f"cannot read WAV sample data: {exc}", self.info.path) from exc

        actual_frames = len(raw) // self.info.block_align
        raw = raw[: actual_frames * self.info.block_align]
        return _decode_samples(raw, self.info)

    def iter_frames(self, chunk_frames: int = 262144) -> Iterator[tuple[int, np.ndarray]]:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        position = 0
        while position < self.frames:
            chunk = self.read_frames(position, min(chunk_frames, self.frames - position))
            if chunk.shape[0] == 0:
                break
            yield position, chunk
            position += chunk.shape[0]

    def read_all(self) -> np.ndarray:
        chunks = [chunk for _, chunk in self.iter_frames()]
        if not chunks:
            return np.empty((0, self.channels), dtype=np.float64)
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        if self._wave is not None:
            self._wave.close()
            self._wave = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "WavReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_wav(path: Union[str, os.PathLike[str]]) -> WavData:
    """Read a supported WAV into a float64 array.

    Mono files use shape ``(frames,)``. Files with two or more channels use
    shape ``(frames, channels)``. Integer samples are scaled to [-1, 1).
    """

    with WavReader(path) as reader:
        samples = reader.read_all()
        if reader.channels == 1:
            samples = samples[:, 0]
        return WavData(samples=samples, info=reader.info)


def downmix_chunk(chunk: np.ndarray) -> np.ndarray:
    """Convert a decoded chunk to mono float64 without changing frame count."""

    if chunk.ndim == 1:
        return chunk.astype(np.float64, copy=False)
    if chunk.shape[1] == 1:
        return chunk[:, 0].astype(np.float64, copy=False)
    return chunk.mean(axis=1, dtype=np.float64)


def _atomic_temp_path(destination: Path) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        os.close(fd)
        return Path(name)
    except OSError as exc:
        raise BusinessError(
            ErrorCode.OUTPUT_DENIED,
            f"cannot create temporary output in {destination.parent}: {exc}",
            details={"path": str(destination)},
        ) from exc


def atomic_write_json(path: Union[str, os.PathLike[str]], payload: dict) -> None:
    import json

    destination = Path(path).expanduser()
    temp_path = _atomic_temp_path(destination)
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, destination)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise BusinessError(
            ErrorCode.OUTPUT_DENIED,
            f"cannot write report {destination}: {exc}",
            details={"path": str(destination)},
        ) from exc


def _quantize(chunk: np.ndarray, bit_depth: int) -> bytes:
    clipped = np.clip(np.ascontiguousarray(chunk), -1.0, 1.0).reshape(-1)
    if bit_depth == 16:
        quantized = np.floor(np.abs(clipped) * 32767.0 + 0.5)
        quantized = np.copysign(quantized, clipped).astype("<i2")
        return quantized.tobytes(order="C")
    if bit_depth == 24:
        quantized = np.floor(np.abs(clipped) * 8388607.0 + 0.5)
        quantized = np.copysign(quantized, clipped).astype(np.int32)
        unsigned = (quantized & 0xFFFFFF).astype(np.int32)
        packed = np.empty((unsigned.shape[0], 3), dtype=np.uint8)
        packed[:, 0] = unsigned & 0xFF
        packed[:, 1] = (unsigned >> 8) & 0xFF
        packed[:, 2] = (unsigned >> 16) & 0xFF
        return packed.tobytes(order="C")
    if bit_depth == 32:
        quantized = np.floor(np.abs(clipped) * 2147483647.0 + 0.5)
        quantized = np.copysign(quantized, clipped).astype("<i4")
        return quantized.tobytes(order="C")
    raise BusinessError(
        ErrorCode.UNSUPPORTED_FORMAT,
        f"unsupported output bit depth: {bit_depth}; use 16 or 24",
    )


class AtomicPcmWavWriter:
    """Write stereo 16- or 24-bit PCM through a same-directory temp file."""

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        sample_rate: int,
        bit_depth: int = 16,
    ):
        if bit_depth not in (16, 24):
            raise BusinessError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"unsupported output bit depth: {bit_depth}; use 16 or 24",
            )
        if sample_rate <= 0:
            raise BusinessError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"invalid output sample rate: {sample_rate}",
            )
        self.destination = Path(path).expanduser()
        self.sample_rate = int(sample_rate)
        self.bit_depth = int(bit_depth)
        self.temp_path: Optional[Path] = None
        self._wave: Optional[wave.Wave_write] = None
        self._committed = False
        self.frames_written = 0

    def __enter__(self) -> "AtomicPcmWavWriter":
        self.temp_path = _atomic_temp_path(self.destination)
        try:
            self._wave = wave.open(str(self.temp_path), "wb")
            self._wave.setnchannels(2)
            self._wave.setsampwidth(self.bit_depth // 8)
            self._wave.setframerate(self.sample_rate)
            self._wave.setcomptype("NONE", "not compressed")
        except (OSError, wave.Error) as exc:
            if self._wave is not None:
                self._wave.close()
                self._wave = None
            self.temp_path.unlink(missing_ok=True)
            self.temp_path = None
            raise BusinessError(
                ErrorCode.OUTPUT_DENIED,
                f"cannot open output WAV: {exc}",
                details={"path": str(self.destination)},
            ) from exc
        return self

    def write(self, chunk: np.ndarray) -> None:
        if self._wave is None:
            raise BusinessError(ErrorCode.OUTPUT_DENIED, "output WAV is not open")
        array = np.asarray(chunk, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise BusinessError(
                ErrorCode.OUTPUT_DENIED,
                "output chunks must have shape (frames, 2)",
            )
        raw = _quantize(array, self.bit_depth)
        try:
            self._wave.writeframesraw(raw)
        except (OSError, wave.Error) as exc:
            raise BusinessError(
                ErrorCode.OUTPUT_DENIED,
                f"cannot write output WAV: {exc}",
                details={"path": str(self.destination)},
            ) from exc
        self.frames_written += array.shape[0]

    def commit(self) -> None:
        if self._committed:
            return
        if self._wave is None or self.temp_path is None:
            raise BusinessError(ErrorCode.OUTPUT_DENIED, "output WAV is not open")
        try:
            self._wave.close()
            self._wave = None
            with self.temp_path.open("rb+") as fh:
                os.fsync(fh.fileno())
            os.replace(self.temp_path, self.destination)
            self.temp_path = None
            self._committed = True
        except OSError as exc:
            self.abort()
            raise BusinessError(
                ErrorCode.OUTPUT_DENIED,
                f"cannot finalize output WAV: {exc}",
                details={"path": str(self.destination)},
            ) from exc

    def abort(self) -> None:
        if self._wave is not None:
            try:
                self._wave.close()
            except (OSError, wave.Error):
                pass
            self._wave = None
        if self.temp_path is not None:
            try:
                self.temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.temp_path = None

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self._committed:
            self.commit()
        elif not self._committed:
            self.abort()


def write_wav_float32(
    path: Union[str, os.PathLike[str]],
    samples: np.ndarray,
    sample_rate: int,
) -> WavInfo:
    """Write deterministic 32-bit IEEE-float WAV (used by the sample generator)."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("samples must have shape (frames,) or (frames, channels)")

    channels = int(array.shape[1])
    interleaved = np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")
    fmt = struct.pack(
        "<HHIIHH",
        _IEEE_FLOAT,
        channels,
        sample_rate,
        sample_rate * channels * 4,
        channels * 4,
        32,
    )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(interleaved)) + interleaved
    if len(interleaved) & 1:
        data_chunk += b"\x00"
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    payload = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return inspect_wav(destination)


def write_pcm_wav(
    path: Union[str, os.PathLike[str]],
    samples: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
) -> WavInfo:
    """Convenience non-atomic PCM writer for fixtures and tests."""

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("samples must have shape (frames,) or (frames, channels)")
    channels = array.shape[1]
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(sample_rate)
        wf.setcomptype("NONE", "not compressed")
        wf.writeframes(_quantize(array, bit_depth))
    return inspect_wav(destination)
