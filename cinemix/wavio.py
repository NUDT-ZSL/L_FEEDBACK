from __future__ import annotations

import os
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CinemixError


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    bit_depth: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def _read_frames(raw: bytes, channels: int, bit_depth: int) -> np.ndarray:
    if bit_depth == 16:
        values = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        scale = 32768.0
    elif bit_depth == 24:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if octets.size % 3:
            raise CinemixError(
                "E_INPUT_FORMAT", "24-bit PCM payload is not a multiple of 3 bytes"
            )
        triples = octets.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        scale = 8388608.0
    else:
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"Unsupported PCM bit depth {bit_depth}; expected 16 or 24",
        )

    if channels <= 0:
        raise CinemixError("E_INPUT_FORMAT", "WAV declares zero channels")
    if values.size % channels:
        raise CinemixError(
            "E_INPUT_FORMAT", "PCM payload does not contain complete frames"
        )
    return (values.reshape(-1, channels).astype(np.float64) / scale)


def read_pcm_wav(path: str | Path) -> tuple[np.ndarray, WavInfo]:
    wav_path = Path(path)
    if not wav_path.is_file():
        raise CinemixError(
            "E_INPUT_MISSING",
            f"Input WAV does not exist: {wav_path}",
            context={"path": str(wav_path)},
        )

    try:
        with wave.open(str(wav_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            bit_depth = wav.getsampwidth() * 8
            frames = wav.getnframes()
            compression = wav.getcomptype()
            raw = wav.readframes(frames)
    except (wave.Error, EOFError, OSError) as exc:
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"Cannot read PCM WAV {wav_path}: {exc}",
            context={"path": str(wav_path)},
        ) from exc

    if compression != "NONE":
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"Compressed WAV is not supported: {wav_path}",
            context={"path": str(wav_path), "compression": compression},
        )
    if bit_depth not in (16, 24):
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"Unsupported WAV sample width {bit_depth}-bit in {wav_path}",
            context={"path": str(wav_path), "bit_depth": bit_depth},
        )
    if channels not in (1, 2):
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"Only mono or stereo tracks are supported, got {channels} channels",
            context={"path": str(wav_path), "channels": channels},
        )
    if sample_rate < 8000 or sample_rate > 384000:
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"Implausible WAV sample rate {sample_rate} in {wav_path}",
            context={"path": str(wav_path), "sample_rate": sample_rate},
        )

    samples = _read_frames(raw, channels, bit_depth)
    if frames and samples.shape[0] != frames:
        raise CinemixError(
            "E_INPUT_FORMAT",
            "WAV header frame count does not match PCM payload",
            context={"path": str(wav_path)},
        )
    if not np.all(np.isfinite(samples)):
        raise CinemixError(
            "E_INPUT_FORMAT",
            f"WAV contains non-finite samples: {wav_path}",
            context={"path": str(wav_path)},
        )

    info = WavInfo(
        path=wav_path,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        frames=frames,
    )
    return samples, info


def quantize_pcm24(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("samples must have shape (frames, channels)")
    scaled = np.asarray(samples, dtype=np.float64) * 8388608.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(rounded, -8388608.0, 8388607.0).astype(np.int32)


def dequantize_pcm24(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float64) / 8388608.0


def _encode_pcm24(values: np.ndarray) -> bytes:
    unsigned = values.astype(np.int32).reshape(-1).view(np.uint32)
    octets = np.empty((unsigned.size, 3), dtype=np.uint8)
    octets[:, 0] = unsigned & 0xFF
    octets[:, 1] = (unsigned >> 8) & 0xFF
    octets[:, 2] = (unsigned >> 16) & 0xFF
    return octets.tobytes()


def write_pcm_wav(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    bit_depth: int = 24,
) -> None:
    wav_path = Path(path)
    if samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise ValueError("samples must have shape (frames, 1 or 2)")
    if bit_depth != 24:
        raise ValueError("Cinemix output is fixed at 24-bit PCM")

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{wav_path.name}.", suffix=".tmp", dir=wav_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        quantized = quantize_pcm24(samples)
        raw = _encode_pcm24(quantized)
        with wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(samples.shape[1])
            wav.setsampwidth(3)
            wav.setframerate(sample_rate)
            wav.setcomptype("NONE", "not compressed")
            wav.writeframes(raw)
        os.replace(temporary, wav_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_pcm16_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write deterministic 16-bit PCM, used only by the sample generator."""
    wav_path = Path(path)
    if samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise ValueError("samples must have shape (frames, 1 or 2)")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.asarray(samples, dtype=np.float64) * 32768.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    values = np.clip(rounded, -32768.0, 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.setcomptype("NONE", "not compressed")
        wav.writeframes(values.tobytes())


def wav_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def riff_info(path: str | Path) -> dict[str, int | str]:
    """Read only immutable RIFF fields used for deterministic verification."""
    with open(path, "rb") as handle:
        header = handle.read(64)
    if len(header) < 44 or header[:4] not in (b"RIFF", b"RIFX") or header[8:12] != b"WAVE":
        raise CinemixError("E_INPUT_FORMAT", f"Not a RIFF/WAVE file: {path}")
    fmt_tag = struct.unpack("<H", header[20:22])[0]
    channels = struct.unpack("<H", header[22:24])[0]
    sample_rate = struct.unpack("<I", header[24:28])[0]
    bits = struct.unpack("<H", header[34:36])[0]
    return {
        "format_tag": fmt_tag,
        "channels": channels,
        "sample_rate": sample_rate,
        "bit_depth": bits,
    }
