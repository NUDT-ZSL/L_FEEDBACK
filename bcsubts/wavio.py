from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import BusinessError


SUPPORTED_RATES = (44100, 48000, 96000)


@dataclass
class WavData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    channel_layout: str
    nframes: int
    source_format: str


def _inspect_riff(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise BusinessError("E_FORMAT", "wav_decode", "Input is not a RIFF/WAVE file")
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                break
            chunk_id = chunk[:4]
            chunk_size = struct.unpack("<I", chunk[4:])[0]
            if chunk_id == b"fmt ":
                fmt = handle.read(min(chunk_size, 40))
                if len(fmt) < 16:
                    raise BusinessError("E_FORMAT", "wav_decode", "Truncated WAV fmt chunk")
                (
                    audio_format,
                    channels,
                    sample_rate,
                    _byte_rate,
                    _block_align,
                    bits_per_sample,
                ) = struct.unpack("<HHIIHH", fmt[:16])
                result.update(
                    audio_format=audio_format,
                    channels=channels,
                    sample_rate=sample_rate,
                    bits_per_sample=bits_per_sample,
                )
                if audio_format == 0xFFFE and len(fmt) >= 26:
                    result["subformat"] = struct.unpack("<H", fmt[24:26])[0]
                handle.seek(chunk_size - min(chunk_size, 40), 1)
            else:
                handle.seek(chunk_size, 1)
            if chunk_size & 1:
                handle.seek(1, 1)
    return result


def read_wav16(path: str | Path) -> WavData:
    path = Path(path)
    if not path.is_file():
        raise BusinessError("E_INPUT", "wav_decode", "WAV file not found", {"path": str(path)})
    info = _inspect_riff(path)
    audio_format = info.get("audio_format")
    effective_format = info.get("subformat", audio_format)
    bits = info.get("bits_per_sample")
    channels = info.get("channels")
    rate = info.get("sample_rate")

    if audio_format == 3 or effective_format == 3:
        raise BusinessError(
            "E_FORMAT",
            "wav_decode",
            "Floating-point WAV is not supported; a 16-bit PCM WAV is required",
            {"audio_format": audio_format},
        )
    if effective_format not in (1, None):
        raise BusinessError(
            "E_FORMAT",
            "wav_decode",
            "Unsupported WAV codec; only 16-bit PCM is supported",
            {"audio_format": audio_format, "subformat": effective_format},
        )
    if bits != 16:
        raise BusinessError(
            "E_FORMAT",
            "wav_decode",
            "Unsupported WAV bit depth; only 16-bit PCM is supported",
            {"bits_per_sample": bits},
        )
    if channels not in (1, 2):
        raise BusinessError(
            "E_FORMAT",
            "wav_decode",
            "Only mono or stereo WAV is supported",
            {"channels": channels},
        )
    if rate not in SUPPORTED_RATES:
        raise BusinessError(
            "E_FORMAT",
            "wav_decode",
            "Only 44.1 kHz, 48 kHz and 96 kHz WAV are supported",
            {"sample_rate": rate},
        )

    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
                raise BusinessError(
                    "E_FORMAT",
                    "wav_decode",
                    "Compressed WAV is not supported",
                )
            nch = wav.getnchannels()
            sr = wav.getframerate()
            nframes = wav.getnframes()
            raw = wav.readframes(nframes)
    except wave.Error as exc:
        raise BusinessError("E_FORMAT", "wav_decode", f"Invalid WAV: {exc}") from exc

    if nch != channels or sr != rate:
        raise BusinessError(
            "E_FORMAT",
            "wav_decode",
            "WAV header and payload disagree",
            {"header_channels": channels, "payload_channels": nch},
        )
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size != nframes * nch:
        raise BusinessError("E_FORMAT", "wav_decode", "Truncated WAV sample payload")
    samples = samples.reshape(-1, nch).astype(np.float64) / 32768.0
    layout = "mono" if nch == 1 else "stereo (FL+FR)"
    return WavData(
        samples=samples,
        sample_rate=sr,
        channels=nch,
        channel_layout=layout,
        nframes=nframes,
        source_format="PCM_S16LE",
    )


def write_wav16(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    pcm = np.clip(np.rint(array * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(array.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
