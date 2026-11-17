"""WAV probing, bit-depth policy and streaming audio I/O."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .errors import (
    BusinessError,
    E_BIT_DEPTH,
    E_INPUT_FORMAT,
    E_INPUT_READ,
    E_IO,
    E_OUTPUT_FORMAT,
    ProcessingInterrupted,
)


@dataclass(frozen=True)
class SubtypeInfo:
    subtype: str
    label: str
    container_bits: int
    precision_bits: int
    floating: bool


@dataclass(frozen=True)
class OutputFormat:
    name: str
    subtype: str
    container_bits: int
    precision_bits: int
    floating: bool


INPUT_SUBTYPES = {
    "PCM_U8": SubtypeInfo("PCM_U8", "pcm8", 8, 8, False),
    "PCM_16": SubtypeInfo("PCM_16", "pcm16", 16, 16, False),
    "PCM_24": SubtypeInfo("PCM_24", "pcm24", 24, 24, False),
    "PCM_32": SubtypeInfo("PCM_32", "pcm32", 32, 32, False),
    "FLOAT": SubtypeInfo("FLOAT", "float32", 32, 24, True),
    "DOUBLE": SubtypeInfo("DOUBLE", "float64", 64, 53, True),
}

OUTPUT_FORMATS = {
    "pcm8": OutputFormat("pcm8", "PCM_U8", 8, 8, False),
    "wav8": OutputFormat("pcm8", "PCM_U8", 8, 8, False),
    "pcm_u8": OutputFormat("pcm8", "PCM_U8", 8, 8, False),
    "u8": OutputFormat("pcm8", "PCM_U8", 8, 8, False),
    "8": OutputFormat("pcm8", "PCM_U8", 8, 8, False),
    "pcm16": OutputFormat("pcm16", "PCM_16", 16, 16, False),
    "wav16": OutputFormat("pcm16", "PCM_16", 16, 16, False),
    "pcm_16": OutputFormat("pcm16", "PCM_16", 16, 16, False),
    "16": OutputFormat("pcm16", "PCM_16", 16, 16, False),
    "pcm24": OutputFormat("pcm24", "PCM_24", 24, 24, False),
    "wav24": OutputFormat("pcm24", "PCM_24", 24, 24, False),
    "pcm_24": OutputFormat("pcm24", "PCM_24", 24, 24, False),
    "24": OutputFormat("pcm24", "PCM_24", 24, 24, False),
    "pcm32": OutputFormat("pcm32", "PCM_32", 32, 32, False),
    "wav32": OutputFormat("pcm32", "PCM_32", 32, 32, False),
    "pcm_32": OutputFormat("pcm32", "PCM_32", 32, 32, False),
    "32": OutputFormat("pcm32", "PCM_32", 32, 32, False),
    "float": OutputFormat("float32", "FLOAT", 32, 24, True),
    "float32": OutputFormat("float32", "FLOAT", 32, 24, True),
    "f32": OutputFormat("float32", "FLOAT", 32, 24, True),
    "float_32": OutputFormat("float32", "FLOAT", 32, 24, True),
    "ieee_float": OutputFormat("float32", "FLOAT", 32, 24, True),
    "ieee_float32": OutputFormat("float32", "FLOAT", 32, 24, True),
    "float64": OutputFormat("float64", "DOUBLE", 64, 53, True),
    "f64": OutputFormat("float64", "DOUBLE", 64, 53, True),
    "float_64": OutputFormat("float64", "DOUBLE", 64, 53, True),
    "ieee_float64": OutputFormat("float64", "DOUBLE", 64, 53, True),
    "double": OutputFormat("float64", "DOUBLE", 64, 53, True),
    "double64": OutputFormat("float64", "DOUBLE", 64, 53, True),
}


@dataclass(frozen=True)
class AudioInfo:
    path: str
    frames: int
    channels: int
    sample_rate: int
    subtype: str
    wav_format: str
    container_bits: int
    decoded_sample_bytes: int
    sha256: str


def parse_output_format(value: str) -> OutputFormat:
    key = value.strip().lower()
    try:
        return OUTPUT_FORMATS[key]
    except KeyError as exc:
        raise BusinessError(
            E_OUTPUT_FORMAT,
            f"Unsupported output format: {value}",
            {"output_format": value, "supported": sorted(set(OUTPUT_FORMATS))},
        ) from exc


def validate_bit_depth(input_info: SubtypeInfo, output: OutputFormat) -> None:
    if input_info.floating and not output.floating:
        raise BusinessError(
            E_BIT_DEPTH,
            "Output format is not compatible with input sample representation",
            {
                "input_subtype": input_info.subtype,
                "output_subtype": output.subtype,
                "rule": "floating input requires floating output",
            },
        )
    if output.precision_bits < input_info.precision_bits:
        raise BusinessError(
            E_BIT_DEPTH,
            "Output precision is lower than input precision",
            {
                "input_subtype": input_info.subtype,
                "input_precision_bits": input_info.precision_bits,
                "output_subtype": output.subtype,
                "output_precision_bits": output.precision_bits,
            },
        )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise BusinessError(
            E_IO,
            f"Unable to read file for hashing: {path}",
            {"path": str(path), "reason": str(exc)},
        ) from exc
    return digest.hexdigest()


def probe_audio(path: str | Path, hash_content: bool = True) -> AudioInfo:
    input_path = Path(path)
    if not input_path.is_file():
        raise BusinessError(
            E_INPUT_FORMAT,
            f"Input WAV does not exist: {input_path}",
            {"path": str(input_path)},
        )
    try:
        info = sf.info(str(input_path))
    except Exception as exc:
        raise BusinessError(
            E_INPUT_FORMAT,
            f"Unsupported or invalid input audio: {input_path}",
            {"path": str(input_path), "reason": str(exc)},
        ) from exc

    if info.format not in {"WAV", "WAVEX"}:
        raise BusinessError(
            E_INPUT_FORMAT,
            "Input must be a WAV file",
            {"path": str(input_path), "detected_format": info.format},
        )
    if info.subtype not in INPUT_SUBTYPES:
        raise BusinessError(
            E_INPUT_FORMAT,
            "Unsupported WAV sample subtype",
            {
                "path": str(input_path),
                "subtype": info.subtype,
                "supported": sorted(INPUT_SUBTYPES),
            },
        )
    subtype_info = INPUT_SUBTYPES[info.subtype]
    digest = sha256_file(input_path) if hash_content else ""
    return AudioInfo(
        path=str(input_path.resolve()),
        frames=int(info.frames),
        channels=int(info.channels),
        sample_rate=int(info.samplerate),
        subtype=info.subtype,
        wav_format=info.format,
        container_bits=subtype_info.container_bits,
        decoded_sample_bytes=subtype_info.container_bits // 8,
        sha256=digest,
    )


def open_input(path: str | Path) -> sf.SoundFile:
    try:
        return sf.SoundFile(str(path), mode="r")
    except Exception as exc:
        raise BusinessError(
            E_INPUT_READ,
            f"Unable to open input WAV: {path}",
            {"path": str(path), "reason": str(exc)},
        ) from exc


def read_source_span(
    handle: sf.SoundFile,
    channels: Tuple[int, ...],
    first_frame: int,
    frame_count: int,
    total_frames: int,
) -> np.ndarray:
    """Read a mono source span while zero-padding outside the input file."""

    if frame_count <= 0:
        return np.zeros(0, dtype=np.float64)
    output = np.zeros(frame_count, dtype=np.float64)
    first = int(first_frame)
    requested_end = first + frame_count
    valid_first = max(0, first)
    valid_end = min(total_frames, requested_end)
    if valid_end <= valid_first:
        return output
    try:
        handle.seek(valid_first)
        data = handle.read(valid_end - valid_first, dtype="float64", always_2d=True)
    except Exception as exc:
        raise BusinessError(
            E_INPUT_READ,
            "Unable to read an input audio block",
            {
                "first_frame": valid_first,
                "frames": valid_end - valid_first,
                "reason": str(exc),
            },
        ) from exc
    selected = data[:, list(channels)]
    mono = np.mean(selected, axis=1, dtype=np.float64)
    destination_start = valid_first - first
    output[destination_start : destination_start + mono.size] = mono
    return output


def open_output_temp(path: str | Path, sample_rate: int, subtype: str) -> sf.SoundFile:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return sf.SoundFile(
            str(path),
            mode="w",
            samplerate=sample_rate,
            channels=2,
            format="WAV",
            subtype=subtype,
        )
    except Exception as exc:
        raise BusinessError(
            E_IO,
            f"Unable to create temporary output WAV: {path}",
            {"path": str(path), "reason": str(exc)},
        ) from exc


def encode_raw_to_wav(
    raw_path: str | Path,
    wav_path: str | Path,
    sample_rate: int,
    subtype: str,
    block_frames: int,
    sample_dtype: np.dtype | type = np.float32,
) -> int:
    """Encode interleaved stereo floating-point raw samples into a WAV file."""

    total_frames = 0
    dtype = np.dtype(sample_dtype)
    output = open_output_temp(wav_path, sample_rate, subtype)
    try:
        with open(raw_path, "rb") as raw:
            while True:
                byte_count = block_frames * 2 * dtype.itemsize
                payload = raw.read(byte_count)
                if not payload:
                    break
                if len(payload) % (2 * dtype.itemsize):
                    raise BusinessError(
                        E_IO,
                        "Temporary mix file is truncated",
                        {"path": str(raw_path)},
                    )
                sample_data = np.frombuffer(payload, dtype=dtype).reshape(-1, 2)
                sample_data = np.clip(sample_data.astype(np.float64), -1.0, 1.0)
                output.write(sample_data)
                total_frames += sample_data.shape[0]
        output.flush()
    except BusinessError:
        raise
    except ProcessingInterrupted:
        raise
    except Exception as exc:
        raise BusinessError(
            E_IO,
            f"Unable to write output WAV: {wav_path}",
            {"path": str(wav_path), "reason": str(exc)},
        ) from exc
    finally:
        output.close()
    return total_frames


def atomic_replace(source: str | Path, destination: str | Path) -> None:
    try:
        os.replace(str(source), str(destination))
    except OSError as exc:
        raise BusinessError(
            E_IO,
            f"Unable to replace output file: {destination}",
            {
                "source": str(source),
                "destination": str(destination),
                "reason": str(exc),
            },
        ) from exc
