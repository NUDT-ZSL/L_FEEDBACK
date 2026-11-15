from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .errors import BakerError
from .util import atomic_write


LOUDNESS_BLOCK_SECONDS = 0.4
LOUDNESS_OVERLAP = 0.75
LOUDNESS_HOP_SECONDS = LOUDNESS_BLOCK_SECONDS * (1.0 - LOUDNESS_OVERLAP)
LOUDNESS_ABSOLUTE_GATE = -70.0
LOUDNESS_RELATIVE_GATE = -10.0


@dataclass
class DecodedDialogue:
    source_rate: int
    source_channels: int
    source_frames: int
    resampled_frames_raw: int
    resampled_frames_expected: int
    resample_length_error_raw: int
    target_frames: int
    length_adjust_samples: int
    resampled: bool
    samples: np.ndarray


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise BakerError(
            "E_FFMPEG_MISSING",
            "Required FFmpeg executable was not found in PATH.",
            {"executable": name},
        )
    return path


def _round_fraction_half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _probe_wav(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            compression = source.getcomptype()
    except (wave.Error, OSError):
        return _probe_wav_with_ffprobe(path)
    if compression != "NONE":
        return _probe_wav_with_ffprobe(path)
    if channels <= 0 or sample_rate <= 0:
        raise BakerError(
            "E_AUDIO_DECODE",
            "WAV has invalid channel count or sample rate.",
            {"path": str(path), "channels": channels, "sample_rate": sample_rate},
        )
    return sample_rate, channels, frames


def _probe_wav_with_ffprobe(path: Path) -> tuple[int, int, int]:
    ffprobe = _require_executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise BakerError(
            "E_AUDIO_DECODE",
            "FFprobe could not read the dry dialogue WAV.",
            {
                "path": str(path),
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-2000:],
            },
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        stream = payload["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        raw_frames = stream.get("nb_frames")
        if raw_frames is not None and int(raw_frames) > 0:
            frames = int(raw_frames)
        else:
            duration = float(stream["duration"])
            frames = int(round(duration * sample_rate))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise BakerError(
            "E_AUDIO_DECODE",
            "FFprobe returned malformed dry-dialogue stream information.",
            {"path": str(path)},
        ) from exc
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise BakerError(
            "E_AUDIO_DECODE",
            "Dry dialogue WAV has invalid duration, channels, or sample rate.",
            {
                "path": str(path),
                "frames": frames,
                "channels": channels,
                "sample_rate": sample_rate,
            },
        )
    return sample_rate, channels, frames


def decode_dialogue(path: Path, target_frames: int) -> DecodedDialogue:
    source_rate, source_channels, source_frames = _probe_wav(path)
    ffmpeg = _require_executable("ffmpeg")
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise BakerError(
            "E_AUDIO_DECODE",
            "FFmpeg failed to decode dry dialogue.",
            {
                "path": str(path),
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-2000:],
            },
        )
    if len(completed.stdout) % 4:
        raise BakerError(
            "E_AUDIO_DECODE",
            "FFmpeg returned a truncated float32 stream.",
            {"path": str(path), "byte_count": len(completed.stdout)},
        )
    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    if not np.all(np.isfinite(samples)):
        raise BakerError(
            "E_AUDIO_DECODE",
            "Decoded dialogue contains non-finite samples.",
            {"path": str(path)},
        )
    expected_frames = _round_fraction_half_up(
        Fraction(source_frames * 48000, source_rate)
    )
    raw_frames = int(samples.size)
    raw_error = raw_frames - expected_frames
    if raw_frames < expected_frames:
        samples = np.pad(samples, (0, expected_frames - raw_frames))
    elif raw_frames > expected_frames:
        samples = samples[:expected_frames]
    if expected_frames < target_frames:
        aligned = np.pad(samples, (0, target_frames - expected_frames))
        length_adjust = target_frames - expected_frames
    elif expected_frames > target_frames:
        aligned = samples[:target_frames]
        length_adjust = target_frames - expected_frames
    else:
        aligned = samples.copy()
        length_adjust = 0
    return DecodedDialogue(
        source_rate=source_rate,
        source_channels=source_channels,
        source_frames=source_frames,
        resampled_frames_raw=raw_frames,
        resampled_frames_expected=expected_frames,
        resample_length_error_raw=raw_error,
        target_frames=target_frames,
        length_adjust_samples=length_adjust,
        resampled=source_rate != 48000,
        samples=np.ascontiguousarray(aligned, dtype=np.float64),
    )


def linear_to_db(value: float) -> Optional[float]:
    if value <= 0.0:
        return None
    return 20.0 * math.log10(value)


def db_to_linear(value: float) -> float:
    return 10.0 ** (value / 20.0)


def peak_dbfs(samples: np.ndarray) -> Optional[float]:
    if samples.size == 0:
        return None
    return linear_to_db(float(np.max(np.abs(samples))))


def clipped_sample_count(samples: np.ndarray, ceiling: float = 1.0) -> int:
    if samples.size == 0:
        return 0
    return int(np.count_nonzero(np.max(np.abs(samples), axis=-1) > ceiling))


_TRUE_PEAK_KERNELS: Dict[int, np.ndarray] = {}


def _true_peak_kernel(factor: int) -> np.ndarray:
    cached = _TRUE_PEAK_KERNELS.get(factor)
    if cached is not None:
        return cached
    half = 32 * factor
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(offsets / factor) * np.kaiser(offsets.size, 8.0)
    for phase in range(factor):
        phase_indices = np.nonzero(np.mod(offsets.astype(np.int64), factor) == phase)[0]
        phase_sum = float(np.sum(kernel[phase_indices]))
        if phase_sum != 0.0:
            kernel[phase_indices] /= phase_sum
    _TRUE_PEAK_KERNELS[factor] = kernel
    return kernel


def true_peak_dbtp(samples: np.ndarray, factor: int = 4) -> Optional[float]:
    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data[:, None]
    kernel = _true_peak_kernel(factor)
    maximum = 0.0
    for channel in range(data.shape[1]):
        column = data[:, channel]
        upsampled = np.zeros(column.size * factor, dtype=np.float64)
        upsampled[::factor] = column
        oversampled = np.convolve(upsampled, kernel, mode="same")
        maximum = max(maximum, float(np.max(np.abs(oversampled))))
    return linear_to_db(maximum)


def _biquad_channel(
    samples: np.ndarray,
    b: Sequence[float],
    a: Sequence[float],
) -> np.ndarray:
    b0, b1, b2 = b
    _, a1, a2 = a
    output = np.empty_like(samples)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for index in range(samples.size):
        x0 = float(samples[index])
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[index] = y0
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
    return output


def k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 48000:
        raise BakerError(
            "E_UNSUPPORTED_SAMPLE_RATE",
            "K-weighting coefficients are defined for 48 kHz only.",
            {"sample_rate": sample_rate},
        )
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    shelf_b = (1.53512485958697, -2.69169618940638, 1.19839281085285)
    shelf_a = (1.0, -1.69065929318241, 0.73248077421585)
    highpass_b = (1.0, -2.0, 1.0)
    highpass_a = (1.0, -1.99004745483398, 0.99007225036621)
    output = np.empty_like(data)
    for channel in range(data.shape[1]):
        filtered = _biquad_channel(data[:, channel], shelf_b, shelf_a)
        output[:, channel] = _biquad_channel(filtered, highpass_b, highpass_a)
    return output


def loudness_parameters(sample_rate: int = 48000) -> Dict[str, Any]:
    block_size = int(round(LOUDNESS_BLOCK_SECONDS * sample_rate))
    hop_size = int(round(LOUDNESS_HOP_SECONDS * sample_rate))
    return {
        "algorithm": "ITU-R_BS.1770-4_integrated_loudness",
        "sample_rate": sample_rate,
        "block_seconds": LOUDNESS_BLOCK_SECONDS,
        "overlap": LOUDNESS_OVERLAP,
        "hop_seconds": LOUDNESS_HOP_SECONDS,
        "block_samples": block_size,
        "hop_samples": hop_size,
        "absolute_gate_lufs": LOUDNESS_ABSOLUTE_GATE,
        "relative_gate_lu": LOUDNESS_RELATIVE_GATE,
        "channel_weights": [1.0, 1.0],
        "k_weighting": {
            "shelf_b": [1.53512485958697, -2.69169618940638, 1.19839281085285],
            "shelf_a": [1.0, -1.69065929318241, 0.73248077421585],
            "highpass_b": [1.0, -2.0, 1.0],
            "highpass_a": [1.0, -1.99004745483398, 0.99007225036621],
        },
    }


def measure_integrated_loudness(
    samples: np.ndarray,
    sample_rate: int = 48000,
) -> Dict[str, Any]:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.size == 0 or float(np.max(np.abs(data))) == 0.0:
        return {
            "integrated_lufs": None,
            "block_count": 0,
            "absolute_gated_block_count": 0,
            "relative_gated_block_count": 0,
        }
    block_size = int(round(LOUDNESS_BLOCK_SECONDS * sample_rate))
    hop_size = int(round(LOUDNESS_HOP_SECONDS * sample_rate))
    if data.shape[0] < block_size:
        data = np.pad(data, ((0, block_size - data.shape[0]), (0, 0)))
    weighted = k_weight(data, sample_rate)
    energies: List[float] = []
    for start in range(0, weighted.shape[0] - block_size + 1, hop_size):
        block = weighted[start : start + block_size]
        channel_energies = np.mean(block * block, axis=0)
        energies.append(float(np.sum(channel_energies)))
    energy_array = np.asarray(energies, dtype=np.float64)
    block_loudness = -0.691 + 10.0 * np.log10(
        np.maximum(energy_array, np.finfo(np.float64).tiny)
    )
    absolute_mask = block_loudness > LOUDNESS_ABSOLUTE_GATE
    if not np.any(absolute_mask):
        return {
            "integrated_lufs": None,
            "block_count": int(energy_array.size),
            "absolute_gated_block_count": 0,
            "relative_gated_block_count": 0,
        }
    absolute_energy = float(np.mean(energy_array[absolute_mask]))
    ungated_loudness = -0.691 + 10.0 * math.log10(
        max(absolute_energy, np.finfo(np.float64).tiny)
    )
    relative_threshold = ungated_loudness + LOUDNESS_RELATIVE_GATE
    relative_mask = absolute_mask & (block_loudness > relative_threshold)
    if not np.any(relative_mask):
        relative_mask = absolute_mask
    final_energy = float(np.mean(energy_array[relative_mask]))
    integrated = -0.691 + 10.0 * math.log10(
        max(final_energy, np.finfo(np.float64).tiny)
    )
    return {
        "integrated_lufs": float(integrated),
        "block_count": int(energy_array.size),
        "absolute_gated_block_count": int(np.count_nonzero(absolute_mask)),
        "relative_gated_block_count": int(np.count_nonzero(relative_mask)),
    }


def _future_rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    window = max(1, min(window, values.size))
    reversed_values = values[::-1]
    output_reversed = np.empty_like(reversed_values)
    deque_indices: List[int] = []
    head = 0
    for index, value in enumerate(reversed_values):
        while len(deque_indices) > head and reversed_values[deque_indices[-1]] >= value:
            deque_indices.pop()
        deque_indices.append(index)
        start = index - window + 1
        while len(deque_indices) > head and deque_indices[head] < start:
            head += 1
        output_reversed[index] = reversed_values[deque_indices[head]]
    return output_reversed[::-1]


def lookahead_limiter_gain(
    samples: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float,
    lookahead_ms: float,
    release_ms: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.size == 0:
        return np.ones(0, dtype=np.float64), {
            "maximum_gain_reduction_db": 0.0,
            "limited_sample_count": 0,
        }
    threshold = db_to_linear(ceiling_dbtp)
    peak = np.max(np.abs(data), axis=1)
    desired = np.minimum(1.0, threshold / np.maximum(peak, 1e-15))
    lookahead_samples = max(1, int(round(lookahead_ms * sample_rate / 1000.0)))
    future = _future_rolling_min(desired, lookahead_samples)
    release_samples = max(1.0, release_ms * sample_rate / 1000.0)
    release_coefficient = math.exp(-1.0 / release_samples)
    envelope = np.empty_like(future)
    previous = 1.0
    for index in range(future.size):
        target = float(future[index])
        if target > previous:
            previous = release_coefficient * previous + (1.0 - release_coefficient) * target
        else:
            previous = target
        envelope[index] = previous
    limited_count = int(np.count_nonzero(envelope < 0.999999))
    minimum_gain = float(np.min(envelope))
    return envelope, {
        "maximum_gain_reduction_db": (
            -linear_to_db(minimum_gain) if minimum_gain > 0 else None
        ),
        "limited_sample_count": limited_count,
        "lookahead_samples": lookahead_samples,
        "release_samples": int(round(release_samples)),
        "sample_ceiling_linear": threshold,
    }


def _chunk(tag: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) % 2 else b""
    return tag + struct.pack("<I", len(payload)) + payload + padding


def write_pcm16_wav(
    path: Path,
    samples: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output samples must be a two-channel array.",
            {"shape": list(data.shape)},
        )
    pre_peak = float(np.max(np.abs(data))) if data.size else 0.0
    quantized = np.rint(np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    clipped = int(np.count_nonzero(np.max(np.abs(data), axis=1) > 1.0)) if data.size else 0
    interleaved = np.ascontiguousarray(quantized).tobytes(order="C")
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    info_payload = b"INFO" + _chunk(b"ICMT", metadata_bytes)
    fmt_payload = struct.pack(
        "<HHIIHH",
        1,
        2,
        48000,
        48000 * 2 * 2,
        2 * 2,
        16,
    )
    chunks = (
        _chunk(b"fmt ", fmt_payload)
        + _chunk(b"LIST", info_payload)
        + _chunk(b"bakr", metadata_bytes)
        + _chunk(b"data", interleaved)
    )
    riff = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    atomic_write(path, riff)
    return {
        "pre_quantization_peak": pre_peak,
        "pre_quantization_peak_dbfs": linear_to_db(pre_peak),
        "quantization_clipped_sample_count": clipped,
        "frame_count": int(data.shape[0]),
        "file_size_bytes": len(riff),
    }


def read_pcm16_wav(path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as source:
            info = {
                "channels": source.getnchannels(),
                "sample_rate": source.getframerate(),
                "sample_width_bytes": source.getsampwidth(),
                "frame_count": source.getnframes(),
                "compression": source.getcomptype(),
            }
            raw = source.readframes(source.getnframes())
    except (wave.Error, OSError) as exc:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output WAV could not be read back.",
            {"path": str(path), "reason": str(exc)},
        ) from exc
    if info["channels"] != 2 or info["sample_rate"] != 48000 or info["sample_width_bytes"] != 2:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output WAV is not 48 kHz, 16-bit, stereo PCM.",
            info,
        )
    integers = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if integers.size != info["frame_count"] * 2:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output WAV data chunk length is inconsistent.",
            {"path": str(path), "decoded_interleaved_samples": int(integers.size)},
        )
    return (integers.reshape(-1, 2) / 32767.0), info


def ffprobe_format(path: Path) -> Dict[str, Any]:
    ffprobe = _require_executable("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,sample_fmt,bits_per_sample",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "FFprobe could not validate the output WAV.",
            {
                "path": str(path),
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-2000:],
            },
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        stream = payload["streams"][0]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "FFprobe returned malformed stream information.",
            {"path": str(path)},
        ) from exc
    required = {
        "codec_name": "pcm_s16le",
        "sample_rate": "48000",
        "channels": 2,
        "bits_per_sample": 16,
    }
    mismatched = {
        key: {"actual": stream.get(key), "expected": expected}
        for key, expected in required.items()
        if stream.get(key) != expected
    }
    if mismatched:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "FFprobe reported an unexpected output audio format.",
            {"path": str(path), "mismatched": mismatched},
        )
    return {
        "tool": "ffprobe",
        "ok": True,
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate")),
        "channels": int(stream.get("channels")),
        "bits_per_sample": int(stream.get("bits_per_sample")),
        "sample_fmt": stream.get("sample_fmt"),
    }


def read_riff_metadata(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output WAV could not be opened.",
            {"path": str(path), "reason": str(exc)},
        ) from exc
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output file is not a RIFF/WAVE file.",
            {"path": str(path)},
        )
    position = 12
    metadata: Optional[Dict[str, Any]] = None
    while position + 8 <= len(data):
        tag = data[position : position + 4]
        size = struct.unpack("<I", data[position + 4 : position + 8])[0]
        payload_start = position + 8
        payload_end = payload_start + size
        if payload_end > len(data):
            break
        if tag == b"bakr":
            metadata = json.loads(data[payload_start:payload_end].decode("utf-8"))
            break
        position = payload_end + (size % 2)
    if metadata is None:
        raise BakerError(
            "E_OUTPUT_FORMAT",
            "Output WAV does not contain the dialogue_baker metadata chunk.",
            {"path": str(path)},
        )
    return metadata

