"""WAV PCM output writer using the standard library ``wave`` module.

Supports 16/24/32-bit PCM. 16/24-bit outputs use seeded triangular-probability
(TPDF) dither so repeated renders produce byte-identical files; 32-bit output
is written without dither because its headroom makes quantization noise
inaudible and tests can compare exact integer values.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .errors import CineMixerError


def _quantize(audio: np.ndarray, bit_depth: int, rng: np.random.Generator) -> np.ndarray:
    max_amplitude = 2.0 ** (bit_depth - 1)
    scaled = audio * max_amplitude
    if bit_depth < 32:
        # TPDF dither: sum of two uniform variables, peak amplitude 1 LSB.
        dither = rng.uniform(-1.0, 1.0, size=scaled.shape) + rng.uniform(
            -1.0, 1.0, size=scaled.shape
        )
        scaled = scaled + dither
    clipped = np.clip(scaled, -max_amplitude, max_amplitude - 1.0)
    return np.rint(clipped).astype("<i4")


def _pack_samples(samples: np.ndarray, bit_depth: int) -> bytes:
    little = samples.astype("<i4", copy=False)
    if bit_depth == 16:
        return little.astype("<i2").tobytes()
    if bit_depth == 32:
        return little.astype("<i4").tobytes()
    # 24-bit: pack each little-endian int32 into three bytes manually.
    flat = little.reshape(-1).astype(np.uint32)
    raw = np.zeros((flat.shape[0], 3), dtype=np.uint8)
    raw[:, 0] = flat & 0xFF
    raw[:, 1] = (flat >> 8) & 0xFF
    raw[:, 2] = (flat >> 16) & 0xFF
    return raw.tobytes()


def write_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
    seed: int = 12345,
) -> Path:
    if audio.ndim != 2:
        raise ValueError("write_wav expects an (samples, channels) array")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = _quantize(np.asarray(audio, dtype=np.float64), bit_depth, rng)
    payload = _pack_samples(samples, bit_depth)

    try:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(audio.shape[1])
            wav_file.setsampwidth(bit_depth // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(payload)
    except OSError as exc:
        raise CineMixerError("E_OUTPUT_FAIL", f"cannot write output WAV {output_path}: {exc}", 8) from exc
    return output_path


def read_wav(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM WAV back as float64 ``(samples, channels)`` plus metadata."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            raw = wav_file.readframes(frames)
    except (wave.Error, OSError) as exc:
        raise CineMixerError("E_OUTPUT_FAIL", f"cannot read WAV {path}: {exc}", 8) from exc

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 2.0**15
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2.0**31
    elif sample_width == 3:
        triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        data = values.astype(np.float64) / 2.0**23
    else:
        raise CineMixerError("E_OUTPUT_FAIL", f"unsupported sample width {sample_width}", 8)
    return data.reshape(-1, channels), sample_rate, sample_width * 8
