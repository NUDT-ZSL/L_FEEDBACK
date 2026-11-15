"""Deterministically construct the multi-format demonstration dataset."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Union

import numpy as np

from .wavio import inspect_wav, write_pcm_wav, write_wav_float32


def _time_axis(sample_rate: int, duration_seconds: float) -> np.ndarray:
    frames = int(math.floor(duration_seconds * sample_rate + 0.5))
    return np.arange(frames, dtype=np.float64) / float(sample_rate)


def _deterministic_texture(frames: int, seed: int) -> np.ndarray:
    """Small deterministic LCG noise, independent of NumPy's RNG streams."""

    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    values = np.empty(frames, dtype=np.float64)
    for index in range(frames):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
        values[index] = (float(state >> 40) / 8388608.0) - 1.0
    return values


def _mono_pad(
    sample_rate: int,
    duration_seconds: float,
    frequencies: tuple[float, ...],
    seed: int,
) -> np.ndarray:
    t = _time_axis(sample_rate, duration_seconds)
    signal = np.zeros_like(t)
    for index, frequency in enumerate(frequencies):
        signal += (
            0.28 / (index + 1)
        ) * np.sin(2.0 * math.pi * frequency * t + 0.17 * index)
    signal += 0.012 * _deterministic_texture(t.size, seed)
    attack = np.minimum(1.0, np.arange(t.size) / max(1.0, 0.004 * sample_rate))
    release_frames = max(1, int(0.006 * sample_rate))
    release = np.ones(t.size, dtype=np.float64)
    release[-release_frames:] = np.linspace(
        1.0, 0.0, release_frames, endpoint=False
    )
    return signal * attack * release


def _stereo_pad(
    sample_rate: int,
    duration_seconds: float,
    left_frequencies: tuple[float, ...],
    right_frequencies: tuple[float, ...],
) -> np.ndarray:
    left = _mono_pad(sample_rate, duration_seconds, left_frequencies, 17)
    right = _mono_pad(sample_rate, duration_seconds, right_frequencies, 29)
    return np.column_stack((left, right))


def _write_mulaw_like_unsupported(path: Path, sample_rate: int = 8000) -> None:
    """Create a valid RIFF container with compressed format tag 7."""

    channels = 1
    bits = 8
    block_align = 1
    fmt = struct.pack(
        "<HHIIHH",
        7,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    data = bytes((index * 37 + 11) & 0xFF for index in range(400))
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    payload = (
        b"RIFF"
        + struct.pack("<I", 4 + len(fmt_chunk) + len(data_chunk))
        + b"WAVE"
        + fmt_chunk
        + data_chunk
    )
    path.write_bytes(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_demo_dataset(
    out_dir: Union[str, Path],
) -> dict:
    """Write deterministic WAV sources, events, and a manifest.

    Event times deliberately include fractional sample positions and
    overlapping events. The loop regions have mismatched boundary samples so
    the crossfade result is measurable.
    """

    root = Path(out_dir).expanduser().resolve()
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    path_44k_16_mono = samples_dir / "source_44k_s16_mono.wav"
    write_pcm_wav(
        path_44k_16_mono,
        _mono_pad(44100, 0.70, (220.0, 331.0), seed=101),
        sample_rate=44100,
        bit_depth=16,
    )

    path_48k_24_stereo = samples_dir / "source_48k_s24_stereo.wav"
    write_pcm_wav(
        path_48k_24_stereo,
        _stereo_pad(48000, 0.42, (180.0, 277.0), (240.0, 359.0)),
        sample_rate=48000,
        bit_depth=24,
    )

    path_96k_float_mono = samples_dir / "source_96k_f32_mono.wav"
    write_wav_float32(
        path_96k_float_mono,
        _mono_pad(96000, 0.32, (300.0, 451.0), seed=211),
        sample_rate=96000,
    )

    path_32k_32_mono = samples_dir / "source_32k_s32_mono.wav"
    write_pcm_wav(
        path_32k_32_mono,
        _mono_pad(32000, 0.20, (150.0, 225.0), seed=307),
        sample_rate=32000,
        bit_depth=32,
    )

    unsupported_path = samples_dir / "source_unsupported_mulaw.wav"
    _write_mulaw_like_unsupported(unsupported_path)

    events = {
        "schema_version": "1.0",
        "description": "Deterministic multi-format event timeline",
        "events": [
            {
                "id": "footstep_loop",
                "file": "samples/source_44k_s16_mono.wav",
                "time": 0.10025,
                "pan_frequency_hz": 0.50,
                "pan_depth": 0.80,
                "loop": {"start": 0.05, "end": 0.55},
            },
            {
                "id": "stereo_ambience",
                "file": "samples/source_48k_s24_stereo.wav",
                "time": 0.28,
                "pan_frequency_hz": 1.25,
                "pan_depth": 0.60,
            },
            {
                "id": "float_loop",
                "file": "samples/source_96k_f32_mono.wav",
                "time": 0.46,
                "pan_frequency_hz": 2.00,
                "pan_depth": 0.35,
                "loop": {"start": 0.02, "end": 0.24},
            },
            {
                "id": "center_overlap",
                "file": "samples/source_32k_s32_mono.wav",
                "time": 0.62,
                "pan_frequency_hz": 0.0,
                "pan_depth": 0.0,
            },
        ],
    }
    events_path = root / "events.json"
    events_path.write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    wav_paths = [
        path_44k_16_mono,
        path_48k_24_stereo,
        path_96k_float_mono,
        path_32k_32_mono,
        unsupported_path,
    ]
    files = []
    for path in wav_paths:
        info = None
        if path != unsupported_path:
            info = inspect_wav(path)
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "supported": info is not None,
                "sample_rate": info.sample_rate if info else None,
                "channels": info.channels if info else None,
                "bit_depth": info.bits_per_sample if info else None,
                "encoding": info.encoding if info else "mu-law-like format tag 7",
            }
        )

    manifest = {
        "schema_version": "1.0",
        "deterministic": True,
        "root": str(root),
        "events_path": str(events_path),
        "output_sample_rate": 48000,
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
