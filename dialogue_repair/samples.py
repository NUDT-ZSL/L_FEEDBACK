"""Deterministic built-in audio fixtures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic import commit_outputs
from .jsonutil import fixed_json
from .manifest import SAMPLE_RATE
from .wavio import encode_wav


@dataclass(frozen=True)
class SampleBundle:
    directory: Path
    dialogue_wav: Path
    reference_wav: Path
    manifest_json: Path


def _dialogue_samples() -> np.ndarray:
    count = 72000
    index = np.arange(count, dtype=np.float64)
    time = index / float(SAMPLE_RATE)
    voice = (
        0.105 * np.sin(2.0 * math.pi * 173.0 * time)
        + 0.048 * np.sin(2.0 * math.pi * 311.0 * time + 0.37)
        + 0.026 * np.sin(2.0 * math.pi * 619.0 * time + 1.10)
    )
    envelope = 0.45 + 0.55 * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * 2.7 * time - math.pi / 2.0)
    )
    samples = voice * envelope
    samples[17000:20500] = 0.0
    samples[43000:43800] = 0.0

    pop_start = 11800
    pop_length = 180
    pop_index = np.arange(pop_length, dtype=np.float64)
    pop = (
        0.68
        * np.sin(2.0 * math.pi * 1875.0 * pop_index / float(SAMPLE_RATE))
        * np.exp(-pop_index / 22.0)
    )
    samples[pop_start : pop_start + pop_length] += pop
    samples[46800:46864] += (
        np.where(np.arange(64) % 2 == 0, 0.35, -0.35)
        * np.exp(-np.arange(64, dtype=np.float64) / 7.0)
    )
    return np.clip(samples, -0.98, 0.98)


def _reference_samples() -> np.ndarray:
    count = 12000
    index = np.arange(count, dtype=np.float64)
    time = index / float(SAMPLE_RATE)
    return (
        (0.12 + 0.03 * np.sin(2.0 * math.pi * 1.3 * time))
        * np.sin(2.0 * math.pi * 247.0 * time + 0.20)
        + 0.045 * np.sin(2.0 * math.pi * 494.0 * time + 0.75)
    )


def builtin_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "sample_rate": SAMPLE_RATE,
        "quantization_bits": 24,
        "operations": [
            {
                "index": 0,
                "type": "replace",
                "start_sample": 11600,
                "end_sample": 12800,
                "frequency_hz": 523.25,
                "phase_radians": 0.125,
                "amplitude": 0.22,
                "fade_ms": 2.0,
            },
            {
                "index": 1,
                "type": "splice",
                "start_sample": 18000,
                "end_sample": 20400,
                "reference_file": "reference.wav",
                "reference_start": 1200,
                "reference_end": 3600,
                "fade_ms": 3.0,
            },
            {
                "index": 2,
                "type": "gain",
                "start_sample": 34000,
                "end_sample": 36000,
                "gain": 0.35,
            },
        ],
    }


def generate_sample_bundle(directory: str | Path) -> SampleBundle:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dialogue_path = directory / "dialogue.wav"
    reference_path = directory / "reference.wav"
    manifest_path = directory / "repair.json"

    dialogue_wav, _ = encode_wav(
        _dialogue_samples(), quantization_bits=24, sample_rate=SAMPLE_RATE
    )
    reference_wav, _ = encode_wav(
        _reference_samples(), quantization_bits=24, sample_rate=SAMPLE_RATE
    )
    manifest_bytes = (fixed_json(builtin_manifest(), indent=2) + "\n").encode("utf-8")
    commit_outputs(
        [
            (dialogue_path, dialogue_wav),
            (reference_path, reference_wav),
            (manifest_path, manifest_bytes),
        ]
    )
    return SampleBundle(
        directory=directory,
        dialogue_wav=dialogue_path,
        reference_wav=reference_path,
        manifest_json=manifest_path,
    )


def write_nonconforming_reference(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = 0.1 * np.sin(
        2.0 * math.pi * 220.0 * np.arange(4410, dtype=np.float64) / 44100.0
    )
    wav_bytes, _ = encode_wav(samples, quantization_bits=16, sample_rate=44100)
    commit_outputs([(path, wav_bytes)])
    return path

