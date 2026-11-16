from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


def make_base_wav(
    path: Path,
    *,
    sample_rate: int = 44100,
    duration: float = 1.5,
    marker_seconds: float = 0.12,
    seed: int = 197,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration * sample_rate))
    time_axis = np.arange(frame_count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    samples = np.zeros(frame_count, dtype=np.float64)
    marker = int(round(marker_seconds * sample_rate))
    click_length = max(1, int(round(0.026 * sample_rate)))
    click_time = np.arange(click_length, dtype=np.float64) / sample_rate
    decay = np.exp(-click_time / 0.0045)
    click = rng.normal(0.0, 1.0, click_length) * decay
    end = min(frame_count, marker + click_length)
    samples[marker:end] += 0.58 * click[: end - marker]

    sustain = (
        np.sin(2.0 * np.pi * 220.0 * time_axis)
        + 0.52 * np.sin(2.0 * np.pi * 329.63 * time_axis)
        + 0.22 * np.sin(2.0 * np.pi * 880.0 * time_axis)
    )
    gate = np.clip((time_axis - marker_seconds) * 9.0, 0.0, 1.0)
    decay_env = np.exp(-np.maximum(time_axis - marker_seconds, 0.0) / 0.85)
    samples += 0.12 * sustain * gate * decay_env
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples *= 0.35 / peak
    wavfile.write(str(path), sample_rate, samples.astype(np.float32))
    return marker


def skill_entry(
    skill_id: str,
    base_sample: str,
    marker: int,
    *,
    variant_count: int = 3,
    target_duration: float = 1.2,
    target_loudness_mlfs: float = -20.0,
    random_seed: int = 1234,
) -> dict[str, Any]:
    return {
        "skill_id": skill_id,
        "base_sample": base_sample,
        "variant_count": variant_count,
        "target_duration": target_duration,
        "transient_markers": [marker],
        "target_loudness_mlfs": target_loudness_mlfs,
        "random_seed": random_seed,
    }


def write_config(
    path: Path,
    skills: list[dict[str, Any]],
    *,
    output_dir: str = "out",
    temp_dir: str = "tmp",
    concurrency: int = 1,
    ffmpeg: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "version": 1,
        "output_dir": output_dir,
        "temp_dir": temp_dir,
        "concurrency": concurrency,
        "skills": skills,
    }
    if ffmpeg is not None:
        config["ffmpeg"] = ffmpeg
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
