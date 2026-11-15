from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from podcast_post.wavio import write_wav


FS = 48000


def write_asset(path: Path, frames: int, channels: int = 1, value: float = 0.2) -> np.ndarray:
    samples = np.full((frames, channels), value, dtype=np.float64)
    write_wav(path, samples, FS, 24)
    return samples


def manifest_document(
    frames: int,
    clips: list[dict[str, Any]],
    *,
    edit_points: Iterable[dict[str, Any]] = (),
    chapters: Iterable[dict[str, Any]] | None = None,
    channels: int = 1,
    ducking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "sample_rate": FS,
        "channels": channels,
        "bit_depth": 24,
        "total_samples": frames,
        "tracks": [
            {
                "name": "host",
                "role": "host",
                "asset": "asset.wav",
                "gain_db": 0.0,
            },
            {
                "name": "guest",
                "role": "guest",
                "asset": "asset.wav",
                "gain_db": 0.0,
            },
            {
                "name": "music",
                "role": "music",
                "asset": "asset.wav",
                "gain_db": 0.0,
            },
        ],
        "clips": clips,
        "edit_points": list(edit_points),
        "chapters": list(chapters or [{"title": "All", "start_sample": 0}]),
        "processing": {
            "fade_samples": 100,
            "ducking": ducking
            or {
                "enabled": False,
                "trigger_track": "host",
                "target_track": "music",
                "threshold_dbfs": -30.0,
                "energy_window_ms": 20.0,
                "depth_db": 12.0,
                "attack_ms": 10.0,
                "release_ms": 250.0,
            },
            "loudness": {
                "target_lufs": -16.0,
                "max_true_peak_dbtp": -1.0,
            },
        },
    }


def write_manifest(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path

