from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_tone(
    path: Path,
    *,
    sample_rate: int = 48000,
    duration_ms: int = 500,
    amplitude: float = 0.01,
    channels: int = 2,
    bit_depth: int = 24,
    frequency: float = 250.0,
    phase: float = np.pi / 2.0,
) -> np.ndarray:
    frame_count = int(round(duration_ms * sample_rate / 1000.0))
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(
        2.0 * np.pi * frequency * time + phase
    )
    data = np.repeat(mono[:, None], channels, axis=1)
    subtype = {8: "PCM_U8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}[
        bit_depth
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        data,
        sample_rate,
        format="WAVEX",
        subtype=subtype,
    )
    return data


def base_config(
    *,
    timeline_ms: int = 4000,
    sample_rate: int = 48000,
    bit_depth: int = 24,
    channels: int = 2,
    channel_layout: str = "stereo",
    clips: Optional[list] = None,
) -> Dict[str, Any]:
    return {
        "timeline_ms": timeline_ms,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "channels": channels,
        "channel_layout": channel_layout,
        "clips": clips if clips is not None else [],
    }


def clip(
    clip_id: str,
    input_path: str,
    start_ms: float,
    target_lufs: float,
    fade_in_ms: float = 0,
    fade_out_ms: float = 0,
) -> Dict[str, Any]:
    return {
        "id": clip_id,
        "input_path": input_path,
        "start_ms": start_ms,
        "target_lufs": target_lufs,
        "fade_in_ms": fade_in_ms,
        "fade_out_ms": fade_out_ms,
    }


def write_config(path: Path, value: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def run_cli(config_path: Path, output_dir: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing
        else str(REPO_ROOT) + os.pathsep + existing
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dialogue_master",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        text=True,
        capture_output=True,
    )


@pytest.fixture
def tone_factory():
    return write_tone
