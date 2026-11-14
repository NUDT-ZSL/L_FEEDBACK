from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tone(
    sample_rate: int = 48000,
    duration: float = 3.0,
    channels: int = 2,
    amplitude: float = 0.1,
    frequency: float = 1000.0,
) -> np.ndarray:
    time = np.arange(int(round(sample_rate * duration))) / float(sample_rate)
    wave = amplitude * np.sin(2.0 * np.pi * frequency * time)
    return np.stack([wave] * channels, axis=1)


def run_cli(
    arguments: Sequence[str],
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing
        else str(PROJECT_ROOT) + os.pathsep + existing
    )
    return subprocess.run(
        [sys.executable, "-m", "loudness_pipeline", *arguments],
        cwd=str(cwd or PROJECT_ROOT),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.fixture
def stereo_path(tmp_path: Path) -> Path:
    import soundfile as sf

    path = tmp_path / "stereo.wav"
    sf.write(
        str(path),
        tone(duration=4.0),
        48000,
        subtype="PCM_24",
        format="WAV",
    )
    return path


@pytest.fixture
def surround_path(tmp_path: Path) -> Path:
    import soundfile as sf

    sample_rate = 48000
    duration = 4.0
    time = np.arange(int(sample_rate * duration)) / float(sample_rate)
    specs = [
        (220.0, 0.12),
        (330.0, 0.11),
        (440.0, 0.10),
        (55.0, 0.35),
        (550.0, 0.08),
        (660.0, 0.07),
    ]
    audio = np.stack(
        [
            amplitude * np.sin(2.0 * np.pi * frequency * time)
            for frequency, amplitude in specs
        ],
        axis=1,
    )
    path = tmp_path / "surround.wav"
    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_24",
        format="WAV",
    )
    return path

