from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def invoke_cli(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "spatialmix", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def error_payload(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode != 0, result.stdout + result.stderr
    try:
        return json.loads(result.stderr)
    except json.JSONDecodeError:
        return {"stderr": result.stderr}


def make_static_config(
    path: Path,
    *,
    positions: Sequence[Sequence[float]] | None = None,
    channel: int = 0,
    air_threshold: float = 3.0,
) -> Path:
    config = {
        "hrtf_version": "local-hrtf-v1",
        "distance_model": {
            "type": "free_field",
            "reference_distance": 1.0,
            "minimum_distance": 0.05,
        },
        "air_absorption": {
            "enabled": True,
            "threshold": air_threshold,
            "coefficient_per_meter": 0.15,
        },
        "sources": [
            {
                "id": "static-source",
                "channel": channel,
                "position": [1.2, 0.3, 2.0],
                "gain": 0.8,
                "delay_samples": 32,
            }
        ],
    }
    if positions is not None:
        config["sources"][0]["positions"] = positions
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def make_stereo_wav(
    path: Path,
    *,
    sample_rate: int = 48000,
    duration: float = 0.5,
    subtype: str = "PCM_16",
    frames: int | None = None,
) -> Path:
    count = frames if frames is not None else int(round(sample_rate * duration))
    time = np.arange(count, dtype=np.float64) / sample_rate
    left = 0.35 * np.sin(2.0 * np.pi * 440.0 * time)
    right = 0.28 * np.sin(2.0 * np.pi * 777.0 * time + 0.3)
    stereo = np.column_stack((left, right))
    sf.write(str(path), stereo, samplerate=sample_rate, format="WAV", subtype=subtype)
    return path


@pytest.fixture()
def demo_files(tmp_path: Path) -> tuple[Path, Path]:
    from spatialmix.demo import make_demo

    return make_demo(tmp_path / "demo", sample_rate=48000, duration_seconds=0.5)
