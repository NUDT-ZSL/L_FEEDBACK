from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_audio(
    path: Path,
    sample_rate: int,
    channels: int,
    duration: float,
    *,
    frequency: float = 440.0,
    amplitude: float = 0.1,
    impulse_at: int | None = None,
    subtype: str = "PCM_24",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(sample_rate * duration))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * t)
    if impulse_at is not None and 0 <= impulse_at < frames:
        mono[impulse_at] = 1.0
    if channels == 1:
        data = mono.astype(np.float32)
    else:
        data = np.repeat(mono[:, None], channels, axis=1).astype(np.float32)
    sf.write(str(path), data, sample_rate, subtype=subtype)


def manifest(
    root: Path,
    tracks: list[dict[str, Any]],
    *,
    duration_ms: float = 5_000.0,
    sample_rate: int = 48_000,
    version: str = "v1",
    target_lufs: float = -16.0,
    true_peak_db: float = -1.0,
    duck_db: float = -12.0,
    master: str = "out/master.wav",
    report: str = "out/report.json",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "project": {
            "name": "test-project",
            "version": version,
            "segment_id": "scene",
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "channels": 2,
        },
        "loudness": {
            "target_lufs": target_lufs,
            "true_peak_db": true_peak_db,
        },
        "dialogue_ducking": {
            "duck_db": duck_db,
            "safe_headroom_db": -0.5,
        },
        "tracks": tracks,
        "output": {
            "master": master,
            "report": report,
            "sample_format": "PCM_24",
        },
    }


def write_manifest(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_cli(
    manifest_path: Path,
    *,
    state_dir: Path | None = None,
    env_extra: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (os.pathsep + current if current else "")
    if env_extra:
        env.update(env_extra)
    command = [
        sys.executable,
        "-m",
        "cutmaster",
        "--manifest",
        str(manifest_path),
    ]
    if state_dir is not None:
        command.extend(("--state-dir", str(state_dir)))
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    return data, sample_rate


def goertzel_amplitude(
    signal: np.ndarray, sample_rate: int, frequency: float
) -> float:
    frames = signal.shape[0]
    t = np.arange(frames, dtype=np.float64) / sample_rate
    basis = np.exp(-2j * np.pi * frequency * t)
    return float(abs(np.dot(signal, basis)) * 2.0 / frames)
