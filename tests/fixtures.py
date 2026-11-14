from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def tone(
    sample_rate: int,
    duration: float,
    frequency: float = 440.0,
    amplitude: float = 0.1,
) -> np.ndarray:
    count = int(round(sample_rate * duration))
    time = np.arange(count, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def write_samples(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = 48_000,
    subtype: str = "FLOAT",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(samples, dtype=np.float32), sample_rate, subtype=subtype)


def keyframe(
    time: float,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    distance: float = 1.0,
    gain: float = 1.0,
    size: float = 0.0,
) -> dict[str, float]:
    return {
        "time": time,
        "azimuth": azimuth,
        "elevation": elevation,
        "distance": distance,
        "gain": gain,
        "size": size,
    }


def audio_object(
    object_id: str,
    keyframes: list[dict[str, float]],
    clips: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {"id": object_id, "keyframes": keyframes, "clips": clips or []}


def write_scene(
    path: Path,
    objects: list[dict[str, object]],
    sample_rate: int = 48_000,
    frame_length: int = 1024,
    duration: float | None = 1.0,
) -> Path:
    scene: dict[str, object] = {
        "sample_rate": sample_rate,
        "frame_length": frame_length,
        "objects": objects,
    }
    if duration is not None:
        scene["duration_seconds"] = duration
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    return path


def run_cli(
    *arguments: str,
    cwd: Path | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "immersive_audio", *arguments],
        cwd=os_fspath(cwd or ROOT),
        capture_output=True,
        text=True,
        check=check,
        timeout=120,
    )


def os_fspath(path: Path) -> str:
    return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")
