"""Deterministic test-asset generator.

Every event asset is a sine wave rendered with a phase accumulator so the
sample stream is bit-identical for identical configs on every machine and
every run (no wall-clock data, no randomness, no network access).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .config import SceneConfig
from .wavio import write_wav_atomic


def render_sine(
    sample_rate: int,
    duration: float,
    frequency: float,
    amplitude: float,
    channels: int = 1,
) -> np.ndarray:
    """Render a deterministic sine starting at phase zero.

    A running phase accumulator (rather than ``2*pi*f*n/fs``) keeps floating
    point growth bounded for long assets while remaining fully deterministic.
    """
    frames = int(round(duration * sample_rate))
    phase_increment = 2.0 * np.pi * frequency / sample_rate
    indices = np.arange(frames, dtype=np.float64)
    phase = np.mod(indices * phase_increment, 2.0 * np.pi)
    mono = amplitude * np.sin(phase)
    if channels == 1:
        return mono.reshape(-1, 1)
    return np.repeat(mono.reshape(-1, 1), 2, axis=1)


def asset_path_for(scene: SceneConfig, event_id: str, asset_dir: Path) -> Path:
    safe = event_id.replace("/", "_").replace("\\", "_")
    return asset_dir / f"{scene.scene_id}_{safe}.wav"


def generate_assets(scene: SceneConfig, base_dir: str | Path | None = None) -> list[Path]:
    """Generate every event WAV declared by ``scene`` and return their paths."""
    base = Path(base_dir) if base_dir else Path.cwd()
    asset_dir = (
        Path(scene.asset_dir)
        if Path(scene.asset_dir).is_absolute()
        else base / scene.asset_dir
    )
    asset_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for track in scene.tracks:
        for event in track.events:
            sample_rate = event.sample_rate or scene.project_sample_rate
            target = (
                Path(event.asset)
                if event.asset
                else asset_path_for(scene, event.event_id, asset_dir)
            )
            if not target.is_absolute() and event.asset:
                target = base / target
            samples = render_sine(
                sample_rate=sample_rate,
                duration=event.duration,
                frequency=event.frequency,
                amplitude=event.amplitude,
                channels=event.channels,
            )
            write_wav_atomic(target, samples, sample_rate, event.sample_width)
            written.append(target)
    return written
