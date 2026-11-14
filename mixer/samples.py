"""Deterministic local audio fixture generation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ms_to_samples

_TRACK_PHASE = {
    "dialogue": 0.17,
    "ambience": 0.83,
    "music": 1.31,
}


def generate_sample(
    path: str | Path,
    *,
    duration_ms: int | float,
    sample_rate: int = 48_000,
    channels: int = 1,
    amplitude: float = 0.25,
    frequency: float = 440.0,
    track_type: str = "dialogue",
    seed: int = 0,
    waveform: str = "tone",
    subtype: str = "PCM_16",
) -> Path:
    """Create a deterministic WAV fixture without external assets.

    ``waveform='tone'`` mixes three harmonically related sines. Stereo and
    other multichannel files use deterministic phase/level offsets per channel,
    which is useful for exercising the mono downmix path.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be between 0 and 1")
    if waveform not in ("tone", "constant"):
        raise ValueError("waveform must be 'tone' or 'constant'")

    frame_count = ms_to_samples(duration_ms, sample_rate)
    if frame_count <= 0:
        raise ValueError("duration must produce at least one sample")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if waveform == "constant":
        mono = np.full(frame_count, amplitude, dtype=np.float64)
    else:
        time = np.arange(frame_count, dtype=np.float64) / float(sample_rate)
        phase = (
            _TRACK_PHASE.get(track_type, 0.0)
            + (float(seed) * 0.6180339887498949)
            + 0.031 * float(seed)
        )
        mono = amplitude * (
            0.65 * np.sin(2.0 * math.pi * frequency * time + phase)
            + 0.25
            * np.sin(2.0 * math.pi * 2.0 * frequency * time + 0.5 * phase)
            + 0.10
            * np.sin(2.0 * math.pi * 3.0 * frequency * time + 0.25 * phase)
        )

    if channels == 1:
        output = mono[:, np.newaxis]
    else:
        channel_signals = []
        for channel in range(channels):
            offset = (channel - (channels - 1) / 2.0) * 0.04
            channel_signals.append(mono * (1.0 + offset))
        output = np.column_stack(channel_signals)

    output = np.clip(output, -1.0, 1.0)
    sf.write(
        str(target),
        output,
        int(sample_rate),
        format="WAV",
        subtype=subtype,
    )
    return target


generate_track = generate_sample
