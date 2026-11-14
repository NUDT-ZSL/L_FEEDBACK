"""Deterministic WAV output helpers built on top of PySoundFile."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

# libsndfile command id; not exposed as a constant by the Python binding.
SFC_SET_ADD_PEAK_CHUNK = 0x1050


def open_float_wav(
    path: str | Path, sample_rate: int, channels: int
) -> sf.SoundFile:
    """Open a 32-bit float WAV writer without a non-deterministic PEAK chunk."""
    handle = sf.SoundFile(
        str(path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="FLOAT",
    )
    # The default float WAV PEAK chunk carries a wall-clock timestamp.
    sf._snd.sf_command(
        handle._file,
        SFC_SET_ADD_PEAK_CHUNK,
        sf._ffi.NULL,
        0,
    )
    return handle
