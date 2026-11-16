"""Audio file IO helpers built on soundfile (+ ffmpeg/pydub availability)."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from . import TARGET_SR


def ffmpeg_available() -> str | None:
    """Return the ffmpeg executable path if it is on PATH, else None."""
    try:
        from pydub.utils import which
    except Exception:  # pragma: no cover - pydub is a declared dependency
        which = None
    path = which("ffmpeg") if which is not None else _which_plain("ffmpeg")
    return path


def _which_plain(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float64 mono (down-mixing channels) with its SR.

    WAV/FLAC/OGG are read through libsndfile. soundfile raises RuntimeError on
    unreadable or unsupported files; callers translate that into diagnostics.
    """
    data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return np.ascontiguousarray(data), int(sample_rate)


def save_wav_stereo(path: str, mono_mix: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    """Write a float mono signal as a 16-bit PCM stereo WAV (dual mono)."""
    mono_mix = np.asarray(mono_mix, dtype=np.float64)
    stereo = np.column_stack([mono_mix, mono_mix])
    stereo = np.clip(stereo, -1.0, 1.0)
    pcm16 = np.round(stereo * 32767.0).astype(np.int16)
    sf.write(path, pcm16, sample_rate, subtype="PCM_16")


def file_duration_seconds(path: str) -> float:
    """Return total duration in seconds via libsndfile header info."""
    info = sf.info(path)
    if info.samplerate <= 0:
        return 0.0
    return info.frames / float(info.samplerate)
