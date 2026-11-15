"""WAV loading, internal channel normalization and final WAV writing."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from .errors import E_CORRUPT_CLIP, E_MISSING_CLIP, PipelineError
from .resample import resample


def load_clip(path: str, target_sr: int) -> np.ndarray:
    """Load a WAV as float32 ``(frames, channels <= 2)`` at ``target_sr``.

    Missing files raise ``E_MISSING_CLIP``; unreadable / malformed files raise
    ``E_CORRUPT_CLIP``.  Mono stays mono here; stereo expansion happens while
    rendering so cached clips stay compact.  Surround (>2 ch) is down-mixed to
    mono by an unweighted channel average.
    """
    if not os.path.isfile(path):
        raise PipelineError(E_MISSING_CLIP, {"clip": path, "reason": "file not found"})

    try:
        data, source_sr = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise PipelineError(
            E_CORRUPT_CLIP, {"clip": path, "reason": str(exc)}
        ) from exc

    if data.size == 0 or data.ndim != 2 or data.shape[0] == 0:
        raise PipelineError(
            E_CORRUPT_CLIP, {"clip": path, "reason": "empty or unreadable audio data"}
        )
    if not np.isfinite(data).all():
        raise PipelineError(
            E_CORRUPT_CLIP, {"clip": path, "reason": "non-finite sample values"}
        )

    if data.shape[1] > 2:
        data = data.mean(axis=1, keepdims=True)

    if int(source_sr) != int(target_sr):
        data = resample(data, int(source_sr), int(target_sr))

    return np.ascontiguousarray(data, dtype=np.float32)


def to_stereo(data: np.ndarray) -> np.ndarray:
    """Expand a mono ``(frames, 1)`` buffer to stereo; stereo passes through."""
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write 24-bit PCM stereo WAV deterministically."""
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("write_wav expects a stereo (frames, 2) float32 array")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    sf.write(
        path,
        np.ascontiguousarray(audio, dtype=np.float32),
        int(sample_rate),
        subtype="PCM_24",
        format="WAV",
    )
