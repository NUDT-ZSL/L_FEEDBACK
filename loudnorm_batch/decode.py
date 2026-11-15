"""ffmpeg based decoding to float32 PCM."""

from __future__ import annotations

import subprocess

import numpy as np

FFMPEG = "ffmpeg"


class DecodeError(Exception):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def decode(
    path: str,
    channels: int,
    sample_rate: int | None = None,
    timeout: float = 30.0,
) -> tuple[np.ndarray, int, str, int]:
    """Decode a file to (samples, channels) float32 PCM.

    Returns (pcm, sample_rate, stderr_text, returncode). Partial decodes
    (non-zero exit code but some PCM produced) are returned so the caller
    can flag them; a DecodeError is raised only when no audio at all
    could be decoded.
    """
    cmd = [FFMPEG, "-v", "warning", "-i", path]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError("decode timeout", "ffmpeg decode timeout") from exc

    raw = proc.stdout
    frame_bytes = 4 * channels
    usable = len(raw) - (len(raw) % frame_bytes)
    pcm = np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, channels).astype(np.float64)
    stderr = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0 and pcm.shape[0] == 0:
        raise DecodeError("ffmpeg decode failed", stderr)
    return pcm, sample_rate or 0, stderr, proc.returncode
