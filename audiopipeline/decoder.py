"""FFmpeg based audio decoding and high-quality resampling.

Inputs are decoded to interleaved 32-bit float PCM and resampled to
48 kHz stereo inside FFmpeg using the ``soxr`` resampler (Very High Q),
which provides strong alias rejection.  Only WAV and FLAC containers are
accepted at the pipeline level; a corrupt / unsupported file makes the
FFmpeg process fail and the caller maps that to ``E_DECODE_FAILED``.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

from .config import CHANNELS, SAMPLE_RATE
from .errors import PipelineError

SUPPORTED_SUFFIXES = {".wav", ".flac"}


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise PipelineError("E_DECODE_FAILED", f"无法调用 FFmpeg: {exc}")


def check_ffmpeg() -> tuple[str, str]:
    """Verify ffmpeg / ffprobe availability and return their versions."""
    for tool in ("ffmpeg", "ffprobe"):
        proc = _run([tool, "-version"])
        if proc.returncode != 0:
            raise PipelineError("E_DECODE_FAILED", f"{tool} 不可用或无法执行")
    ff = _run(["ffmpeg", "-version"]).stdout.decode("utf-8", "replace").splitlines()[0]
    fp = _run(["ffprobe", "-version"]).stdout.decode("utf-8", "replace").splitlines()[0]
    return ff, fp


def probe_duration_seconds(path: str) -> float | None:
    """Return container duration in seconds, or ``None`` when unavailable."""
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
        value = float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value >= 0 else None


def decode_audio(path: str) -> tuple[np.ndarray, int]:
    """Decode ``path`` to a float32 ``(frames, 2)`` array at 48 kHz.

    Raises :class:`PipelineError` with ``E_DECODE_FAILED`` when FFmpeg
    cannot decode the stream.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        path,
        # Resample to 48 kHz stereo float, highest-quality soxr engine.
        "-af",
        (
            "aresample=resampler=soxr:precision=28:osf=fltp:"
            f"sample_rate={SAMPLE_RATE}:out_chlayout=stereo"
        ),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise PipelineError(
            "E_DECODE_FAILED",
            f"{path}: {detail[-1] if detail else 'FFmpeg 未输出任何采样数据'}",
        )
    raw = np.frombuffer(proc.stdout, dtype="<f4")
    if raw.size == 0 or raw.size % CHANNELS != 0:
        raise PipelineError("E_DECODE_FAILED", f"{path}: 解码输出长度异常 ({raw.size})")
    audio = raw.reshape(-1, CHANNELS).astype(np.float32, copy=True)
    if not np.isfinite(audio).all():
        raise PipelineError("E_DECODE_FAILED", f"{path}: 解码数据包含非有限值")
    return audio, SAMPLE_RATE
