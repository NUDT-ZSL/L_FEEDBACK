"""WAV I/O, FFmpeg validation / resampling and level metrics."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import CorruptAudioError, FFmpegUnavailableError, SampleRateError

# Serialise ffmpeg subprocess spawns; each call still uses its own temp files.
_ffmpeg_lock = threading.Lock()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def read_wav(path: str | os.PathLike):
    """Return (float64 samples shape (n,) or (n,ch), sample_rate).

    Raises classified CorruptAudioError for unreadable files instead of
    leaking soundfile internals.
    """
    p = Path(path)
    if not p.is_file():
        from .errors import MissingAudioError

        raise MissingAudioError(f"音频文件不存在: {p}")
    try:
        data, sr = sf.read(str(p), dtype="float64", always_2d=False)
    except Exception as exc:  # soundfile raises RuntimeError on corrupt files
        raise CorruptAudioError(f"WAV 无法解析（可能已损坏）: {p} ({exc})") from exc
    return data, int(sr)


def probe_sample_rate(path: str | os.PathLike) -> int:
    """Read the declared sample rate from the WAV header without decoding."""
    p = Path(path)
    if not p.is_file():
        from .errors import MissingAudioError

        raise MissingAudioError(f"音频文件不存在: {p}")
    try:
        info = sf.info(str(p))
    except Exception as exc:
        raise CorruptAudioError(f"WAV 头无法解析（可能已损坏）: {p} ({exc})") from exc
    return int(info.samplerate)


def write_wav_pcm16(path: str | os.PathLike, samples: np.ndarray, sample_rate: int) -> None:
    """Write interleaved PCM_16 WAV atomically (temp file + os.replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".wav", dir=str(target.parent))
    os.close(fd)
    try:
        clipped = np.clip(samples, -1.0, 1.0)
        sf.write(tmp_name, clipped, sample_rate, subtype="PCM_16")
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _run_ffmpeg(argv: list[str]) -> subprocess.CompletedProcess:
    if not ffmpeg_available():
        raise FFmpegUnavailableError("PATH 中未找到 ffmpeg/ffprobe，无法执行音频校验或重采样。")
    # Unique temp outputs live inside each call; the lock additionally guards
    # any shared FFmpeg resource on the host.
    with _ffmpeg_lock:
        return subprocess.run(argv, capture_output=True, text=True, check=False)


def verify_wav(path: str | os.PathLike) -> tuple[bool, str]:
    """Decode-check a WAV with ffmpeg. Returns (ok, ffmpeg_stderr_text)."""
    proc = _run_ffmpeg(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"]
    )
    return proc.returncode == 0, (proc.stderr or "").strip()


def resample_wav(
    source: str | os.PathLike,
    target_sr: int,
    *,
    work_dir: str | os.PathLike,
) -> Path:
    """Resample *source* to *target_sr* into a unique temp WAV (PCM_16)."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="resample_", suffix=".wav", dir=str(work))
    os.close(fd)
    os.unlink(tmp_name)  # ffmpeg creates the file itself
    proc = _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ar",
            str(target_sr),
            "-c:a",
            "pcm_s16le",
            tmp_name,
        ]
    )
    if proc.returncode != 0 or not Path(tmp_name).is_file():
        try:
            Path(tmp_name).unlink()
        except OSError:
            pass
        raise CorruptAudioError(f"FFmpeg 重采样失败: {source} -> {target_sr}Hz; {proc.stderr.strip()}")
    return Path(tmp_name)


def ensure_sample_rate(
    path: str | os.PathLike,
    expected_sr: int,
    *,
    work_dir: str | os.PathLike,
):
    """Read a WAV, using FFmpeg to resample when the rate differs.

    Returns (samples, sr, resampled: bool). A corrupt file still raises
    CorruptAudioError; a failed resample likewise.
    """
    # Probe the header first so a rate mismatch is detected even though the
    # decoding backend can transparently read differently rated files.
    declared_sr = probe_sample_rate(path)
    samples, _sr = read_wav(path)
    sr = declared_sr
    if sr == expected_sr:
        return samples, sr, False
    resampled_path = resample_wav(path, expected_sr, work_dir=work_dir)
    try:
        rs_samples, rs_sr = read_wav(resampled_path)
        rs_sr = probe_sample_rate(resampled_path)
    finally:
        try:
            resampled_path.unlink()
        except OSError:
            pass
    if rs_sr != expected_sr:
        raise SampleRateError(f"重采样后采样率仍不匹配: {path} 期望 {expected_sr}Hz，实际 {rs_sr}Hz")
    return rs_samples, rs_sr, True


def downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=1)


def to_interleaved(samples: np.ndarray) -> np.ndarray:
    """Ensure shape (n_frames, n_channels)."""
    return np.asarray(samples, dtype=np.float64)
