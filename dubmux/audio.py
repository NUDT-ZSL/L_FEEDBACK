"""Deterministic WAV I/O and frame-energy based dialogue segmentation.

All detection parameters are fixed up-front and recorded in the report, so
the same input always yields the same segment boundaries.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np

from .errors import DubError, E_INPUT_UNREADABLE


@dataclass(frozen=True)
class DetectConfig:
    threshold: float = 0.01        # RMS threshold on normalized [-1, 1] signal
    frame_ms: float = 25.0         # analysis frame length
    hop_ms: float = 10.0           # analysis hop
    min_gap_ms: float = 300.0      # silences shorter than this do not split
    min_segment_ms: float = 150.0  # speech runs shorter than this are dropped

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "min_gap_ms": self.min_gap_ms,
            "min_segment_ms": self.min_segment_ms,
        }


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as mono float64 in [-1, 1]. Returns (samples, rate)."""
    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, OSError) as exc:
        raise DubError(E_INPUT_UNREADABLE, f"cannot read WAV {path}: {exc}") from exc
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise DubError(E_INPUT_UNREADABLE, f"unsupported sample width {width} in {path}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    """Write mono 16-bit PCM WAV (deterministic bytes for identical input)."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def frame_rms(samples: np.ndarray, rate: int, cfg: DetectConfig) -> tuple[np.ndarray, int, int]:
    """Return (rms per frame, frame_len, hop) using fixed frame parameters."""
    frame_len = max(1, round(rate * cfg.frame_ms / 1000.0))
    hop = max(1, round(rate * cfg.hop_ms / 1000.0))
    if len(samples) < frame_len:
        return np.zeros(0), frame_len, hop
    n_frames = 1 + (len(samples) - frame_len) // hop
    windows = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, frame_len),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    return rms, frame_len, hop


def detect_segments(
    samples: np.ndarray, rate: int, cfg: DetectConfig
) -> tuple[list[tuple[float, float]], float]:
    """Detect dialogue segments via fixed-threshold frame energy.

    Returns (segments, leading_silence). Segments are (start, end) seconds on
    the frame grid; leading_silence is the start time of the first segment.
    """
    rms, frame_len, hop = frame_rms(samples, rate, cfg)
    if rms.size == 0 or float(rms.max()) < cfg.threshold:
        return [], 0.0
    speech = rms >= cfg.threshold
    # Raw speech runs in frame indices.
    runs: list[list[int]] = []
    start = None
    for i, flag in enumerate(speech):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(speech) - 1])
    # Merge runs separated by less than min_gap.
    gap_frames = round(cfg.min_gap_ms / cfg.hop_ms)
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 < gap_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))
    # Drop runs shorter than min_segment.
    min_frames = max(1, round(cfg.min_segment_ms / cfg.hop_ms))
    merged = [r for r in merged if r[1] - r[0] + 1 >= min_frames]
    segments = [
        (r[0] * hop / rate, (r[1] * hop + frame_len) / rate) for r in merged
    ]
    leading = segments[0][0] if segments else 0.0
    return segments, leading
