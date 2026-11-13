from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from localizer.errors import OperationCancelled
from localizer.manifest import Clip, TargetFormat


TARGET_PEAK = 10.0 ** (-1.0 / 20.0)
_SUBTYPE_TO_BITS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ULAW": 8,
    "ALAW": 8,
}
_BITS_TO_SUBTYPE = {
    8: "PCM_U8",
    16: "PCM_16",
    24: "PCM_24",
    32: "PCM_32",
}


@dataclass(frozen=True)
class LoadedClip:
    clip: Clip
    audio: np.ndarray
    changes: dict[str, Any]


@dataclass(frozen=True)
class MixedAudio:
    mono: np.ndarray
    stereo: np.ndarray
    mono_peak_before_normalization: float
    stereo_peak_before_normalization: float


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise OperationCancelled()


def _sample_count(milliseconds: int, sample_rate: int) -> int:
    return int(round(milliseconds * sample_rate / 1000.0))


def _resample_linear(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    source_frames = audio.shape[0]
    target_frames = max(1, int(round(source_frames * target_rate / source_rate)))
    if source_frames == 1:
        return np.repeat(audio, target_frames, axis=0)
    positions = np.arange(target_frames, dtype=np.float64) * (
        source_rate / target_rate
    )
    positions = np.minimum(positions, source_frames - 1)
    result = np.empty((target_frames, audio.shape[1]), dtype=np.float64)
    x = np.arange(source_frames, dtype=np.float64)
    for channel in range(audio.shape[1]):
        result[:, channel] = np.interp(positions, x, audio[:, channel])
    return result


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.shape[1] == 2:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    mono = audio.mean(axis=1, keepdims=True)
    return np.repeat(mono, 2, axis=1)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.shape[1] == 1:
        return audio[:, 0]
    return audio.mean(axis=1)


def _fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[0] == frames:
        return audio
    if audio.shape[0] > frames:
        return audio[:frames]
    padding = np.zeros((frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.concatenate((audio, padding), axis=0)


def load_clip(
    clip: Clip,
    target: TargetFormat,
    cancel: threading.Event | None = None,
) -> LoadedClip:
    _check_cancel(cancel)
    info = sf.info(clip.path)
    if not info.format.upper().startswith("WAV"):
        raise ValueError(f"not a WAV file (detected {info.format})")
    source_bit_depth = _SUBTYPE_TO_BITS.get(info.subtype.upper())
    if source_bit_depth is None:
        raise ValueError(f"unsupported WAV sample subtype: {info.subtype}")

    audio, source_rate = sf.read(
        clip.path,
        dtype="float64",
        always_2d=True,
        frames=-1,
    )
    if audio.shape[0] == 0:
        raise ValueError("WAV contains no audio frames")
    if not np.all(np.isfinite(audio)):
        raise ValueError("WAV contains NaN or infinite samples")

    changes: dict[str, Any] = {}
    if source_rate != target.sample_rate:
        changes["sample_rate"] = [source_rate, target.sample_rate]
        audio = _resample_linear(audio, source_rate, target.sample_rate)
    if audio.shape[1] != target.channels:
        changes["channels"] = [audio.shape[1], target.channels]
        audio = _to_stereo(audio) if target.channels == 2 else _to_mono(audio)[:, None]
    if source_bit_depth != target.bit_depth:
        changes["bit_depth"] = [source_bit_depth, target.bit_depth]

    expected_frames = _sample_count(clip.duration_ms, target.sample_rate)
    actual_frames = audio.shape[0]
    if actual_frames != expected_frames:
        changes["sample_count"] = [actual_frames, expected_frames]
    audio = _fit_length(audio, expected_frames)
    _check_cancel(cancel)
    return LoadedClip(clip=clip, audio=audio, changes=changes)


def _apply_crossfades(
    loaded: list[LoadedClip],
    sample_rate: int,
) -> list[np.ndarray]:
    result = [item.audio.copy() for item in loaded]
    for index in range(len(loaded) - 1):
        current = loaded[index].clip
        following = loaded[index + 1].clip
        if following.start_ms >= current.end_ms:
            continue
        current_start = _sample_count(current.start_ms, sample_rate)
        current_end = current_start + result[index].shape[0]
        following_start = _sample_count(following.start_ms, sample_rate)
        fade_frames = current_end - following_start
        if fade_frames <= 0:
            continue
        current_offset = result[index].shape[0] - fade_frames
        current_gain = np.linspace(1.0, 0.0, fade_frames, endpoint=False)
        following_gain = np.linspace(0.0, 1.0, fade_frames, endpoint=False)
        result[index][current_offset:current_end - current_start] *= current_gain[:, None]
        following_frames = result[index + 1].shape[0]
        applicable = min(fade_frames, following_frames)
        result[index + 1][:applicable] *= following_gain[:applicable, None]
    return result


def mix_language(
    loaded: list[LoadedClip],
    target: TargetFormat,
    total_ms: int,
    cancel: threading.Event | None = None,
) -> MixedAudio:
    total_frames = _sample_count(total_ms, target.sample_rate)
    mono_base = np.zeros(total_frames, dtype=np.float64)
    stereo_base = np.zeros((total_frames, 2), dtype=np.float64)
    crossfaded = _apply_crossfades(loaded, target.sample_rate)

    for item, audio in zip(loaded, crossfaded):
        _check_cancel(cancel)
        clip = item.clip
        start = _sample_count(clip.start_ms, target.sample_rate)
        if start >= total_frames:
            continue
        clip_audio = audio
        if target.channels == 1:
            stereo_audio = np.repeat(clip_audio, 2, axis=1)
            mono_audio = clip_audio[:, 0]
        else:
            stereo_audio = clip_audio
            mono_audio = clip_audio.mean(axis=1)
        available = min(clip_audio.shape[0], total_frames - start)
        if available <= 0:
            continue
        mono_base[start:start + available] += mono_audio[:available]
        stereo_base[start:start + available] += stereo_audio[:available]

    mono_peak = float(np.max(np.abs(mono_base), initial=0.0))
    stereo_peak = float(np.max(np.abs(stereo_base), initial=0.0))
    if mono_peak > 0.0:
        mono_base *= TARGET_PEAK / mono_peak
    if stereo_peak > 0.0:
        stereo_base *= TARGET_PEAK / stereo_peak
    mono = np.clip(mono_base, -1.0, 1.0)
    stereo = np.clip(stereo_base, -1.0, 1.0)
    _check_cancel(cancel)
    return MixedAudio(
        mono=mono,
        stereo=stereo,
        mono_peak_before_normalization=mono_peak,
        stereo_peak_before_normalization=stereo_peak,
    )


def write_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    bit_depth: int,
) -> None:
    subtype = _BITS_TO_SUBTYPE.get(bit_depth)
    if subtype is None:
        raise ValueError(f"unsupported output bit depth: {bit_depth}")
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if not math.isfinite(float(np.max(np.abs(data), initial=0.0))):
        raise ValueError("refusing to write non-finite audio")
    sf.write(
        path,
        np.clip(data, -1.0, 1.0),
        sample_rate,
        subtype=subtype,
        format="WAV",
        endian="FILE",
    )


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio), initial=0.0))
    return -math.inf if peak == 0.0 else 20.0 * math.log10(peak)
