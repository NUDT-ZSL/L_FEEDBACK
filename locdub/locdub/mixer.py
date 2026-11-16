"""Streaming timeline mixer.

The background track is read in fixed-size blocks. Adapted dialogue clips
are opened lazily and only kept open while the write cursor is inside their
time window, so peak memory is independent of the number of segments (the
50+ / 5-minute scenario). Output is written straight to a WAV file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from .audioio import probe_wav


@dataclass
class MixJob:
    segment_id: str
    start_ms: int
    end_ms: int
    wav_path: Optional[Path]   # None => silence placeholder


def mix_stream(background_wav: Path, jobs: List[MixJob], out_wav: Path,
               sample_rate: int, channels: int, gain_db: float,
               block_seconds: float = 60.0) -> None:
    """Mix ``jobs`` over ``background_wav`` and write ``out_wav``.

    ``jobs`` must be sorted by ``start_ms`` and non-overlapping (guaranteed
    by timeline validation). Missing clips contribute silence.
    """
    bg_info = probe_wav(background_wav)
    total_frames = bg_info.frames
    block_frames = max(1, int(round(sample_rate * block_seconds)))
    gain_linear = 10.0 ** (gain_db / 20.0)

    ordered = sorted(jobs, key=lambda job: job.start_ms)
    open_handles: Dict[str, sf.SoundFile] = {}
    next_index = 0

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(out_wav), "w", samplerate=sample_rate,
                      channels=channels, subtype="PCM_16") as writer:
        for block_start in range(0, total_frames, block_frames):
            block_end = min(total_frames, block_start + block_frames)
            with sf.SoundFile(str(background_wav), "r") as bg:
                bg.seek(block_start)
                block = bg.read(block_end - block_start, dtype="float64",
                                 always_2d=True)
            mix = _match_channels(block, channels).copy()

            # Open every clip whose window starts within this block.
            while next_index < len(ordered):
                job = ordered[next_index]
                start_frame = int(round(job.start_ms * sample_rate / 1000.0))
                if start_frame >= block_end:
                    break
                if job.wav_path is not None and job.segment_id not in open_handles:
                    handle = sf.SoundFile(str(job.wav_path), "r")
                    clip_rate = handle.samplerate
                    if clip_rate != sample_rate:
                        handle.close()
                        raise ValueError(
                            f"clip {job.segment_id} sample rate "
                            f"{clip_rate} != mix rate {sample_rate}"
                        )
                    open_handles[job.segment_id] = handle
                next_index += 1

            # Close clips whose window ended before this block.
            for job in ordered[:next_index]:
                end_frame = int(round(job.end_ms * sample_rate / 1000.0))
                if end_frame <= block_start and job.segment_id in open_handles:
                    open_handles.pop(job.segment_id).close()

            for job in ordered[:next_index]:
                start_frame = int(round(job.start_ms * sample_rate / 1000.0))
                end_frame = int(round(job.end_ms * sample_rate / 1000.0))
                if end_frame <= block_start or start_frame >= block_end:
                    continue
                handle = open_handles.get(job.segment_id)
                if handle is None:
                    continue
                _overlay_job(handle, job, mix, block_start, block_end,
                             start_frame, end_frame, channels, gain_linear)

            # Soft clip protection, deterministic and allocation free.
            np.clip(mix, -1.0, 1.0, out=mix)
            writer.write(mix.astype(np.float32))

    for handle in open_handles.values():
        handle.close()


def _overlay_job(handle: sf.SoundFile, job: MixJob, mix: np.ndarray,
                 block_start: int, block_end: int,
                 start_frame: int, end_frame: int,
                 channels: int, gain_linear: float) -> None:
    overlap_start = max(block_start, start_frame)
    overlap_end = min(block_end, end_frame)
    if overlap_end <= overlap_start:
        return
    handle.seek(overlap_start - start_frame)
    wanted = overlap_end - overlap_start
    clip = handle.read(wanted, dtype="float64", always_2d=True)
    if len(clip) == 0:
        return
    clip = _match_channels(clip, channels) * gain_linear
    offset = overlap_start - block_start
    mix[offset:offset + len(clip)] += clip


def _match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1 and channels == 2:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] == 2 and channels == 1:
        return data.mean(axis=1, keepdims=True)
    # Fallback: truncate or tile to the requested channel count.
    if data.shape[1] > channels:
        return data[:, :channels]
    return np.pad(
        data,
        ((0, 0), (0, channels - data.shape[1])),
        mode="edge",
    )
