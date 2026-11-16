from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import (
    SAMPLE_RATE,
    LoadedAudio,
    SilenceTrim,
    load_mono_wav,
    peak_dbfs,
    time_stretch,
    trim_silence,
)
from .config import RenderConfig
from .errors import BusinessError
from .io_utils import sha256_file
from .model import Dialogue, Timeline
from .samples import dialogue_sample_path


@dataclass
class DialogueAdaptation:
    dialogue: Dialogue
    audio_path: Path
    audio_sha256: str
    source: LoadedAudio
    window_start_frame: int
    window_end_frame: int
    trim: SilenceTrim
    stretched_samples: np.ndarray | None
    final_samples: np.ndarray | None
    final_start_frame: int
    final_end_frame: int
    requested_ratio: float
    applied_ratio: float
    unadaptable: bool
    error: dict[str, Any] | None
    max_speedup: float
    atempo_crop_frames: int = 0
    atempo_pad_frames: int = 0

    @property
    def window_frames(self) -> int:
        return self.window_end_frame - self.window_start_frame

    @property
    def final_frames(self) -> int:
        if self.final_samples is None:
            return 0
        return int(self.final_samples.shape[0])

    def as_report(self) -> dict[str, Any]:
        dialogue = self.dialogue
        source_duration_s = self.source.frames / float(SAMPLE_RATE)
        source_peak = peak_dbfs(self.source.samples)
        if not math.isfinite(source_peak):
            source_peak = None
        subtitle_start_frame = int(round(dialogue.subtitle_start_s * SAMPLE_RATE))
        subtitle_end_frame = int(round(dialogue.subtitle_end_s * SAMPLE_RATE))
        trim = self.trim
        final_start_s = self.final_start_frame / float(SAMPLE_RATE)
        final_end_s = self.final_end_frame / float(SAMPLE_RATE)
        report: dict[str, Any] = {
            "id": dialogue.id,
            "shot_id": dialogue.shot_id,
            "subtitle_id": dialogue.subtitle_id,
            "target_language": dialogue.target_language,
            "text": dialogue.text,
            "status": "unadaptable" if self.unadaptable else "adapted",
            "unadaptable": self.unadaptable,
            "source_audio": str(self.audio_path),
            "source_audio_sha256": self.audio_sha256,
            "window": {
                "start_s": dialogue.start_s,
                "end_s": dialogue.end_s,
                "duration_s": dialogue.window_s,
                "start_frame": self.window_start_frame,
                "end_frame": self.window_end_frame,
                "frames": self.window_frames,
            },
            "subtitle": {
                "id": dialogue.subtitle_id,
                "start_s": dialogue.subtitle_start_s,
                "end_s": dialogue.subtitle_end_s,
                "duration_s": dialogue.subtitle_end_s - dialogue.subtitle_start_s,
                "start_frame": subtitle_start_frame,
                "end_frame": subtitle_end_frame,
            },
            "original": {
                "duration_s": source_duration_s,
                "frames": self.source.frames,
                "peak_dbfs": source_peak,
            },
            "silence_trim": {
                "applied": self.source.frames > self.window_frames,
                "threshold_dbfs": trim.threshold_dbfs,
                "margin_ms": trim.margin_ms,
                "lead_frames": trim.lead_frames,
                "lead_s": trim.lead_frames / float(SAMPLE_RATE),
                "trail_frames": trim.trail_frames,
                "trail_s": trim.trail_frames / float(SAMPLE_RATE),
                "total_frames": trim.total_silence_frames,
                "total_s": trim.total_silence_frames / float(SAMPLE_RATE),
                "trimmed_frames": trim.trimmed_frames,
                "trimmed_duration_s": trim.trimmed_frames / float(SAMPLE_RATE),
                "fully_silent": trim.fully_silent,
            },
            "stretch": {
                "max_ratio": self.max_speedup,
                "requested_ratio": self.requested_ratio,
                "applied_ratio": self.applied_ratio if not self.unadaptable else None,
                "filter": (
                    "none"
                    if abs(self.applied_ratio - 1.0) < 1e-12 and not self.unadaptable
                    else ("atempo" if not self.unadaptable else "none")
                ),
                "atempo_crop_frames": self.atempo_crop_frames,
                "atempo_pad_frames": self.atempo_pad_frames,
            },
            "final": {
                "start_s": final_start_s,
                "end_s": final_end_s,
                "duration_s": self.final_frames / float(SAMPLE_RATE),
                "start_frame": self.final_start_frame,
                "end_frame": self.final_end_frame,
                "frames": self.final_frames,
                "within_window": (
                    not self.unadaptable
                    and self.final_start_frame >= self.window_start_frame
                    and self.final_end_frame <= self.window_end_frame
                ),
                "within_subtitle": (
                    not self.unadaptable
                    and self.final_start_frame >= subtitle_start_frame
                    and self.final_end_frame <= subtitle_end_frame
                ),
            },
            "error": self.error,
        }
        return report


def adapt_dialogues(
    timeline: Timeline,
    *,
    audio_root: str | Path,
    config: RenderConfig,
) -> list[DialogueAdaptation]:
    root = Path(audio_root)
    adaptations: list[DialogueAdaptation] = []
    epsilon = 1e-9
    for dialogue in timeline.dialogues:
        path = dialogue_sample_path(root, dialogue.target_language, dialogue.id)
        source = load_mono_wav(path, entry_id=dialogue.id)
        audio_hash = sha256_file(path)
        window_start = int(round(dialogue.start_s * SAMPLE_RATE))
        window_end = int(round(dialogue.end_s * SAMPLE_RATE))
        window_frames = max(0, window_end - window_start)
        if window_frames <= 0:
            raise BusinessError(
                code="E_DIALOGUE_INVALID",
                message="Dialogue window is shorter than one output sample frame.",
                entry_id=dialogue.id,
                path=str(path),
            )

        overlong = source.frames > window_frames
        if overlong:
            trimmed_samples, trim = trim_silence(
                source.samples,
                sample_rate=SAMPLE_RATE,
                threshold_dbfs=config.silence_threshold_dbfs,
                margin_ms=config.silence_margin_ms,
            )
        else:
            trimmed_samples = source.samples.copy()
            trim = SilenceTrim(
                threshold_dbfs=config.silence_threshold_dbfs,
                margin_ms=config.silence_margin_ms,
                start_frame=0,
                end_frame=source.frames,
                original_frames=source.frames,
                fully_silent=False,
            )

        trimmed_frames = int(trimmed_samples.shape[0])
        if trimmed_frames > window_frames:
            requested_ratio = trimmed_frames / float(window_frames)
            if requested_ratio > config.max_speedup + epsilon:
                error = {
                    "code": "E_DIALOGUE_UNFIT",
                    "message": (
                        "Dialogue exceeds its window even after silence trimming, "
                        "and the required time-stretch exceeds max_speedup."
                    ),
                    "entry_id": dialogue.id,
                    "details": {
                        "required_ratio": requested_ratio,
                        "max_speedup": config.max_speedup,
                        "window_duration_s": dialogue.window_s,
                        "trimmed_duration_s": trimmed_frames / float(SAMPLE_RATE),
                    },
                }
                adaptations.append(
                    DialogueAdaptation(
                        dialogue=dialogue,
                        audio_path=path,
                        audio_sha256=audio_hash,
                        source=source,
                        window_start_frame=window_start,
                        window_end_frame=window_end,
                        trim=trim,
                        stretched_samples=None,
                        final_samples=None,
                        final_start_frame=0,
                        final_end_frame=0,
                        requested_ratio=requested_ratio,
                        applied_ratio=0.0,
                        unadaptable=True,
                        error=error,
                        max_speedup=config.max_speedup,
                    )
                )
                continue

            stretched = time_stretch(
                trimmed_samples,
                ratio=requested_ratio,
                sample_rate=SAMPLE_RATE,
                ffmpeg_bin=config.ffmpeg_bin,
                entry_id=dialogue.id,
                audio_path=str(path),
            )
            crop_frames = max(0, int(stretched.shape[0]) - window_frames)
            if stretched.shape[0] > window_frames:
                final_samples = stretched[:window_frames].copy()
            elif stretched.shape[0] < window_frames:
                pad_frames = window_frames - int(stretched.shape[0])
                final_samples = np.pad(stretched, (0, pad_frames))
            else:
                final_samples = stretched.copy()
            final_end = window_start + window_frames
            adaptations.append(
                DialogueAdaptation(
                    dialogue=dialogue,
                    audio_path=path,
                    audio_sha256=audio_hash,
                    source=source,
                    window_start_frame=window_start,
                    window_end_frame=window_end,
                    trim=trim,
                    stretched_samples=stretched,
                    final_samples=final_samples,
                    final_start_frame=window_start,
                    final_end_frame=final_end,
                    requested_ratio=requested_ratio,
                    applied_ratio=requested_ratio,
                    unadaptable=False,
                    error=None,
                    max_speedup=config.max_speedup,
                    atempo_crop_frames=crop_frames,
                    atempo_pad_frames=max(0, window_frames - int(stretched.shape[0])),
                )
            )
        else:
            final_end = window_start + trimmed_frames
            adaptations.append(
                DialogueAdaptation(
                    dialogue=dialogue,
                    audio_path=path,
                    audio_sha256=audio_hash,
                    source=source,
                    window_start_frame=window_start,
                    window_end_frame=window_end,
                    trim=trim,
                    stretched_samples=trimmed_samples,
                    final_samples=trimmed_samples.copy(),
                    final_start_frame=window_start,
                    final_end_frame=final_end,
                    requested_ratio=1.0,
                    applied_ratio=1.0,
                    unadaptable=False,
                    error=None,
                    max_speedup=config.max_speedup,
                )
            )
    return adaptations

