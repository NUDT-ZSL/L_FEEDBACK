from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ProcessingError
from .ffmpeg_tools import format_timecode
from .models import Manifest, MediaInfo


@dataclass
class EffectiveSwitch:
    original_index: int
    frame: int
    language_id: str
    sample: int
    implicit: bool = False


@dataclass
class SegmentPlan:
    index: int
    language_id: str
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    fade_in_samples: int = 0
    fade_out_samples: int = 0
    source_shortfall_samples: int = 0
    same_language_boundary_in: bool = False
    same_language_boundary_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_report(self, sample_rate: int, frame_rate: float) -> dict:
        nominal_samples = max(0, self.end_sample - self.start_sample)
        return {
            "index": self.index,
            "language": self.language_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "sample_count": nominal_samples,
            "start_timecode": format_timecode(self.start_frame / frame_rate, frame_rate),
            "end_timecode": format_timecode(self.end_frame / frame_rate, frame_rate),
            "fade_in_samples": self.fade_in_samples,
            "fade_out_samples": self.fade_out_samples,
            "fade_in_ms": round(self.fade_in_samples * 1000.0 / sample_rate, 3),
            "fade_out_ms": round(self.fade_out_samples * 1000.0 / sample_rate, 3),
            "source_shortfall_samples": self.source_shortfall_samples,
        }


@dataclass
class TimelinePlan:
    duration_frames: int
    total_samples: int
    segments: list[SegmentPlan]
    fallbacks: list[dict]
    effective_switches: list[EffectiveSwitch]
    crossfade_samples: int


def frames_to_samples(frame: int, frame_rate: float, sample_rate: int) -> int:
    return int(round(frame * sample_rate / frame_rate))


def plan_timeline(manifest: Manifest, media: MediaInfo) -> TimelinePlan:
    sample_rate = manifest.audio.sample_rate
    duration_frames = manifest.duration_frames
    if duration_frames is None:
        duration_frames = media.duration_frames(manifest.frame_rate)
    if duration_frames <= 0:
        raise ProcessingError(3040, "language WAVs contain no usable audio frames")
    total_samples = frames_to_samples(duration_frames, manifest.frame_rate, sample_rate)
    if total_samples <= 0:
        raise ProcessingError(3040, "manifest timeline maps to zero output samples")

    effective: list[EffectiveSwitch] = []
    fallbacks: list[dict] = []
    current_language: str | None = None
    for switch in manifest.switches:
        info = media.wavs[switch.language_id]
        sample = frames_to_samples(switch.frame, manifest.frame_rate, sample_rate)
        reason: str | None = None
        if switch.frame >= duration_frames:
            reason = "frame_outside_timeline"
        elif sample >= total_samples:
            reason = "frame_outside_timeline"
        elif sample >= info.data_frames:
            reason = "frame_out_of_track_range"

        if reason is not None:
            if current_language is None:
                raise ProcessingError(
                    3041,
                    f"initial switch frame {switch.frame} is outside language track "
                    f"{switch.language_id}",
                )
            fallbacks.append(
                {
                    "event": "switch_fallback",
                    "switch_index": switch.original_index,
                    "requested_language": switch.language_id,
                    "requested_frame": switch.frame,
                    "requested_timecode": format_timecode(
                        switch.frame / manifest.frame_rate, manifest.frame_rate
                    ),
                    "reason": reason,
                    "fallback_language": current_language,
                    "previous_valid_switch_frame": effective[-1].frame,
                }
            )
            continue

        effective.append(
            EffectiveSwitch(
                original_index=switch.original_index,
                frame=switch.frame,
                language_id=switch.language_id,
                sample=sample,
                implicit=switch.implicit,
            )
        )
        current_language = switch.language_id

    if not effective:
        raise ProcessingError(3041, "no valid language switch points remain")

    requested_fade = int(round(manifest.crossfade_ms * sample_rate / 1000.0))
    if requested_fade <= 0:
        raise ProcessingError(3042, "crossfade duration rounds to zero samples")

    segments: list[SegmentPlan] = []
    for index, switch in enumerate(effective):
        end_frame = (
            effective[index + 1].frame
            if index + 1 < len(effective)
            else duration_frames
        )
        start_sample = switch.sample
        end_sample = min(
            frames_to_samples(end_frame, manifest.frame_rate, sample_rate),
            total_samples,
        )
        if end_sample <= start_sample:
            if index == 0:
                raise ProcessingError(3041, "initial language segment has no samples")
            continue
        info = media.wavs[switch.language_id]
        shortfall = max(0, end_sample - info.data_frames)
        segments.append(
            SegmentPlan(
                index=len(segments),
                language_id=switch.language_id,
                start_frame=switch.frame,
                end_frame=end_frame,
                start_sample=start_sample,
                end_sample=end_sample,
                source_shortfall_samples=shortfall,
            )
        )

    if not segments:
        raise ProcessingError(3041, "timeline contains no usable language segments")

    for index in range(len(segments) - 1):
        left = segments[index]
        right = segments[index + 1]
        left_info = media.wavs[left.language_id]
        right_info = media.wavs[right.language_id]
        available = min(
            requested_fade,
            max(0, left_info.data_frames - left.end_sample),
            max(0, right_info.data_frames - right.start_sample),
            max(0, total_samples - right.start_sample),
        )
        left.fade_out_samples = available
        right.fade_in_samples = available
        same_language = left.language_id == right.language_id
        left.same_language_boundary_out = same_language
        right.same_language_boundary_in = same_language

    return TimelinePlan(
        duration_frames=duration_frames,
        total_samples=total_samples,
        segments=segments,
        fallbacks=fallbacks,
        effective_switches=effective,
        crossfade_samples=requested_fade,
    )


def _fade_envelopes(
    fade_in_samples: int,
    fade_out_samples: int,
    length: int,
    nominal_length: int,
    same_language_in: bool,
    same_language_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    fade_in = np.ones(length, dtype=np.float32)
    fade_out = np.ones(length, dtype=np.float32)
    if fade_in_samples > 0:
        t = (np.arange(fade_in_samples, dtype=np.float32) + 0.5) / fade_in_samples
        if same_language_in:
            fade_in[:fade_in_samples] = t
        else:
            fade_in[:fade_in_samples] = np.sin(t * np.float32(np.pi / 2))
    if fade_out_samples > 0:
        t = (np.arange(fade_out_samples, dtype=np.float32) + 0.5) / fade_out_samples
        values = t if same_language_out else np.cos(t * np.float32(np.pi / 2))
        fade_out[nominal_length : nominal_length + fade_out_samples] = values
    return fade_in, fade_out


def synthesize_timeline(
    manifest: Manifest,
    plan: TimelinePlan,
    language_audio: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    channels = manifest.audio.channels
    output = np.zeros((plan.total_samples, channels), dtype=np.float32)
    reports: list[dict] = []

    for segment in plan.segments:
        audio = language_audio[segment.language_id]
        if audio.ndim != 2 or audio.shape[1] != channels:
            raise ProcessingError(
                3043,
                f"language {segment.language_id} channel layout changed during synthesis",
            )
        nominal_length = segment.end_sample - segment.start_sample
        copy_length = min(
            nominal_length + segment.fade_out_samples,
            max(0, audio.shape[0] - segment.start_sample),
            max(0, output.shape[0] - segment.start_sample),
        )
        if copy_length <= 0:
            raise ProcessingError(
                3044,
                f"language segment {segment.index} ({segment.language_id}) has no source samples",
            )
        fade_in, fade_out = _fade_envelopes(
            segment.fade_in_samples,
            segment.fade_out_samples,
            copy_length,
            min(nominal_length, copy_length),
            segment.same_language_boundary_in,
            segment.same_language_boundary_out,
        )
        envelope = (fade_in * fade_out).astype(np.float32)
        source = audio[segment.start_sample : segment.start_sample + copy_length]
        output[segment.start_sample : segment.start_sample + copy_length] += source * envelope[:, None]
        report = segment.to_report(manifest.audio.sample_rate, manifest.frame_rate)
        report["rendered_samples"] = copy_length
        reports.append(report)

    mono = np.mean(output, axis=1, dtype=np.float32) if channels > 1 else output[:, 0]
    if not np.all(np.isfinite(mono)):
        raise ProcessingError(3045, "synthesized timeline contains NaN or infinity")
    return output, reports

