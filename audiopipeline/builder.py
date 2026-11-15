"""Turn decoded source audio into a timeline-ready event segment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .fades import apply_edge_fades, loop_source, ms_to_samples
from .manifest import Event, ms_to_samples as manifest_ms


@dataclass
class BuiltSegment:
    audio: np.ndarray
    looped: bool = False
    trimmed: bool = False
    padded: bool = False
    requested_samples: int = 0


def build_segment(
    event: Event,
    source: np.ndarray,
    config: Config,
    timeline_end_sample: int,
) -> BuiltSegment:
    """Build the event segment honoring offset, duration, loop and fades."""
    offset_frames = manifest_ms(event.offset_ms)
    start_sample = manifest_ms(event.start_ms)
    available = source.shape[0] - offset_frames

    if event.duration_ms is not None:
        need = manifest_ms(event.duration_ms)
    elif event.loop:
        # A loop without explicit duration fills the rest of the timeline.
        need = max(0, timeline_end_sample - start_sample)
    else:
        need = available

    need = max(0, need)
    xf = ms_to_samples(config.crossfade_ms)
    head = ms_to_samples(config.head_fade_ms)
    tail = ms_to_samples(config.tail_fade_ms)
    body = source[offset_frames:]

    if event.loop:
        segment = loop_source(body, need, xf)
        built = BuiltSegment(
            audio=segment,
            looped=need > available,
            trimmed=True,
            requested_samples=need,
        )
        # The cut may land mid-waveform: always smooth the looped tail.
        built.audio = apply_edge_fades(built.audio, head, tail)
        return built

    if need < available:
        segment = body[:need].copy()
        built = BuiltSegment(audio=segment, trimmed=True, requested_samples=need)
        # Trimmed inside the source: cosine tail fade removes the click.
        built.audio = apply_edge_fades(built.audio, head, tail)
        return built

    segment = body.copy()
    padded = False
    if need > available:
        silence = np.zeros(
            (need - available, source.shape[1]), dtype=source.dtype
        )
        segment = np.concatenate([segment, silence], axis=0)
        padded = True
    built = BuiltSegment(
        audio=segment, padded=padded, requested_samples=need
    )
    # Natural source end: only the optional configured head fade applies.
    built.audio = apply_edge_fades(built.audio, head, 0)
    return built
