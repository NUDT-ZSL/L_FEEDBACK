"""Multitrack clip placement and equal-power crossfades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifest import Clip, Manifest


@dataclass(frozen=True)
class FadeInfo:
    edit_id: str
    track: str
    outgoing_clip: str
    incoming_clip: str
    at_sample: int
    requested_samples: int
    effective_samples: int
    fade_start: int
    fade_end: int

    @property
    def truncated(self) -> bool:
        return self.effective_samples < self.requested_samples


@dataclass
class TimelineResult:
    tracks: dict[str, np.ndarray]
    fades: list[FadeInfo]


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def equal_power_gains(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return outgoing/incoming cosine envelopes for an overlap of length N."""

    if length <= 0:
        raise ValueError("equal-power fade length must be positive")
    if length == 1:
        value = np.sqrt(0.5)
        return np.array([value]), np.array([value])
    phase = np.linspace(0.0, np.pi / 2.0, length, dtype=np.float64)
    return np.cos(phase), np.sin(phase)


def _clip_source(manifest: Manifest, clip: Clip) -> np.ndarray:
    samples, _ = manifest.assets[clip.asset]
    return samples[clip.source_in : clip.source_out]


def render_timeline(manifest: Manifest) -> TimelineResult:
    """Render tracks to float64 and apply explicit crossfades.

    Normal placement is manifest-order copy-on-write within a track. An
    explicit edit point then recombines exactly the two named clips over its
    overlap. Cross-track accumulation is performed by the caller.
    """

    buffers = {
        name: np.zeros((manifest.total_samples, manifest.channels), dtype=np.float64)
        for name in manifest.tracks
    }
    owners = {
        name: np.full(manifest.total_samples, -1, dtype=np.int32)
        for name in manifest.tracks
    }

    for clip in sorted(manifest.clips, key=lambda value: value.order):
        track = manifest.tracks[clip.track]
        gain = db_to_linear(track.gain_db + clip.gain_db)
        source = _clip_source(manifest, clip)
        target = buffers[clip.track]
        target[clip.timeline_in : clip.timeline_out] = source * gain
        owners[clip.track][clip.timeline_in : clip.timeline_out] = clip.order

    clip_by_id = {clip.clip_id: clip for clip in manifest.clips}
    fades: list[FadeInfo] = []
    for edit in manifest.edit_points:
        outgoing = clip_by_id[edit.outgoing_clip]
        incoming = clip_by_id[edit.incoming_clip]
        available = edit.at_sample - max(
            outgoing.timeline_in,
            incoming.timeline_in,
            edit.at_sample - edit.fade_samples,
        )
        effective = min(edit.fade_samples, available)
        if effective <= 0:
            raise ValueError(f"edit point {edit.edit_id} has no fade overlap")

        start = edit.at_sample - effective
        out_gain, in_gain = equal_power_gains(effective)
        out_source = _clip_source(manifest, outgoing)[
            start - outgoing.timeline_in : edit.at_sample - outgoing.timeline_in
        ]
        in_source = _clip_source(manifest, incoming)[
            start - incoming.timeline_in : edit.at_sample - incoming.timeline_in
        ]
        if out_source.shape[0] != effective or in_source.shape[0] != effective:
            raise ValueError(f"edit point {edit.edit_id} extends beyond clip material")

        out_db = manifest.tracks[outgoing.track].gain_db + outgoing.gain_db
        in_db = manifest.tracks[incoming.track].gain_db + incoming.gain_db
        mixed = (
            out_source * (db_to_linear(out_db) * out_gain[:, None])
            + in_source * (db_to_linear(in_db) * in_gain[:, None])
        )
        # A clip written later on the same track still wins. The transition
        # only replaces content owned by the two clips it explicitly joins.
        transition_order = max(outgoing.order, incoming.order)
        owner_slice = owners[edit.track][start : edit.at_sample]
        replace = owner_slice <= transition_order
        target_slice = buffers[edit.track][start : edit.at_sample]
        target_slice[replace] = mixed[replace]
        owner_slice[replace] = transition_order
        fades.append(
            FadeInfo(
                edit_id=edit.edit_id,
                track=edit.track,
                outgoing_clip=edit.outgoing_clip,
                incoming_clip=edit.incoming_clip,
                at_sample=edit.at_sample,
                requested_samples=edit.fade_samples,
                effective_samples=effective,
                fade_start=start,
                fade_end=edit.at_sample,
            )
        )

    return TimelineResult(buffers, fades)


def mix_tracks(tracks: dict[str, np.ndarray]) -> np.ndarray:
    if not tracks:
        raise ValueError("at least one track is required")
    names = sorted(tracks)
    mixture = np.zeros_like(tracks[names[0]], dtype=np.float64)
    for name in names:
        mixture += tracks[name]
    return mixture

