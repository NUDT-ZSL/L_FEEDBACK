"""Timeline assembly: event alignment, fades, crossfades and gain staging."""

import numpy as np

from . import loudness
from .timeline import frame_to_sample


class Segment:
    """A slice of track audio anchored to an absolute timeline position."""

    def __init__(self, event, abs_sample, src_start, src_end):
        self.event = event
        self.abs_sample = abs_sample
        self.src_start = src_start
        self.src_end = src_end

    @property
    def length(self):
        return self.src_end - self.src_start


def build_segments(events, audio_length, frame_rate, sample_rate):
    """Turn a track's event list into timeline segments.

    Each event anchors ``track_sample`` of the source audio to the absolute
    sample position derived from its video frame. A segment spans from its
    event's ``track_sample`` up to the next event's ``track_sample`` (or the
    end of the decoded audio). Events are processed in frame order.
    """
    ordered = sorted(
        events,
        key=lambda ev: (frame_to_sample(ev["frame"], frame_rate, sample_rate),
                        ev["track_sample"]),
    )
    segments = []
    for index, event in enumerate(ordered):
        abs_sample = frame_to_sample(event["frame"], frame_rate, sample_rate)
        src_start = min(event["track_sample"], audio_length)
        if index + 1 < len(ordered):
            src_end = min(ordered[index + 1]["track_sample"], audio_length)
        else:
            src_end = audio_length
        if src_end <= src_start:
            continue
        segments.append(Segment(event, abs_sample, src_start, src_end))
    return segments


def apply_fades(data, fade_samples):
    """Apply linear fade-in/out to a segment, in place, per sample.

    When the segment is shorter than twice the fade length, the fade is
    reduced to half the segment length and never crosses the boundaries.
    """
    length = data.shape[0]
    if length == 0 or fade_samples <= 0:
        return
    fade = min(fade_samples, length // 2)
    if fade <= 0:
        return
    ramp_in = np.linspace(0.0, 1.0, num=fade, endpoint=False)[:, None]
    ramp_out = np.linspace(1.0, 0.0, num=fade, endpoint=False)[:, None]
    data[:fade] *= ramp_in
    data[length - fade:] *= ramp_out


def assemble_track(audio, segments, crossfade_samples, fade_samples, channels):
    """Place a track's segments on the output timeline.

    Overlapping heads/tails of consecutive segments are blended with a
    linear crossfade of ``crossfade_samples`` (shrunk to the actual overlap
    when shorter). Non-overlapping regions are concatenated directly, with
    silence filling any timeline gap. Per-segment linear fades are applied
    before placement.
    """
    if not segments:
        return np.zeros((0, channels), dtype=np.float64)

    buffer = None
    cursor = 0
    for segment in segments:
        data = audio[segment.src_start:segment.src_end].copy()
        apply_fades(data, fade_samples)
        start = segment.abs_sample
        if buffer is None or start >= cursor:
            pad = start if buffer is None else start - cursor
            prefix = np.zeros((pad, channels), dtype=np.float64)
            parts = [prefix, data] if buffer is None else [buffer, prefix, data]
            buffer = np.vstack(parts)
            cursor = start + data.shape[0]
            continue

        overlap = cursor - start
        fade_len = min(crossfade_samples, overlap, data.shape[0])
        if fade_len > 0:
            outgoing = buffer[cursor - fade_len:cursor]
            incoming = data[overlap - fade_len:overlap]
            ramp_down = np.linspace(1.0, 0.0, num=fade_len, endpoint=False)[:, None]
            ramp_up = np.linspace(0.0, 1.0, num=fade_len, endpoint=False)[:, None]
            buffer[cursor - fade_len:cursor] = outgoing * ramp_down + incoming * ramp_up
        tail = data[overlap:]
        if tail.shape[0] > 0:
            buffer = np.vstack([buffer, tail])
            cursor += tail.shape[0]
    return buffer


def clamp_db(value, limits):
    return max(limits["min"], min(limits["max"], value))


def mix_tracks(track_buffers, limits):
    """Compute per-track gains and sum the tracks.

    Per-track gains pull every track to the mean track loudness (relative
    spread within +/-2 LU when unclamped); each gain is clamped to
    ``limits``. Returns (mix, per-track report list).
    """
    measured = [loudness.integrated_loudness(buf) for buf in track_buffers]
    mean_loudness = float(np.mean(measured)) if measured else 0.0

    track_reports = []
    gained = []
    for buffer, loudness_lufs in zip(track_buffers, measured):
        gain_db = clamp_db(mean_loudness - loudness_lufs, limits)
        gained.append(buffer * (10.0 ** (gain_db / 20.0)))
        track_reports.append({
            "loudness_lufs": round(loudness_lufs, 3),
            "gain_db": round(gain_db, 3),
        })

    total_length = max((b.shape[0] for b in gained), default=0)
    channels = gained[0].shape[1] if gained else 2
    mix = np.zeros((total_length, channels), dtype=np.float64)
    for buffer in gained:
        mix[:buffer.shape[0]] += buffer
    return mix, track_reports


def normalize_mix(mix, target_lufs, limits):
    """Apply the master gain so the mix hits ``target_lufs`` (clamped)."""
    current = loudness.integrated_loudness(mix)
    gain_db = clamp_db(target_lufs - current, limits)
    normalized = mix * (10.0 ** (gain_db / 20.0))
    np.clip(normalized, -1.0, 1.0, out=normalized)
    report = {
        "loudness_before_lufs": round(current, 3),
        "gain_db": round(gain_db, 3),
        "loudness_after_lufs": round(loudness.integrated_loudness(normalized), 3),
    }
    return normalized, report
