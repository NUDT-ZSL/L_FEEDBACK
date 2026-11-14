"""Block-based deterministic multi-channel mixing engine."""

import math
import struct
from array import array

from .synthesis import SourceCache, cue_motif, ptt_burst


def db_to_linear(gain_db):
    return 10.0 ** (gain_db / 20.0)


def soft_clip(value):
    """Smooth tanh soft limiter, identity near zero, asymptotic at +/-1."""
    return math.tanh(value)


def _pan_gains(pan):
    angle = (pan + 1.0) * math.pi * 0.25
    return math.cos(angle), math.sin(angle)


class RenderEngine:
    def __init__(self, project, timeline, stereo=False):
        self.project = project
        self.timeline = timeline
        self.sample_rate = project.sample_rate
        self.stereo = stereo
        self.channels = project.channels
        self.source_cache = SourceCache(self.sample_rate)
        self.channel_gain = {
            cid: db_to_linear(channel.gain_db)
            for cid, channel in project.channels.items()
        }
        self.speaker_pan = {
            sid: _pan_gains(speaker.pan)
            for sid, speaker in project.speakers.items()
        }
        self.fragment_pos = {sid: 0 for sid in project.speakers}
        self.ptt_pos = {sid: 0 for sid in project.speakers}
        self.fragments_by_speaker = {sid: [] for sid in project.speakers}
        for fragment in timeline.fragments:
            self.fragments_by_speaker[fragment.speaker_id].append(fragment)
        self.ptt_by_speaker = {sid: [] for sid in project.speakers}
        for burst in timeline.ptt_bursts:
            self.ptt_by_speaker[burst.speaker_id].append(burst)
        self.intro = self._intro_buffer()
        self.outro = self._outro_buffer()
        self._ptt_cache = {}
        self.total_samples = self._compute_total_samples()
        self.stats = {
            "peak_premix": 0.0,
            "peak_pre_soft": 0.0,
            "peak_post_soft": 0.0,
            "clipped_frames": 0,
            "clipped_samples": 0,
        }

    def seek_to_sample(self, sample_position):
        """Advance fragment/PTT cursors past already persisted samples."""
        for speaker_id, fragments in self.fragments_by_speaker.items():
            position = 0
            while position < len(fragments) and fragments[position].start_sample < sample_position:
                position += 1
            self.fragment_pos[speaker_id] = position
        for speaker_id, bursts in self.ptt_by_speaker.items():
            position = 0
            while position < len(bursts) and bursts[position].start_sample < sample_position:
                position += 1
            self.ptt_pos[speaker_id] = position

    def _intro_buffer(self):
        if not self.project.export.enable_prompts:
            return None
        return cue_motif(
            self.project.export.intro_ms,
            self.sample_rate,
            seed=0xC0FFEE,
            intro=True,
        )

    def _outro_buffer(self):
        if not self.project.export.enable_prompts:
            return None
        return cue_motif(
            self.project.export.outro_ms,
            self.sample_rate,
            seed=0xBADC0DE,
            intro=False,
        )

    def outro_extension(self):
        if self.outro is None:
            return 0
        natural_end = self.timeline.last_voice_end
        start = natural_end
        end = start + len(self.outro)
        return max(0, end - natural_end)

    def _compute_total_samples(self):
        natural_end = self.timeline.last_voice_end
        total = natural_end
        if self.intro is not None:
            total = max(total, len(self.intro))
        if self.outro is not None:
            total = max(total, natural_end + len(self.outro))
        return max(total, 1)

    def render_chunk(self, chunk_start, chunk_end):
        length = chunk_end - chunk_start
        frame_count = 2 if self.stereo else 1
        mix = [array("d", [0.0]) * length for _ in range(frame_count)]
        source_peak = 0.0

        for speaker_id, speaker in self.project.speakers.items():
            left_gain, right_gain = self.speaker_pan[speaker_id]
            source_peak = max(
                source_peak,
                self._render_speech(speaker, chunk_start, chunk_end, mix, left_gain, right_gain),
            )
            source_peak = max(
                source_peak,
                self._render_ptt(speaker, chunk_start, chunk_end, mix, left_gain, right_gain),
            )

        self._overlay_cue(self.intro, 0, mix, chunk_start, chunk_end)
        if self.outro is not None:
            self._overlay_cue(
                self.outro,
                self.timeline.last_voice_end,
                mix,
                chunk_start,
                chunk_end,
            )

        return self._finalize_chunk(mix, source_peak)

    def _render_speech(self, speaker, chunk_start, chunk_end, mix, left_gain, right_gain):
        fragments = self.fragments_by_speaker[speaker.speaker_id]
        position = self.fragment_pos[speaker.speaker_id]
        source = self.source_cache.get(speaker)
        source_length = len(source)
        source_peak = 0.0
        stereo = self.stereo
        bus_left = mix[0] if stereo else None
        bus_right = mix[1] if stereo else None
        bus_mono = mix[0]
        channels = self.channels
        channel_gain = self.channel_gain
        while position < len(fragments):
            fragment = fragments[position]
            if fragment.start_sample >= chunk_end:
                break
            if fragment.end_sample <= chunk_start:
                position += 1
                continue
            channel = channels[fragment.channel_id]
            gain = channel_gain[fragment.channel_id]
            silent = channel.muted or fragment.blocked
            overlap_start = max(chunk_start, fragment.start_sample)
            overlap_end = min(chunk_end, fragment.end_sample)
            base_offset = fragment.source_offset - overlap_start + fragment.start_sample
            for sample_index in range(overlap_start, overlap_end):
                destination = sample_index - chunk_start
                if silent:
                    continue
                source_index = (base_offset + sample_index) % source_length
                value = source[source_index] * gain
                magnitude = abs(value)
                if magnitude > source_peak:
                    source_peak = magnitude
                if stereo:
                    bus_left[destination] += value * left_gain
                    bus_right[destination] += value * right_gain
                else:
                    bus_mono[destination] += value
            position += 1
        self.fragment_pos[speaker.speaker_id] = position
        return source_peak

    def _render_ptt(self, speaker, chunk_start, chunk_end, mix, left_gain, right_gain):
        bursts = self.ptt_by_speaker[speaker.speaker_id]
        position = self.ptt_pos[speaker.speaker_id]
        source_peak = 0.0
        while position < len(bursts):
            burst = bursts[position]
            burst_end = burst.start_sample + burst.length_samples
            if burst.start_sample >= chunk_end:
                break
            if burst_end <= chunk_start:
                position += 1
                continue
            channel = self.channels[burst.channel_id]
            gain = self.channel_gain[burst.channel_id]
            silent = channel.muted or burst.blocked
            cache_key = (speaker.speaker_id, burst.length_samples)
            if cache_key not in self._ptt_cache:
                self._ptt_cache[cache_key] = ptt_burst(
                    speaker, burst.length_samples, self.sample_rate
                )
            waveform = self._ptt_cache[cache_key]
            overlap_start = max(chunk_start, burst.start_sample)
            overlap_end = min(chunk_end, burst_end)
            for sample_index in range(overlap_start, overlap_end):
                destination = sample_index - chunk_start
                if silent:
                    continue
                wave_index = sample_index - burst.start_sample
                value = waveform[wave_index] * gain
                source_peak = max(source_peak, abs(value))
                self._add_to_mix(mix, destination, value, left_gain, right_gain)
            position += 1
        self.ptt_pos[speaker.speaker_id] = position
        return source_peak

    def _add_to_mix(self, mix, destination, value, left_gain, right_gain):
        if self.stereo:
            mix[0][destination] += value * left_gain
            mix[1][destination] += value * right_gain
        else:
            mix[0][destination] += value

    def _overlay_cue(self, cue, cue_origin, mix, chunk_start, chunk_end):
        if cue is None:
            return
        cue_start = cue_origin
        cue_end = cue_origin + len(cue)
        overlap_start = max(chunk_start, cue_start)
        overlap_end = min(chunk_end, cue_end)
        for sample_index in range(overlap_start, overlap_end):
            destination = sample_index - chunk_start
            value = cue[sample_index - cue_start]
            if self.stereo:
                mix[0][destination] += value
                mix[1][destination] += value
            else:
                mix[0][destination] += value

    def _finalize_chunk(self, mix, source_peak):
        length = len(mix[0])
        frame_count = len(mix)
        pre_soft_peak = 0.0
        post_soft_peak = 0.0
        clipped_frames = 0
        clipped_samples = 0
        packed = []
        pack = struct.Struct("<h").pack
        tanh = math.tanh
        for frame in range(length):
            frame_clipped = False
            for channel_index in range(frame_count):
                raw = mix[channel_index][frame]
                magnitude = abs(raw)
                if magnitude > pre_soft_peak:
                    pre_soft_peak = magnitude
                if magnitude > 1.0:
                    limited = tanh(raw)
                    clipped_samples += 1
                    frame_clipped = True
                else:
                    limited = raw
                limited_magnitude = abs(limited)
                if limited_magnitude > post_soft_peak:
                    post_soft_peak = limited_magnitude
                integer_sample = int(round(limited * 32767.0))
                if integer_sample > 32767:
                    integer_sample = 32767
                elif integer_sample < -32768:
                    integer_sample = -32768
                packed.append(pack(integer_sample))
            if frame_clipped:
                clipped_frames += 1

        self.stats["peak_premix"] = max(self.stats["peak_premix"], source_peak)
        self.stats["peak_pre_soft"] = max(self.stats["peak_pre_soft"], pre_soft_peak)
        self.stats["peak_post_soft"] = max(self.stats["peak_post_soft"], post_soft_peak)
        self.stats["clipped_frames"] += clipped_frames
        self.stats["clipped_samples"] += clipped_samples
        return b"".join(packed)
