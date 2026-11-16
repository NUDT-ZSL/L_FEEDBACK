"""Chunked dynamic-layer mixing, normalization and deterministic WAV output."""

import numpy as np
import soundfile as sf

from .looping import build_layer_source

TARGET_PEAK_DBFS = -1.0
EDGE_FADE_MS = 10.0
CHUNK_SAMPLES = 1 << 16


class LayerMixer:
    """Mixes planned layer segments onto the output timeline."""

    def __init__(self, chart, beat_grid, planned_events, output_wav):
        self.chart = chart
        self.grid = beat_grid
        self.events = planned_events
        self.output_wav = output_wav
        self.total_samples = beat_grid.total_samples(chart.total_duration_ms)
        self.edge_fade = max(
            1, int(round(EDGE_FADE_MS * 0.001 * chart.sample_rate))
        )
        self.beat_period_samples = int(
            round(beat_grid.beat_period_s * chart.sample_rate)
        )

    def _segments_by_layer(self):
        """Turn the snapped event stream into gain envelopes (segments).

        Each segment is ``(start_sample, end_sample, gain)`` with linear
        edge fades applied while mixing. Repeated enable/disable events are
        idempotent; a gain change while enabled closes and reopens a segment
        so the two gains overlap-add with a cross-fade (no click).
        """
        segments = {name: [] for name in self.chart.layers}
        active = {}
        for event in self.events:
            name = event["layer"]
            sample = event["sample_index"]
            if event["action"] == "enable":
                if name in active:
                    start, old_gain = active.pop(name)
                    if start < sample and old_gain != event["gain"]:
                        segments[name].append((start, sample, old_gain))
                        start = sample
                else:
                    start = sample
                active[name] = (start, float(event["gain"]))
            else:  # disable
                if name in active:
                    start, gain = active.pop(name)
                    if start < sample:
                        segments[name].append((start, sample, gain))
        for name, (start, gain) in active.items():
            if start < self.total_samples:
                segments[name].append((start, self.total_samples, gain))
        clipped = {}
        for name, layer_segments in segments.items():
            cleaned = []
            for start, end, gain in layer_segments:
                start = max(0, min(start, self.total_samples))
                end = max(0, min(end, self.total_samples))
                if end > start:
                    cleaned.append((start, end, gain))
            clipped[name] = cleaned
        return clipped

    def _envelope(self, segment, chunk_start, chunk_end):
        start, end, gain = segment
        lo = max(start, chunk_start)
        hi = min(end, chunk_end)
        if lo >= hi:
            return None
        env = np.full(hi - lo, float(gain), dtype=np.float64)
        fade = min(self.edge_fade, (end - start) // 2)
        if fade > 0:
            ramp = (np.arange(1, fade + 1, dtype=np.float64) / (fade + 1))
            head = lo - start
            if head < fade:
                take = min(fade - head, env.size)
                env[:take] *= ramp[head:head + take]
            tail = end - hi
            if tail < fade:
                take = min(fade - tail, env.size)
                env[-take:] *= ramp[:take][::-1]
        return lo, hi, env

    def _build_sources(self):
        sources = {}
        for name, layer in self.chart.layers.items():
            sources[name] = build_layer_source(
                layer.samples,
                self.chart.crossfade_ms,
                self.chart.sample_rate,
                self.beat_period_samples,
                self.chart.loop_enabled,
            )
        return sources

    def _render_chunk(self, segments, sources, chunk_start, chunk_end):
        """Render one output chunk; chunk order is fixed for determinism."""
        count = chunk_end - chunk_start
        accum = np.zeros(count, dtype=np.float64)
        chunk_counts = {}
        for name in self.chart.layers:
            layer_chunk = np.zeros(count, dtype=np.float64)
            touched = False
            for segment in segments[name]:
                envelope = self._envelope(segment, chunk_start, chunk_end)
                if envelope is None:
                    continue
                lo, hi, env = envelope
                audio = sources[name].read(lo, hi - lo).astype(np.float64)
                layer_chunk[lo - chunk_start:hi - chunk_start] += audio * env
                touched = True
            if touched:
                chunk_counts[name] = int(np.count_nonzero(layer_chunk))
                accum += layer_chunk
        return accum, chunk_counts

    def measure_peak(self):
        """First pass: stream the whole mix, retaining only chunk buffers."""
        segments = self._segments_by_layer()
        sources = self._build_sources()
        mixed_counts = {name: 0 for name in self.chart.layers}
        peak = 0.0
        for chunk_start in range(0, self.total_samples, CHUNK_SAMPLES):
            chunk_end = min(chunk_start + CHUNK_SAMPLES, self.total_samples)
            accum, chunk_counts = self._render_chunk(
                segments, sources, chunk_start, chunk_end
            )
            for name, count in chunk_counts.items():
                mixed_counts[name] += count
            chunk_peak = float(np.max(np.abs(accum)))
            if chunk_peak > peak:
                peak = chunk_peak
        return peak, mixed_counts, segments, sources

    def write_normalized(self, segments, sources, peak):
        """Second pass: re-render chunk by chunk straight into the WAV."""
        target_linear = 10.0 ** (TARGET_PEAK_DBFS / 20.0)
        master_gain = target_linear / peak if peak > 0.0 else 1.0
        int16_max = float(np.iinfo(np.int16).max)
        try:
            with sf.SoundFile(
                self.output_wav,
                mode="w",
                samplerate=self.chart.sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as handle:
                for chunk_start in range(
                    0, self.total_samples, CHUNK_SAMPLES
                ):
                    chunk_end = min(
                        chunk_start + CHUNK_SAMPLES, self.total_samples
                    )
                    accum, _ = self._render_chunk(
                        segments, sources, chunk_start, chunk_end
                    )
                    scaled = np.clip(accum * master_gain, -1.0, 1.0)
                    pcm = np.rint(scaled * int16_max).astype(np.int16)
                    handle.write(pcm)
        except OSError as exc:
            from .errors import OutputNotWritableError

            raise OutputNotWritableError(
                f"cannot write output WAV {self.output_wav}: {exc}",
                detail={"path": self.output_wav},
            ) from exc
        return master_gain
