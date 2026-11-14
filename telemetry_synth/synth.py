"""Streaming engine, tire and event bus synthesis."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from . import CHANNELS, SAMPLE_RATE
from .errors import SynthError
from .schema import ParamsModel, TelemetryEvent, TelemetryModel
from .util import (
    deterministic_noise,
    linear_frame_interp,
    round_half_up,
    soft_limit,
)


@dataclass
class EngineVoice:
    phase: np.ndarray
    frozen_rpm: float | None
    static_gain: float = 1.0
    fade_in_start: int | None = None
    fade_out_start: int | None = None

    def gain_at(self, sample_index: int, fade_samples: int) -> float:
        gain = self.static_gain
        if self.fade_in_start is not None:
            progress = min(1.0, max(0.0, (sample_index - self.fade_in_start) / fade_samples))
            gain *= progress
        if self.fade_out_start is not None:
            progress = min(1.0, max(0.0, (sample_index - self.fade_out_start) / fade_samples))
            gain *= 1.0 - progress
        return gain

    def gain_vector(self, sample_start: int, count: int, fade_samples: int) -> np.ndarray:
        indices = sample_start + np.arange(count, dtype=np.float64)
        gain = np.full(count, self.static_gain, dtype=np.float64)
        if self.fade_in_start is not None:
            progress = np.clip(
                (indices - self.fade_in_start) / float(fade_samples), 0.0, 1.0
            )
            gain *= progress
        if self.fade_out_start is not None:
            progress = np.clip(
                (indices - self.fade_out_start) / float(fade_samples), 0.0, 1.0
            )
            gain *= 1.0 - progress
        return gain


@dataclass(frozen=True)
class EngineShift:
    sample: int
    previous_rpm: float
    new_rpm: float
    frame_index: int


class EngineSynth:
    """Phase-continuous harmonic engine with audio-domain shift crossfades."""

    def __init__(self, telemetry: TelemetryModel, params: ParamsModel, seed: int) -> None:
        self.telemetry = telemetry
        self.model = params.engine
        self.seed = seed
        self.orders = self.model.orders
        self.harmonic_gains = self.model.gains
        self.harmonic_count = len(self.orders)
        self.fade_samples = max(
            1, round_half_up(self.model.shift_crossfade_ms / 1000.0 * SAMPLE_RATE)
        )
        self.nyquist = SAMPLE_RATE / 2.0
        self.shifts = self._find_shifts()
        self.shift_cursor = 0
        self.shift_count = 0
        self.output_gain = 1.0

        phases = np.array(
            [harmonic.phase_rad for harmonic in self.model.harmonics],
            dtype=np.float64,
        )
        spread_rad = math.radians(self.model.stereo_phase_spread_deg)
        channel_phases = np.stack(
            [phases, phases + spread_rad],
            axis=1,
        )
        self.voices: list[EngineVoice] = [
            EngineVoice(phase=channel_phases, frozen_rpm=None)
        ]

    def _find_shifts(self) -> list[EngineShift]:
        rpm = self.telemetry.rpm
        shifts: list[EngineShift] = []
        for index in range(1, len(rpm)):
            delta = rpm[index] - rpm[index - 1]
            if abs(delta) >= self.model.shift_rpm_threshold:
                shifts.append(
                    EngineShift(
                        sample=round_half_up(
                            index * self.telemetry.samples_per_frame
                        ),
                        previous_rpm=float(rpm[index - 1]),
                        new_rpm=float(rpm[index]),
                        frame_index=index,
                    )
                )
        return shifts

    def _prune_voices(self, sample_after: int) -> None:
        kept: list[EngineVoice] = []
        fade_end = sample_after - self.fade_samples
        for voice in self.voices:
            if voice.fade_out_start is not None and voice.fade_out_start + self.fade_samples <= sample_after:
                continue
            if voice.fade_in_start is not None and voice.fade_in_start + self.fade_samples <= sample_after:
                voice.fade_in_start = None
            kept.append(voice)
        if not kept:
            # This should be unreachable because every shift creates a new
            # fading-in voice. Retain the last voice rather than produce silence.
            kept = [self.voices[-1]]
        self.voices = kept

    def _apply_shift(self, shift: EngineShift) -> None:
        sample = shift.sample
        current_gains = [
            max(0.0, voice.gain_at(sample, self.fade_samples))
            for voice in self.voices
        ]
        dominant_index = int(np.argmax(current_gains)) if current_gains else 0
        copied_phase = self.voices[dominant_index].phase.copy()

        for voice, gain in zip(self.voices, current_gains):
            voice.static_gain = gain
            voice.fade_in_start = None
            voice.fade_out_start = sample if gain > 1e-12 else voice.fade_out_start
            if voice.frozen_rpm is None:
                voice.frozen_rpm = shift.previous_rpm

        self.voices.append(
            EngineVoice(
                phase=copied_phase,
                frozen_rpm=None,
                static_gain=1.0,
                fade_in_start=sample,
            )
        )
        self.voices = [voice for voice in self.voices if voice.static_gain > 1e-12]
        self.shift_count += 1

    def _render_segment(self, sample_start: int, count: int) -> np.ndarray:
        rpm = linear_frame_interp(
            self.telemetry.rpm,
            sample_start,
            count,
            self.telemetry.samples_per_frame,
        )
        throttle = linear_frame_interp(
            self.telemetry.throttle,
            sample_start,
            count,
            self.telemetry.samples_per_frame,
        )
        amplitude = (
            self.model.base_amplitude
            + self.model.throttle_amplitude * throttle
        )
        amplitude_h = amplitude[:, None] * self.harmonic_gains[None, :]
        output = np.zeros((count, CHANNELS), dtype=np.float64)

        for voice in self.voices:
            if voice.frozen_rpm is None:
                rpm_series = rpm
            else:
                rpm_series = np.full(count, voice.frozen_rpm, dtype=np.float64)
            frequencies = (
                rpm_series[:, None] / 60.0
                * self.model.firing_order
                * self.orders[None, :]
            )
            harmonic_fade = np.clip(
                (self.nyquist * 0.98 - frequencies) / (self.nyquist * 0.02),
                0.0,
                1.0,
            )
            safe_frequencies = np.minimum(frequencies, self.nyquist * 0.98)
            increments = 2.0 * math.pi * safe_frequencies / SAMPLE_RATE
            cumulative = np.cumsum(increments, axis=0)
            offsets = np.concatenate(
                [np.zeros((1, self.harmonic_count), dtype=np.float64), cumulative[:-1]],
                axis=0,
            )
            phases = voice.phase[None, :, :] + offsets[:, :, None]
            waveform = np.sin(phases)
            voice_signal = np.sum(
                waveform
                * amplitude_h[:, :, None]
                * harmonic_fade[:, :, None],
                axis=1,
            )
            voice.phase = voice.phase + cumulative[-1][:, None]
            gains = voice.gain_vector(sample_start, count, self.fade_samples)
            output += voice_signal * gains[:, None]
        return output

    def render_block(self, sample_start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, CHANNELS), dtype=np.float64)
        block_end = sample_start + count
        output = np.zeros((count, CHANNELS), dtype=np.float64)
        cursor = sample_start

        while cursor < block_end:
            if (
                self.shift_cursor < len(self.shifts)
                and self.shifts[self.shift_cursor].sample < cursor
            ):
                raise SynthError(
                    "E_SYNTH",
                    "engine shift cursor fell behind the stream position",
                )
            if (
                self.shift_cursor < len(self.shifts)
                and self.shifts[self.shift_cursor].sample == cursor
            ):
                self._apply_shift(self.shifts[self.shift_cursor])
                self.shift_cursor += 1
                continue

            segment_end = block_end
            if self.shift_cursor < len(self.shifts):
                segment_end = min(segment_end, self.shifts[self.shift_cursor].sample)
            if segment_end <= cursor:
                raise SynthError("E_SYNTH", "engine crossfade segmentation stalled")
            segment = self._render_segment(cursor, segment_end - cursor)
            offset = cursor - sample_start
            output[offset : offset + len(segment)] = segment
            cursor = segment_end

        self._prune_voices(sample_start + count)
        return output


class TireSynth:
    """Slip-gated, surface-aware deterministic filtered noise."""

    def __init__(self, telemetry: TelemetryModel, params: ParamsModel, seed: int) -> None:
        self.telemetry = telemetry
        self.model = params.tires
        self.seed = seed
        self.filter_state = np.zeros(CHANNELS, dtype=np.float64)
        self.state_switch_count = 0
        self._build_frame_profiles()

    def _build_frame_profiles(self) -> None:
        profiles = [self.model.profiles[name] for name in self.telemetry.surfaces]
        grip_cutoff = np.array(
            [profiles[index].grip_cutoff_hz for index in self.telemetry.surface_indices],
            dtype=np.float64,
        )
        grip_gain = np.array(
            [profiles[index].grip_gain for index in self.telemetry.surface_indices],
            dtype=np.float64,
        )
        slip_cutoff = np.array(
            [profiles[index].slip_cutoff_hz for index in self.telemetry.surface_indices],
            dtype=np.float64,
        )
        slip_gain = np.array(
            [profiles[index].slip_gain for index in self.telemetry.surface_indices],
            dtype=np.float64,
        )

        dt = self.telemetry.frame_interval_s
        attack_coefficient = 1.0 - math.exp(-dt / (self.model.attack_ms / 1000.0))
        release_coefficient = 1.0 - math.exp(-dt / (self.model.release_ms / 1000.0))
        release_threshold = self.model.slip_threshold * self.model.release_ratio
        active = False
        envelope = 0.0
        state_envelopes = np.empty(self.telemetry.frame_count, dtype=np.float64)

        for index, slip in enumerate(self.telemetry.slip):
            if not active and slip >= self.model.slip_threshold:
                active = True
                self.state_switch_count += 1
            elif active and slip <= release_threshold:
                active = False
                self.state_switch_count += 1
            target = 1.0 if active else 0.0
            coefficient = (
                attack_coefficient if target > envelope else release_coefficient
            )
            envelope += (target - envelope) * coefficient
            state_envelopes[index] = envelope

        load = np.clip(
            self.telemetry.slip / self.model.slip_full_scale,
            0.0,
            1.0,
        )
        slip_blend = state_envelopes * (
            self.model.slip_load_floor
            + (1.0 - self.model.slip_load_floor) * load
        )
        self.frame_cutoff = grip_cutoff + (slip_cutoff - grip_cutoff) * slip_blend
        self.frame_gain = (
            grip_gain + (slip_gain - grip_gain) * slip_blend
        ) * self.model.level

    def render_block(self, sample_start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, CHANNELS), dtype=np.float64)
        cutoff = linear_frame_interp(
            self.frame_cutoff,
            sample_start,
            count,
            self.telemetry.samples_per_frame,
        )
        cutoff = np.clip(cutoff, 10.0, SAMPLE_RATE * 0.49)
        gain = linear_frame_interp(
            self.frame_gain,
            sample_start,
            count,
            self.telemetry.samples_per_frame,
        )
        noise = deterministic_noise(
            self.seed,
            "tire_noise",
            sample_start,
            count,
            CHANNELS,
        )
        alpha = 1.0 - np.exp(-2.0 * math.pi * cutoff / SAMPLE_RATE)
        output = np.empty((count, CHANNELS), dtype=np.float64)
        state = self.filter_state
        for index in range(count):
            state += alpha[index] * (noise[index] - state)
            output[index] = state * gain[index]
        self.filter_state = state
        return output


@dataclass(frozen=True)
class ScheduledEvent:
    source: TelemetryEvent
    duration_samples: int
    gain_factor: float

    @property
    def sample_start(self) -> int:
        return self.source.sample_start

    @property
    def sample_end(self) -> int:
        return self.source.sample_start + self.duration_samples


class EventSynth:
    """Half-frame-aligned event envelopes with deterministic overlap ducking."""

    def __init__(self, telemetry: TelemetryModel, params: ParamsModel, seed: int) -> None:
        self.telemetry = telemetry
        self.model = params.events
        self.seed = seed
        source_events = sorted(
            enumerate(telemetry.events),
            key=lambda pair: (pair[1].sample_start, pair[0]),
        )
        window_samples = round_half_up(
            self.model.overlap_window_ms / 1000.0 * SAMPLE_RATE
        )
        recent: list[int] = []
        scheduled: list[ScheduledEvent] = []
        attenuated_count = 0
        for _, event in source_events:
            while recent and event.sample_start - recent[0] > window_samples:
                recent.pop(0)
            overlap_count = len(recent)
            factor = self.model.overlap_attenuation ** overlap_count
            if overlap_count:
                attenuated_count += 1
            type_model = self.model.types[event.event_type]
            duration_samples = max(
                1,
                round_half_up(type_model.duration_ms / 1000.0 * SAMPLE_RATE),
            )
            scheduled.append(
                ScheduledEvent(
                    source=event,
                    duration_samples=duration_samples,
                    gain_factor=factor,
                )
            )
            recent.append(event.sample_start)
        self.events = scheduled
        self.starts = [event.sample_start for event in scheduled]
        self.max_duration = max(
            (event.duration_samples for event in scheduled), default=0
        )
        self.trigger_count = len(scheduled)
        self.attenuated_count = attenuated_count
        self.counts_by_type: dict[str, int] = {}
        for event in scheduled:
            event_type = event.source.event_type
            self.counts_by_type[event_type] = self.counts_by_type.get(event_type, 0) + 1

    def _render_event(
        self,
        event: ScheduledEvent,
        ordinal: int,
        global_start: int,
        count: int,
    ) -> np.ndarray:
        type_model = self.model.types[event.source.event_type]
        duration_s = event.duration_samples / SAMPLE_RATE
        attack_s = type_model.attack_ms / 1000.0
        decay_s = type_model.decay_ms / 1000.0
        local_t = (
            global_start
            + np.arange(count, dtype=np.float64)
            - event.sample_start
        ) / SAMPLE_RATE

        if attack_s > 0.0:
            attack = np.where(
                local_t < attack_s,
                0.5 - 0.5 * np.cos(math.pi * np.clip(local_t / attack_s, 0.0, 1.0)),
                1.0,
            )
        else:
            attack = np.ones(count, dtype=np.float64)
        decay = np.exp(-np.maximum(local_t - attack_s, 0.0) / decay_s)
        tail_length = min(0.01, duration_s * 0.25)
        tail_start = duration_s - tail_length
        if tail_length > 0.0:
            tail_progress = np.clip(
                (local_t - tail_start) / tail_length,
                0.0,
                1.0,
            )
            tail = 0.5 + 0.5 * np.cos(math.pi * tail_progress)
        else:
            tail = np.ones(count, dtype=np.float64)
        envelope = attack * decay * tail * event.source.intensity * event.gain_factor

        frequency_delta = type_model.frequency_end_hz - type_model.frequency_hz
        phase = (
            2.0
            * math.pi
            * (
                type_model.frequency_hz * local_t
                + 0.5 * frequency_delta * local_t * local_t / duration_s
            )
            + (ordinal * 0.6180339887498949)
        )
        if type_model.waveform == "impact":
            tone = np.sin(phase) + 0.28 * np.sin(2.0 * phase + 0.35)
        else:
            tone = np.sin(phase)

        noise = deterministic_noise(
            self.seed,
            f"event:{ordinal}:{event.source.event_type}",
            global_start,
            count,
            CHANNELS,
        )
        signal = (
            (1.0 - type_model.noise_mix) * tone[:, None]
            + type_model.noise_mix * noise
        )
        signal *= envelope[:, None] * type_model.gain
        left_gain = 1.0 - max(0.0, event.source.pan)
        right_gain = 1.0 + min(0.0, event.source.pan)
        signal[:, 0] *= left_gain
        signal[:, 1] *= right_gain
        return signal

    def render_block(self, sample_start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, CHANNELS), dtype=np.float64)
        block_end = sample_start + count
        output = np.zeros((count, CHANNELS), dtype=np.float64)
        if not self.events:
            return output
        search_from = sample_start - self.max_duration
        first = bisect.bisect_left(self.starts, search_from)
        index = first
        while index < len(self.events) and self.events[index].sample_start < block_end:
            event = self.events[index]
            event_start = max(sample_start, event.sample_start)
            event_end = min(block_end, event.sample_end)
            if event_end > event_start:
                offset = event_start - sample_start
                local_count = event_end - event_start
                rendered = self._render_event(
                    event,
                    index,
                    event_start,
                    local_count,
                )
                output[offset : offset + local_count] += rendered
            index += 1
        return output


@dataclass
class SynthStats:
    bus_peaks: dict[str, float] = field(
        default_factory=lambda: {"engine": 0.0, "tire": 0.0, "events": 0.0}
    )
    pre_limit_peak: float = 0.0
    post_limit_peak: float = 0.0
    engine_shift_count: int = 0
    tire_state_switch_count: int = 0
    event_trigger_count: int = 0
    events_attenuated: int = 0
    event_counts_by_type: dict[str, int] = field(default_factory=dict)


class AudioSynthesizer:
    def __init__(
        self,
        telemetry: TelemetryModel,
        params: ParamsModel,
        seed: int,
        block_size: int,
    ) -> None:
        if block_size <= 0:
            raise SynthError("E_USAGE", "block size must be greater than zero")
        self.telemetry = telemetry
        self.params = params
        self.block_size = block_size
        self.engine = EngineSynth(telemetry, params, seed)
        self.tire = TireSynth(telemetry, params, seed)
        self.events = EventSynth(telemetry, params, seed)
        self.stats = SynthStats()

    def iter_float_blocks(self):
        written = 0
        while written < self.telemetry.total_samples:
            count = min(self.block_size, self.telemetry.total_samples - written)
            engine_bus = (
                self.engine.render_block(written, count)
                * self.params.mix.engine_gain
            )
            tire_bus = (
                self.tire.render_block(written, count)
                * self.params.mix.tire_gain
            )
            event_bus = (
                self.events.render_block(written, count)
                * self.params.mix.event_gain
            )

            engine_peak = float(np.max(np.abs(engine_bus))) if count else 0.0
            tire_peak = float(np.max(np.abs(tire_bus))) if count else 0.0
            event_peak = float(np.max(np.abs(event_bus))) if count else 0.0
            self.stats.bus_peaks["engine"] = max(
                self.stats.bus_peaks["engine"], engine_peak
            )
            self.stats.bus_peaks["tire"] = max(
                self.stats.bus_peaks["tire"], tire_peak
            )
            self.stats.bus_peaks["events"] = max(
                self.stats.bus_peaks["events"], event_peak
            )

            mixed = (
                engine_bus + tire_bus + event_bus
            ) * self.params.mix.master_gain
            self.stats.pre_limit_peak = max(
                self.stats.pre_limit_peak,
                float(np.max(np.abs(mixed))),
            )
            limited = soft_limit(
                mixed,
                self.params.mix.soft_limit_threshold,
            )
            self.stats.post_limit_peak = max(
                self.stats.post_limit_peak,
                float(np.max(np.abs(limited))),
            )
            yield limited
            written += count

        self.stats.engine_shift_count = self.engine.shift_count
        self.stats.tire_state_switch_count = self.tire.state_switch_count
        self.stats.event_trigger_count = self.events.trigger_count
        self.stats.events_attenuated = self.events.attenuated_count
        self.stats.event_counts_by_type = dict(
            sorted(self.events.counts_by_type.items())
        )

