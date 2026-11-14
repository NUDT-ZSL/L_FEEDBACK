"""Deterministic PCM synthesis, anti-aliased pitch compensation and cues."""

import math
from array import array

TWO_PI = 2.0 * math.pi


class LcgRandom:
    """Deterministic 32-bit LCG so output is identical on every machine."""

    MODULUS = 2147483647
    MULTIPLIER = 1103515245
    INCREMENT = 12345

    def __init__(self, seed):
        self.state = int(seed) & 0x7FFFFFFF

    def next_uint(self):
        self.state = (
            self.MULTIPLIER * self.state + self.INCREMENT
        ) % self.MODULUS
        return self.state

    def next_uniform(self):
        return self.next_uint() / float(self.MODULUS)


def source_waveform(speaker, sample_rate):
    """Synthesize the fixed-length source pulse train for one speaker."""
    length = max(1, int(round(speaker.source_duration_ms * sample_rate / 1000.0)))
    rng = LcgRandom(speaker.seed * 2654435761 & 0x7FFFFFFF)
    base = speaker.base_frequency
    harmonic_ratio = speaker.harmonic_ratio
    phase = 0.0
    harmonic_phase = 0.0
    tremolo_phase = 0.0
    phase_step = TWO_PI * base / sample_rate
    harmonic_step = TWO_PI * base * 2.0 / sample_rate
    tremolo_step = TWO_PI * 5.5 / sample_rate
    samples = array("f", [0.0]) * length
    amplitude = 0.5
    for index in range(length):
        fundamental = math.sin(phase)
        harmonic = harmonic_ratio * math.sin(harmonic_phase)
        tremolo = 0.9 + 0.1 * math.sin(tremolo_phase)
        jitter = (rng.next_uniform() - 0.5) * 0.01
        value = amplitude * (fundamental + harmonic) * tremolo + jitter
        samples[index] = value
        phase += phase_step
        if phase >= TWO_PI:
            phase -= TWO_PI
        harmonic_phase += harmonic_step
        if harmonic_phase >= TWO_PI:
            harmonic_phase -= TWO_PI
        tremolo_phase += tremolo_step
        if tremolo_phase >= TWO_PI:
            tremolo_phase -= TWO_PI
    return samples


def biquad_lowpass(samples, cutoff, sample_rate, q=0.7071067811865476):
    """Two cascaded RBJ biquad low-pass stages for anti-aliasing."""
    if cutoff <= 0.0 or cutoff >= sample_rate * 0.5:
        return array("f", samples)

    def stage(input_samples):
        omega = TWO_PI * cutoff / sample_rate
        sin_w = math.sin(omega)
        cos_w = math.cos(omega)
        alpha = sin_w / (2.0 * q)
        scale = 1.0 / (1.0 + alpha)
        b0 = (1.0 - cos_w) * 0.5 * scale
        b1 = (1.0 - cos_w) * scale
        b2 = b0
        a1 = -2.0 * cos_w * scale
        a2 = (1.0 - alpha) * scale

        result = array("f", [0.0]) * len(input_samples)
        x1 = x2 = y1 = y2 = 0.0
        for index, x0 in enumerate(input_samples):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            result[index] = y0
            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0
        return result

    return stage(stage(samples))


def pitch_shifted(source, semitones, sample_rate):
    """Resample to shift pitch while preserving the original sample count."""
    if semitones == 0.0:
        return array("f", source)
    ratio = 2.0 ** (semitones / 12.0)
    filtered = source
    if ratio > 1.0:
        nyquist = sample_rate * 0.5
        cutoff = min(nyquist * 0.9, nyquist / ratio)
        filtered = biquad_lowpass(source, cutoff, sample_rate)

    length = len(source)
    result = array("f", [0.0]) * length
    source_length = len(filtered)
    for index in range(length):
        source_position = index * ratio
        left = int(math.floor(source_position))
        fraction = source_position - left
        right = left + 1
        if left >= source_length or right >= source_length:
            value = 0.0
        elif left < 0:
            value = 0.0
        elif right >= source_length:
            value = filtered[left]
        else:
            value = filtered[left] * (1.0 - fraction) + filtered[right] * fraction
        result[index] = value
    return result


class SourceCache:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._cache = {}

    def get(self, speaker):
        sid = speaker.speaker_id
        if sid not in self._cache:
            raw = source_waveform(speaker, self.sample_rate)
            self._cache[sid] = pitch_shifted(
                raw, speaker.pitch_semitones, self.sample_rate
            )
        return self._cache[sid]


def ptt_burst(speaker, length, sample_rate):
    """Short deterministic key-click burst with an attack/decay envelope."""
    rng = LcgRandom((speaker.seed ^ 0x5F3759DF) & 0x7FFFFFFF)
    result = array("f", [0.0]) * length
    attack = max(1, length // 8)
    decay = max(1, length // 2)
    tone_step = TWO_PI * 1800.0 / sample_rate
    phase = 0.0
    for index in range(length):
        noise = rng.next_uniform() * 2.0 - 1.0
        tone = math.sin(phase)
        phase += tone_step
        if phase >= TWO_PI:
            phase -= TWO_PI
        if index < attack:
            envelope = index / attack
        else:
            envelope = max(0.0, 1.0 - (index - attack) / decay)
        result[index] = 0.35 * envelope * (0.6 * noise + 0.4 * tone)
    return result


def cue_motif(duration_ms, sample_rate, seed, intro=True):
    """Deterministic two-tone cue with fade in/out at the timeline edges."""
    length = max(1, int(round(duration_ms * sample_rate / 1000.0)))
    rng = LcgRandom(seed)
    first_freq = 660.0 if intro else 520.0
    second_freq = 880.0 if intro else 390.0
    split = length // 2
    fade = min(length // 4, max(1, int(0.01 * sample_rate)))
    result = array("f", [0.0]) * length
    phase = 0.0
    for index in range(length):
        frequency = first_freq if index < split else second_freq
        phase += TWO_PI * frequency / sample_rate
        if phase >= TWO_PI:
            phase -= TWO_PI
        shimmer = 1.0 + 0.02 * (rng.next_uniform() - 0.5)
        value = 0.25 * math.sin(phase) * shimmer
        if index < fade:
            value *= index / fade
        tail = length - 1 - index
        if tail < fade:
            value *= tail / fade
        result[index] = value
    return result
