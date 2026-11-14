import math
import os
import sys
import unittest
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from voice_mixer.models import Speaker
from voice_mixer.synthesis import (
    LcgRandom,
    biquad_lowpass,
    cue_motif,
    pitch_shifted,
    source_waveform,
)
from voice_mixer.renderer import db_to_linear, soft_clip


def _speaker(**overrides):
    values = dict(
        speaker_id="alpha",
        seed=42,
        base_frequency=120.0,
        harmonic_ratio=0.3,
        pitch_semitones=0.0,
        source_duration_ms=250.0,
        bound_channel="squad",
        pan=0.0,
    )
    values.update(overrides)
    return Speaker(**values)


def dft_energy(samples, target_frequency, sample_rate):
    cos_sum = 0.0
    sin_sum = 0.0
    step = 2.0 * math.pi * target_frequency / sample_rate
    for index, value in enumerate(samples):
        angle = step * index
        cos_sum += value * math.cos(angle)
        sin_sum += value * math.sin(angle)
    return math.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / len(samples)


def sine_wave(frequency, sample_rate, duration_ms=250.0):
    length = int(round(duration_ms * sample_rate / 1000.0))
    return array("f", [
        math.sin(2.0 * math.pi * frequency * index / sample_rate)
        for index in range(length)
    ])


class SynthesisTest(unittest.TestCase):
    def test_lcg_is_deterministic(self):
        first = [LcgRandom(7).next_uint() for _ in range(8)]
        second = [LcgRandom(7).next_uint() for _ in range(8)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, [LcgRandom(8).next_uint() for _ in range(8)])

    def test_source_waveform_byte_stable(self):
        speaker = _speaker()
        a = source_waveform(speaker, 16000)
        b = source_waveform(speaker, 16000)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(len(a), 4000)

    def test_pitch_shift_preserves_length_and_changes_frequency(self):
        sample_rate = 16000
        source = source_waveform(_speaker(), sample_rate)
        shifted = pitch_shifted(source, 12.0, sample_rate)
        self.assertEqual(len(shifted), len(source))
        old_frequency_energy = dft_energy(shifted, 120.0, sample_rate)
        new_frequency_energy = dft_energy(shifted, 240.0, sample_rate)
        self.assertGreater(new_frequency_energy, old_frequency_energy * 10.0)

    def test_antialiasing_suppresses_images(self):
        sample_rate = 16000
        high_source = sine_wave(6000.0, sample_rate)
        low_source = sine_wave(1000.0, sample_rate)
        cutoff = sample_rate * 0.5 / 2.0
        filtered_high = biquad_lowpass(high_source, cutoff, sample_rate)
        filtered_low = biquad_lowpass(low_source, cutoff, sample_rate)
        self.assertLess(
            dft_energy(filtered_high, 6000.0, sample_rate),
            dft_energy(high_source, 6000.0, sample_rate) * 0.1,
        )
        self.assertGreater(
            dft_energy(filtered_low, 1000.0, sample_rate),
            dft_energy(low_source, 1000.0, sample_rate) * 0.5,
        )
        shifted = pitch_shifted(low_source, 12.0, sample_rate)
        self.assertEqual(len(shifted), len(low_source))

    def test_lowpass_attenuates_high_frequency(self):
        sample_rate = 16000
        source = array("f", [math.sin(2 * math.pi * 3000 * n / sample_rate) for n in range(2048)])
        filtered = biquad_lowpass(source, 800.0, sample_rate)
        self.assertLess(
            dft_energy(filtered, 3000.0, sample_rate),
            dft_energy(source, 3000.0, sample_rate) * 0.2,
        )

    def test_cue_is_deterministic_and_faded(self):
        a = cue_motif(300, 16000, 123, intro=True)
        b = cue_motif(300, 16000, 123, intro=True)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertAlmostEqual(a[0], 0.0, places=6)
        self.assertAlmostEqual(a[-1], 0.0, places=6)


class MixerMathTest(unittest.TestCase):
    def test_db_conversion(self):
        self.assertAlmostEqual(db_to_linear(6.0), 1.9952623149688795)
        self.assertAlmostEqual(db_to_linear(0.0), 1.0)

    def test_soft_clip_limits_and_preserves_in_range(self):
        self.assertAlmostEqual(soft_clip(0.0), 0.0)
        self.assertLess(soft_clip(2.0), 1.0)
        self.assertGreater(soft_clip(2.0), 0.95)
        self.assertEqual(soft_clip(-2.0), -soft_clip(2.0))


if __name__ == "__main__":
    unittest.main()
