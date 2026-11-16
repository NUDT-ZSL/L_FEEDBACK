from __future__ import annotations

import math
import unittest

import numpy as np

from spatial_audio_renderer.binaural import BinauralRenderer, CueBlock
from spatial_audio_renderer.compression import Compressor, CompressorConfig
from spatial_audio_renderer.geometry import (
    segment_intersects_aabb,
    spatial_cue,
)
from spatial_audio_renderer.loudness import LoudnessMeter
from spatial_audio_renderer.occlusion import (
    OcclusionFilter,
    occlusion_curves,
)
from spatial_audio_renderer.render import radial_doppler_rates
from spatial_audio_renderer.resampling import (
    DopplerClock,
    SincTrackRenderer,
    sinc_interpolate,
)
from spatial_audio_renderer.scene import OccluderSpec


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def cue_block(cue, length: int) -> CueBlock:
    return CueBlock(
        itd_seconds=np.full(length, cue.itd_seconds),
        left_ild=np.full(length, cue.left_ild),
        right_ild=np.full(length, cue.right_ild),
        left_shadow=np.full(length, cue.left_shadow),
        right_shadow=np.full(length, cue.right_shadow),
        elevation=np.full(length, cue.elevation),
        lateral=np.full(length, cue.lateral),
    )


class BinauralTests(unittest.TestCase):
    def test_ild_and_itd_follow_source_side(self) -> None:
        right = spatial_cue(np.array([2.0, 0.0, 0.0]), IDENTITY)
        left = spatial_cue(np.array([-2.0, 0.0, 0.0]), IDENTITY)
        self.assertGreater(right.itd_seconds, 0.0)
        self.assertLess(left.itd_seconds, 0.0)
        self.assertAlmostEqual(right.left_ild, left.right_ild)
        self.assertLess(right.left_ild, right.right_ild)

    def test_right_source_is_louder_and_earlier_at_right_ear(self) -> None:
        sample_rate = 48000
        length = 4096
        time = np.arange(length) / float(sample_rate)
        tone = 0.2 * np.sin(2.0 * np.pi * 200.0 * time)
        cue = spatial_cue(np.array([2.0, 0.0, 0.0]), IDENTITY)
        output = BinauralRenderer(sample_rate).process(
            tone, cue_block(cue, length)
        )
        rms = np.sqrt(np.mean(output[1000:] ** 2, axis=0))
        self.assertGreater(rms[1], rms[0] * 1.5)

        impulse = np.zeros(512, dtype=np.float64)
        impulse[100] = 1.0
        impulse_output = BinauralRenderer(sample_rate).process(
            impulse, cue_block(cue, len(impulse))
        )
        right_onset = int(np.argmax(np.abs(impulse_output[:, 1]) > 0.03))
        left_onset = int(np.argmax(np.abs(impulse_output[:, 0]) > 0.03))
        self.assertLess(right_onset, left_onset)
        expected = int(round(abs(cue.itd_seconds) * sample_rate))
        self.assertGreaterEqual(left_onset - right_onset, expected - 2)


class DopplerTests(unittest.TestCase):
    def test_radial_rate_matches_closed_form(self) -> None:
        source = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
        listener = np.zeros((2, 3))
        velocity = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]])
        rates = radial_doppler_rates(
            source, listener, velocity, np.zeros((2, 3)), 343.0
        )
        self.assertAlmostEqual(rates[0], 343.0 / (343.0 - 10.0), places=8)
        self.assertAlmostEqual(rates[1], 343.0 / (343.0 + 10.0), places=8)
        self.assertGreater(rates[0], 1.0)
        self.assertLess(rates[1], 1.0)

    def test_doppler_clock_has_no_endpoint_discontinuity(self) -> None:
        clock = DopplerClock()
        first_rates = np.array([1.01, 1.02, 1.03, 1.04])
        first = clock.positions(first_rates)
        second_rates = np.array([1.04, 1.03, 1.02])
        second = clock.positions(second_rates)
        self.assertAlmostEqual(
            second[0], first[-1] + first_rates[-1], places=12
        )
        self.assertAlmostEqual(second[1] - second[0], second_rates[0], places=12)

    def test_sinc_resampler_constant_signal_is_continuous(self) -> None:
        samples = np.ones(1000, dtype=np.float64)
        clock = DopplerClock()
        first = sinc_interpolate(
            samples, clock.positions(np.full(100, 1.03)), np.full(100, 1.03), False
        )
        second = sinc_interpolate(
            samples, clock.positions(np.full(100, 1.03)), np.full(100, 1.03), False
        )
        self.assertAlmostEqual(first[-1], 1.0, places=10)
        self.assertAlmostEqual(second[0], 1.0, places=10)
        self.assertLess(np.max(np.abs(np.diff(np.concatenate((first, second))))), 1.0e-8)

    def test_constant_rate_resampling_shifts_tone_frequency(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        tone = np.sin(2.0 * np.pi * 1000.0 * time)
        rate = 1.1
        renderer = SincTrackRenderer(tone, loop=True)
        output = renderer.render(np.full(len(tone), rate))
        window = output[5000:45000] * np.hanning(40000)
        spectrum = np.abs(np.fft.rfft(window))
        frequencies = np.fft.rfftfreq(len(window), 1.0 / sample_rate)
        peak_frequency = float(frequencies[int(np.argmax(spectrum))])
        self.assertLess(abs(peak_frequency - 1100.0), 2.0)


class OcclusionTests(unittest.TestCase):
    def test_segment_aabb_intersection(self) -> None:
        minimum = np.array([-1.0, -1.0, 1.0])
        maximum = np.array([1.0, 1.0, 2.0])
        self.assertTrue(
            segment_intersects_aabb(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 3.0]),
                minimum,
                maximum,
            )
        )
        self.assertFalse(
            segment_intersects_aabb(
                np.array([3.0, 0.0, 0.0]),
                np.array([3.0, 0.0, 3.0]),
                minimum,
                maximum,
            )
        )

    def test_cutoff_and_gain_decrease_with_layers(self) -> None:
        first = OccluderSpec(
            "first",
            np.array([-1.0, -1.0, 0.5]),
            np.array([1.0, 1.0, 1.0]),
            0.6,
            6000.0,
        )
        second = OccluderSpec(
            "second",
            np.array([-1.0, -1.0, 1.2]),
            np.array([1.0, 1.0, 1.8]),
            0.7,
            4000.0,
        )
        listener = np.zeros((1, 3))
        source = np.array([[0.0, 0.0, 5.0]])
        depths, cutoffs, gains = occlusion_curves(
            [first, second], listener, source, 48000, 18000.0
        )
        self.assertEqual(int(depths[0]), 2)
        self.assertLess(cutoffs[0], 4000.0)
        self.assertLess(cutoffs[0], 6000.0)
        self.assertAlmostEqual(gains[0], 0.42)

    def test_filter_state_is_identical_across_frame_boundaries(self) -> None:
        sample_rate = 48000
        length = 4800
        time = np.arange(length) / float(sample_rate)
        signal = 0.3 * np.sin(2.0 * np.pi * 1000.0 * time)
        cutoff = np.where(
            time < 0.05, 18000.0, np.where(time < 0.1, 3000.0, 6000.0)
        )
        gain = np.where(time < 0.05, 1.0, 0.5)
        whole_filter = OcclusionFilter(sample_rate)
        whole = whole_filter.process(signal, cutoff, gain)
        frame_filter = OcclusionFilter(sample_rate)
        split = np.concatenate(
            [
                frame_filter.process(
                    signal[start : start + 256],
                    cutoff[start : start + 256],
                    gain[start : start + 256],
                )
                for start in range(0, length, 256)
            ]
        )
        self.assertTrue(np.allclose(whole, split, atol=1.0e-14))
        self.assertTrue(np.all(np.isfinite(whole)))
        self.assertLess(float(np.max(np.abs(np.diff(whole)))), 0.05)


class CompressionTests(unittest.TestCase):
    def test_soft_knee_ratio_and_zero_latency(self) -> None:
        sample_rate = 48000
        config = CompressorConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=50.0,
            knee_db=6.0,
        )
        compressor = Compressor(sample_rate, config)
        signal = np.full((sample_rate, 2), 0.1)
        output = compressor.process(signal)
        self.assertEqual(output.shape, signal.shape)
        self.assertGreater(compressor.gain_envelope_db[-1], 0.55)
        self.assertLess(compressor.gain_envelope_db[-1], 0.58)
        self.assertGreater(
            compressor.gain_envelope_db[-1],
            compressor.gain_envelope_db[100],
        )
        self.assertGreater(output[0, 0], 0.099)

    def test_attack_and_release_are_smooth(self) -> None:
        sample_rate = 48000
        compressor = Compressor(
            sample_rate,
            CompressorConfig(
                threshold_db=-30.0,
                ratio=8.0,
                attack_ms=1.0,
                release_ms=80.0,
                knee_db=4.0,
            ),
        )
        signal = np.zeros((sample_rate, 2))
        signal[: sample_rate // 2] = 0.7
        signal[sample_rate // 2 :] = 0.01
        compressor.process(signal)
        envelope = compressor.gain_envelope_db
        self.assertTrue(np.all(np.isfinite(envelope)))
        self.assertLess(float(np.max(np.abs(np.diff(envelope)))), 0.75)
        initial = float(np.mean(envelope[:10]))
        early = float(np.mean(envelope[100:500]))
        late = float(np.mean(envelope[22000:24000]))
        self.assertGreater(early, initial)
        self.assertLess(early, late)
        released = float(np.mean(envelope[-1000:]))
        self.assertLess(released, late)


class LoudnessTests(unittest.TestCase):
    def test_integrated_loudness_tracks_level(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 2) / float(sample_rate)
        quiet = np.column_stack(
            (
                0.05 * np.sin(2.0 * np.pi * 1000.0 * time),
                0.05 * np.sin(2.0 * np.pi * 1000.0 * time),
            )
        )
        loud = quiet * 4.0
        quiet_meter = LoudnessMeter(sample_rate)
        quiet_meter.process(quiet)
        loud_meter = LoudnessMeter(sample_rate)
        loud_meter.process(loud)
        quiet_lufs = quiet_meter.integrated_loudness()
        loud_lufs = loud_meter.integrated_loudness()
        self.assertIsNotNone(quiet_lufs)
        self.assertIsNotNone(loud_lufs)
        self.assertAlmostEqual(
            loud_lufs - quiet_lufs, 20.0 * math.log10(4.0), places=2
        )


if __name__ == "__main__":
    unittest.main()
