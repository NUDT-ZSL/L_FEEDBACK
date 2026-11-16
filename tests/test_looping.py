import unittest

import numpy as np

from rhythm_mixer.looping import PeriodicSource, build_layer_source


class PeriodicSourceTests(unittest.TestCase):
    def test_loop_point_continuous_no_step(self):
        # Smooth body but a deliberate large jump at the raw loop seam.
        length = 1000
        clip = np.linspace(0.0, 0.2, length, dtype=np.float32)
        clip[0] = 0.9
        source = PeriodicSource(clip, crossfade_samples=441)
        period = source.period  # 1000 - 441 = 559

        window = source.read(period - 4, 9)
        # Adjacent samples across the loop point differ by at most a small
        # blend step, never the full scale of an un-crossfaded jump.
        max_step = float(np.max(np.abs(np.diff(window))))
        raw_step = abs(float(clip[0]) - float(clip[-1]))
        self.assertLess(max_step, raw_step)
        self.assertLess(max_step, 0.05)
        # The cross-fade preserves unity gain across the overlap.
        overlap = source.read(period - source.crossfade, source.crossfade)
        self.assertTrue(np.all(np.isfinite(overlap)))
        self.assertLess(float(np.max(np.abs(overlap))), 1.01)

    def test_periodicity_over_multiple_loops(self):
        rng = np.random.default_rng(11)
        clip = rng.standard_normal(500).astype(np.float32)
        source = PeriodicSource(clip, crossfade_samples=64)
        first = source.read(0, source.period)
        later = source.read(3 * source.period, source.period)
        np.testing.assert_allclose(first, later, atol=1e-6)

    def test_body_samples_identical_to_clip(self):
        clip = np.linspace(-1, 1, 300).astype(np.float32)
        source = PeriodicSource(clip, crossfade_samples=20)
        out = source.read(0, 300)
        # The preserved body region passes through untouched.
        np.testing.assert_allclose(out[20:280], clip[20:280], atol=1e-7)

    def test_short_clip_caps_crossfade(self):
        clip = np.ones(4, dtype=np.float32)
        source = PeriodicSource(clip, crossfade_samples=100)
        self.assertEqual(source.crossfade, 2)
        out = source.read(0, 12)
        # Constant signal must stay constant regardless of cross-fading.
        np.testing.assert_allclose(out, np.ones(12, dtype=np.float32), atol=1e-6)

    def test_build_layer_source_musical_period(self):
        sample_rate = 44100
        beat_samples = int(round(0.5 * sample_rate))  # 120 BPM
        clip = np.random.default_rng(3).standard_normal(2000).astype(np.float32)
        source = build_layer_source(
            clip,
            crossfade_ms=10.0,
            sample_rate=sample_rate,
            beat_period_samples=beat_samples * 8,  # 2 bars of 4 beats
            loop_enabled=True,
        )
        # Musical period is an exact number of bars; seam stays on the grid.
        self.assertEqual(source.period + source.crossfade, beat_samples * 8)
        self.assertEqual(source.crossfade, 441)  # 10 ms at 44.1 kHz
        first = source.read(0, 64)
        second = source.read(source.period, 64)
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_crossfade_never_moves_loop_boundary(self):
        from rhythm_mixer.beatgrid import BeatGrid

        sample_rate = 44100
        grid = BeatGrid(121.0, sample_rate)
        beats_per_loop = 8
        bar_samples = grid.sample_index_at_ms(
            grid.beat_time_ms(beats_per_loop)
        )
        clip = np.linspace(-1, 1, 5000).astype(np.float32)
        source = build_layer_source(
            clip,
            crossfade_ms=10.0,
            sample_rate=sample_rate,
            beat_period_samples=bar_samples,
            loop_enabled=True,
        )
        musical_period = source.period + source.crossfade
        for bar in range(1, 4):
            boundary = grid.sample_index_at_ms(
                grid.beat_time_ms(beats_per_loop * bar)
            )
            self.assertEqual(boundary, bar * musical_period)

    def test_zero_crossfade_is_plain_repeat(self):
        clip = np.arange(10, dtype=np.float32)
        source = PeriodicSource(clip, crossfade_samples=0)
        out = source.read(0, 25)
        expected = np.concatenate([clip, clip, clip[:5]])
        np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
