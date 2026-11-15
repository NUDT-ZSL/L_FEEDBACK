import unittest

import numpy as np

from seamless_loop.crossfade import build_loop, equal_power_windows
from seamless_loop.errors import ParameterRangeError


class EqualPowerWindowTests(unittest.TestCase):
    def test_power_is_constant(self):
        for fade_len in (2, 3, 7, 100, 4800):
            fade_out, fade_in = equal_power_windows(fade_len)
            self.assertEqual(fade_out.shape, (fade_len,))
            self.assertEqual(fade_in.shape, (fade_len,))
            power = fade_out ** 2 + fade_in ** 2
            np.testing.assert_allclose(power, np.ones(fade_len), atol=1e-12)

    def test_endpoints_are_exact(self):
        fade_out, fade_in = equal_power_windows(64)
        self.assertEqual(fade_out[0], 1.0)
        self.assertEqual(fade_out[-1], 0.0)
        self.assertEqual(fade_in[0], 0.0)
        self.assertEqual(fade_in[-1], 1.0)

    def test_not_equal_amplitude(self):
        # Equal-power windows must not be the linear (equal-amplitude) ones.
        fade_out, fade_in = equal_power_windows(8)
        linear = np.linspace(1.0, 0.0, 8)
        self.assertFalse(np.allclose(fade_out, linear))
        # Linear windows would sum to 1; equal-power windows sum differently
        # in the interior while their squares sum to 1 (verified elsewhere).
        self.assertFalse(np.allclose(fade_out + fade_in, np.ones(8)))

    def test_single_sample_fade(self):
        tail_weight, head_weight = equal_power_windows(1)
        np.testing.assert_array_equal(tail_weight, [0.0])
        np.testing.assert_array_equal(head_weight, [1.0])

    def test_zero_fade_rejected(self):
        with self.assertRaises(ParameterRangeError):
            equal_power_windows(0)


class BuildLoopTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.region = rng.standard_normal(1000)

    def test_output_length_and_fade(self):
        loop, fade = build_loop(self.region, 100)
        self.assertEqual(fade, 100)
        self.assertEqual(loop.shape[0], 1000 - 100)  # N - F.

    def test_loop_length_for_minimal_region(self):
        # N = 2F leaves an empty middle; the loop is just the crossfade.
        loop, fade = build_loop(self.region[:200], 100)
        self.assertEqual(fade, 100)
        self.assertEqual(loop.shape[0], 100)

    def test_seam_junctions_are_source_continuous(self):
        fade = 128
        frames = self.region.shape[0]
        loop, _ = build_loop(self.region, fade)
        middle_len = frames - 2 * fade

        # M -> Z junction is the source-adjacent step r[N-F-1] -> r[N-F].
        mz_jump = abs(loop[middle_len] - loop[middle_len - 1])
        src_mz = abs(self.region[frames - fade] - self.region[frames - fade - 1])
        self.assertAlmostEqual(mz_jump, src_mz, places=12)

        # Periodic seam Z[-1] -> M[0] is the source-adjacent step
        # r[F-1] -> r[F], i.e. an ordinary in-signal step, never the raw
        # wrap jump r[-1] -> r[0].
        tiled = np.tile(loop, 2)
        seam_jump = abs(tiled[loop.shape[0]] - tiled[loop.shape[0] - 1])
        src_step = abs(self.region[fade] - self.region[fade - 1])
        self.assertAlmostEqual(seam_jump, src_step, places=12)

    def test_raw_wrap_jump_is_removed(self):
        # The periodic seam must not carry the original raw wrap jump; it is
        # replaced by an ordinary in-signal step.
        fade = 128
        raw_wrap = abs(self.region[0] - self.region[-1])
        loop, _ = build_loop(self.region, fade)
        tiled = np.tile(loop, 4)
        seam_jump = abs(tiled[loop.shape[0]] - tiled[loop.shape[0] - 1])
        self.assertNotAlmostEqual(seam_jump, raw_wrap, places=6)
        expected_step = abs(self.region[fade] - self.region[fade - 1])
        self.assertAlmostEqual(seam_jump, expected_step, places=12)

    def test_crossfade_has_constant_power_interior(self):
        fade = 128
        frames = self.region.shape[0]
        tail_w, head_w = equal_power_windows(fade)
        np.testing.assert_allclose(tail_w ** 2 + head_w ** 2, np.ones(fade))
        # The loop's final fade samples equal the documented blend.
        loop, _ = build_loop(self.region, fade)
        head = self.region[:fade]
        tail = self.region[frames - fade:]
        expected = tail * tail_w + head * head_w
        np.testing.assert_allclose(loop[-fade:], expected, atol=1e-12)

    def test_blend_endpoints_match_source(self):
        fade = 128
        frames = self.region.shape[0]
        loop, _ = build_loop(self.region, fade)
        middle_len = frames - 2 * fade
        # Crossfade starts exactly on the unchanged region tail.
        np.testing.assert_array_equal(
            loop[middle_len], self.region[frames - fade]
        )
        # Crossfade ends exactly on the unchanged head's last sample.
        np.testing.assert_array_equal(loop[-1], self.region[fade - 1])

    def test_internal_junction_is_continuous(self):
        fade = 128
        frames = self.region.shape[0]
        loop, _ = build_loop(self.region, fade)
        middle_len = frames - 2 * fade
        # middle -> crossfade junction keeps exact source adjacency.
        junction_gap = abs(loop[middle_len] - loop[middle_len - 1])
        source_gap = abs(
            self.region[frames - fade] - self.region[frames - fade - 1]
        )
        self.assertAlmostEqual(junction_gap, source_gap, places=12)

    def test_no_artificial_spike_introduced(self):
        loop, _ = build_loop(self.region, 128)
        self.assertLessEqual(np.max(np.abs(loop)), np.max(np.abs(self.region)) + 1e-9)

    def test_dynamic_fade_shortening_and_length(self):
        region = self.region[:20]  # max usable fade = 20 // 2 = 10
        with self.assertLogs("seamless_loop", level="WARNING") as logs:
            loop, effective = build_loop(region, 16, track_name="short.wav")
        self.assertEqual(effective, 10)
        self.assertEqual(loop.shape[0], 10)  # 20 - 10
        self.assertTrue(
            any("short.wav" in msg and "shortening crossfade" in msg
                for msg in logs.output)
        )

    def test_stereo_region(self):
        stereo = np.stack([self.region, self.region * 0.5], axis=1)
        loop, fade = build_loop(stereo, 100)
        self.assertEqual(loop.ndim, 2)
        self.assertEqual(loop.shape[1], 2)
        self.assertEqual(loop.shape[0], 900)
        self.assertEqual(fade, 100)
        # Both channels processed identically with the same window shape.
        np.testing.assert_allclose(loop[:, 1], loop[:, 0] * 0.5, atol=1e-12)

    def test_too_short_region_rejected(self):
        with self.assertRaises(ParameterRangeError):
            build_loop(np.zeros(1), 1)


if __name__ == "__main__":
    unittest.main()
