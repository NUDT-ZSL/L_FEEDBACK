import unittest

import numpy as np

from audiopipeline.config import Config, SAMPLE_RATE
from audiopipeline.fades import apply_crossfade, cosine_gain, loop_source
from audiopipeline.builder import build_segment
from audiopipeline.manifest import Event


class FadeTests(unittest.TestCase):
    def test_cosine_endpoints(self):
        fade_in = cosine_gain(100, fade_in=True)
        fade_out = cosine_gain(100, fade_in=False)
        # Half-sample cosine: endpoints sit half a step inside 0 and 1.
        eps = 0.5 * (1 - np.cos(np.pi / 200))
        self.assertAlmostEqual(fade_in[0], eps, places=6)
        self.assertAlmostEqual(fade_in[-1], 1 - eps, places=6)
        self.assertAlmostEqual(fade_out[0], 1 - eps, places=6)
        self.assertAlmostEqual(fade_out[-1], eps, places=6)

    def test_crossfade_preserves_energy_at_center(self):
        tail = np.ones((50, 2))
        head = np.ones((50, 2))
        blended = apply_crossfade(tail, head)
        # Equal-gain cosine: center sums to 1.0.
        self.assertTrue(np.allclose(blended[:, 0], 1.0, atol=3e-4))

    def test_loop_returns_requested_length(self):
        source = np.random.default_rng(1).normal(size=(1000, 2)).astype(np.float32)
        for needed in (500, 1000, 2500, 10000):
            out = loop_source(source, needed, crossfade_samples=480)
            self.assertEqual(out.shape[0], needed)
            self.assertTrue(np.isfinite(out).all())

    def test_loop_seam_is_continuous(self):
        # A constant source must stay constant across the loop seam.
        source = np.full((1000, 2), 0.7, dtype=np.float32)
        out = loop_source(source, 3000, crossfade_samples=480)
        self.assertTrue(np.allclose(out, 0.7, atol=1e-6))


class BuilderTests(unittest.TestCase):
    def _event(self, **kwargs):
        defaults = dict(
            index=0, event_id="x", path="a.wav", start_ms=0.0,
            offset_ms=0.0, duration_ms=None, loop=False, priority=0,
        )
        defaults.update(kwargs)
        return Event(**defaults)

    def test_non_loop_trim_applies_tail_fade(self):
        cfg = Config()
        source = np.ones((SAMPLE_RATE, 2), dtype=np.float32)
        event = self._event(duration_ms=500.0)
        built = build_segment(event, source, cfg, SAMPLE_RATE)
        self.assertEqual(built.audio.shape[0], SAMPLE_RATE // 2)
        self.assertTrue(built.trimmed)
        tail = int(round(cfg.tail_fade_ms * SAMPLE_RATE / 1000))
        self.assertLess(float(built.audio[-1, 0]), 1e-3)
        self.assertAlmostEqual(float(built.audio[-tail, 0]), 1.0, places=4)

    def test_short_non_loop_pads_silence(self):
        cfg = Config()
        source = np.ones((SAMPLE_RATE // 4, 2), dtype=np.float32)
        event = self._event(duration_ms=500.0)
        built = build_segment(event, source, cfg, SAMPLE_RATE)
        self.assertTrue(built.padded)
        self.assertEqual(built.audio.shape[0], SAMPLE_RATE // 2)
        self.assertTrue(np.all(built.audio[SAMPLE_RATE // 4 :] == 0))

    def test_loop_with_offset(self):
        cfg = Config()
        source = np.ones((SAMPLE_RATE, 2), dtype=np.float32)
        event = self._event(loop=True, offset_ms=250.0, duration_ms=2000.0)
        built = build_segment(event, source, cfg, 3 * SAMPLE_RATE)
        self.assertEqual(built.audio.shape[0], int(2 * SAMPLE_RATE))
        self.assertTrue(built.looped)


if __name__ == "__main__":
    unittest.main()
