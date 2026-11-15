import unittest

import numpy as np

from audiopipeline.config import Config
from audiopipeline.manifest import Event
from audiopipeline.mixer import place_events


def make_event(index, start_ms, duration_ms, priority, overlap=None, value=0.5):
    event = Event(
        index=index,
        event_id=f"e{index}",
        path="a.wav",
        start_ms=float(start_ms),
        duration_ms=float(duration_ms),
        priority=priority,
        overlap=overlap,
    )
    samples = int(round(duration_ms * 48))
    segment = np.full((samples, 2), value, dtype=np.float32)
    return event, segment


class MixerTests(unittest.TestCase):
    def test_higher_priority_replaces_lower(self):
        cfg = Config(overlap_strategy="replace")
        low, seg_low = make_event(0, 0, 1000, priority=1, value=0.2)
        high, seg_high = make_event(1, 500, 1000, priority=9, value=0.8)
        total = int(1.5 * 48000)
        timeline, _, warnings = place_events(
            [(low, seg_low), (high, seg_high)], total, cfg
        )
        # First 500 ms belongs to low, overlap region belongs to high.
        self.assertAlmostEqual(float(timeline[100, 0]), 0.2, places=4)
        self.assertAlmostEqual(float(timeline[30000, 0]), 0.8, places=4)
        self.assertNotIn("E_WARN_EQUAL_PRIORITY", high.warnings)

    def test_equal_priority_first_come_first_served(self):
        cfg = Config(overlap_strategy="replace")
        first, seg_a = make_event(0, 0, 1000, priority=5, value=0.3)
        second, seg_b = make_event(1, 500, 500, priority=5, value=0.9)
        total = int(1.5 * 48000)
        timeline, _, warnings = place_events(
            [(first, seg_a), (second, seg_b)], total, cfg
        )
        # First event keeps the overlap region.
        self.assertAlmostEqual(float(timeline[30000, 0]), 0.3, places=4)
        self.assertIn("E_WARN_EQUAL_PRIORITY", second.warnings)
        self.assertTrue(warnings)

    def test_mix_mode_sums_signals(self):
        cfg = Config(overlap_strategy="mix")
        a, seg_a = make_event(0, 0, 1000, priority=5, value=0.2)
        b, seg_b = make_event(1, 500, 500, priority=5, value=0.3)
        total = int(1.5 * 48000)
        timeline, _, _ = place_events([(a, seg_a), (b, seg_b)], total, cfg)
        self.assertAlmostEqual(float(timeline[30000, 0]), 0.5, places=4)

    def test_timeline_padded_to_declared_length(self):
        cfg = Config()
        event, seg = make_event(0, 0, 500, priority=0)
        total = 48000  # 1 second declared, only 0.5 s content
        timeline, _, _ = place_events([(event, seg)], total, cfg)
        self.assertEqual(timeline.shape[0], total)
        self.assertTrue(np.all(timeline[24000:] == 0))

    def test_event_beyond_timeline_is_dropped(self):
        cfg = Config()
        event, seg = make_event(0, 2000, 500, priority=0)
        total = 48000
        timeline, placed, _ = place_events([(event, seg)], total, cfg)
        self.assertEqual(placed, [])
        self.assertIn("E_WARN_OUT_OF_TIMELINE", event.warnings)


if __name__ == "__main__":
    unittest.main()
