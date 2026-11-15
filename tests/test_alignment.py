import unittest

import numpy as np

from audiopipeline.config import SAMPLE_RATE
from audiopipeline.manifest import Event, ms_to_samples
from audiopipeline.mixer import place_events
from audiopipeline.config import Config


class AlignmentTests(unittest.TestCase):
    def test_sample_exact_placement(self):
        cfg = Config()
        # A distinctive ramp so each sample has a unique, traceable value.
        duration_ms = 1000.0
        n = ms_to_samples(duration_ms)
        ramp = np.linspace(0.01, 0.5, n, dtype=np.float32)[:, None]
        segment = np.tile(ramp, (1, 2))
        event = Event(index=0, event_id="r", path="r.wav", start_ms=500.0)
        total = ms_to_samples(3000.0)
        timeline, placed, _ = place_events([(event, segment)], total, cfg)

        start = ms_to_samples(500.0)
        # Silence before the event, exact sample match inside, silence after.
        self.assertTrue(np.all(timeline[:start] == 0))
        np.testing.assert_array_equal(timeline[start : start + n], segment)
        self.assertTrue(np.all(timeline[start + n :] == 0))
        self.assertEqual(placed[0].start, start)
        self.assertEqual(placed[0].end, start + n)
        # Reported timestamps map back to the same samples.
        self.assertEqual(ms_to_samples(event.actual_start_ms), start)
        self.assertEqual(ms_to_samples(event.actual_end_ms), start + n)

    def test_subsample_timestamp_error_below_one_ms(self):
        for ms in (0.333, 12.345, 999.999, 2123.0):
            samples = ms_to_samples(ms)
            back_ms = samples * 1000.0 / SAMPLE_RATE
            self.assertLessEqual(abs(back_ms - ms), 1.0 / SAMPLE_RATE * 1000.0)


if __name__ == "__main__":
    unittest.main()
