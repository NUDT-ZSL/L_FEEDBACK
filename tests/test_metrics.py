from __future__ import annotations

import unittest

from immersive_audio.metrics import PeakMemoryTracker


class MetricsTests(unittest.TestCase):
    def test_peak_memory_is_measured(self) -> None:
        tracker = PeakMemoryTracker()
        tracker.sample()
        self.assertGreater(tracker.peak_bytes, 0)
        self.assertGreater(tracker.peak_mb, 0.0)
