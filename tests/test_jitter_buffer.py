import unittest

import numpy as np

from audio_sync.constants import (
    EVENT_BACKPRESSURE,
    EVENT_LATE,
    EVENT_OVERFLOW,
    MAX_BUFFER_BYTES,
    MAX_BUFFER_DEPTH,
)
from audio_sync.jitter_buffer import JitterBuffer


def frame(value=0.1):
    return np.full(960, value, dtype=np.float32)


class JitterBufferTests(unittest.TestCase):
    def test_rejects_out_of_range_depth(self):
        with self.assertRaises(ValueError):
            JitterBuffer(0, MAX_BUFFER_DEPTH + 1)
        with self.assertRaises(ValueError):
            JitterBuffer(0, 1)

    def test_insert_and_ordered_pop_with_reordering(self):
        buffer = JitterBuffer(0, 3)
        buffer.insert(2, frame(0.2), 40.0)
        buffer.insert(0, frame(0.0), 0.0)
        buffer.insert(1, frame(0.1), 20.0)
        samples, underflow = buffer.pop(0)
        self.assertFalse(underflow)
        self.assertAlmostEqual(float(samples[0]), 0.0, places=5)
        samples, underflow = buffer.pop(1)
        self.assertFalse(underflow)
        self.assertAlmostEqual(float(samples[0]), 0.1, places=5)

    def test_underflow_event(self):
        buffer = JitterBuffer(0, 3)
        buffer.insert(1, frame(), 20.0)
        _, underflow = buffer.pop(0)
        self.assertTrue(underflow)
        kinds = [event.kind for event in buffer.drain_events()]
        self.assertIn("buffer_underflow", kinds)

    def test_overflow_drops_oldest_and_emits_backpressure(self):
        buffer = JitterBuffer(0, 2, max_depth=2)
        buffer.insert(0, frame(0.0), 0.0)
        buffer.insert(1, frame(0.1), 20.0)
        event = buffer.insert(2, frame(0.2), 40.0)
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, EVENT_OVERFLOW)
        kinds = [event.kind for event in buffer.drain_events()]
        self.assertIn(EVENT_OVERFLOW, kinds)
        self.assertIn(EVENT_BACKPRESSURE, kinds)
        # Oldest frame (index 0) was discarded; index 1 is now oldest.
        _, underflow = buffer.pop(0)
        self.assertTrue(underflow)
        samples, underflow = buffer.pop(1)
        self.assertFalse(underflow)

    def test_duplicate_is_backpressure(self):
        buffer = JitterBuffer(0, 3)
        buffer.insert(0, frame(), 0.0)
        event = buffer.insert(0, frame(), 5.0)
        self.assertEqual(event.kind, EVENT_BACKPRESSURE)

    def test_late_frame_after_playout(self):
        buffer = JitterBuffer(0, 3)
        buffer.pop(0)
        event = buffer.insert(0, frame(), 999.0)
        self.assertEqual(event.kind, EVENT_LATE)

    def test_memory_budget_within_limit(self):
        buffer = JitterBuffer(0, MAX_BUFFER_DEPTH)
        for index in range(MAX_BUFFER_DEPTH):
            buffer.insert(index, frame(), float(index))
        # 10 float32 frames + 4-frame PLC history must stay below 256 KB.
        self.assertLess(buffer.memory_bytes, MAX_BUFFER_BYTES)

    def test_adaptive_depth_bounds(self):
        buffer = JitterBuffer(0, 3)
        buffer.update_target_depth(MAX_BUFFER_DEPTH + 5)
        self.assertEqual(buffer.target_depth, MAX_BUFFER_DEPTH)
        buffer.update_target_depth(0)
        self.assertEqual(buffer.target_depth, 2)


if __name__ == "__main__":
    unittest.main()
