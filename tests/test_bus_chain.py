import unittest

import numpy as np

from offline_audio_render.bus_chain import (
    LIMITER_THRESHOLD_DB,
    FeedbackDelay,
    SoftKneeLimiter,
)
from offline_audio_render.curves import db_to_gain


class BusChainTests(unittest.TestCase):
    def test_limiter_ceiling_and_soft_knee(self):
        limiter = SoftKneeLimiter(channels=1, sample_rate=48000)
        block = np.ones((4096, 1), dtype=np.float64) * 2.0
        out = limiter.process(block)
        ceiling = db_to_gain(LIMITER_THRESHOLD_DB)
        self.assertLessEqual(np.max(np.abs(out)), ceiling + 1e-12)
        self.assertGreater(np.max(out), 0.5)

    def test_limiter_state_continuous_across_blocks(self):
        first = SoftKneeLimiter(1, 48000)
        second = SoftKneeLimiter(1, 48000)
        signal = (np.random.default_rng(0).normal(size=1024) * 1.5)[:, None]
        split = np.vstack(
            [
                first.process(signal[:512]),
                first.process(signal[512:]),
            ]
        )
        whole = second.process(signal)
        np.testing.assert_allclose(split, whole, rtol=1e-12)

    def test_feedback_delay_matches_reference_recurrence(self):
        sample_rate = 1000
        delay = FeedbackDelay(1, sample_rate, 0.01, 0.4)
        signal = np.zeros((500, 1), dtype=np.float64)
        signal[0, 0] = 1.0
        out = delay.process(signal)
        reference = np.zeros(500)
        buffer = np.zeros(10)
        write = 0
        for index in range(500):
            read = (write - 10) % 10
            delayed = buffer[read]
            reference[index] = signal[index, 0] + delayed
            buffer[write] = signal[index, 0] + 0.4 * delayed
            write = (write + 1) % 10
        np.testing.assert_allclose(out[:, 0], reference)
        self.assertAlmostEqual(out[10, 0], 1.0)
        self.assertAlmostEqual(out[20, 0], 0.4)


if __name__ == "__main__":
    unittest.main()
