import unittest

import numpy as np

from audio_sync.constants import FRAME_SAMPLES, MAX_PLC_FRAMES
from audio_sync.plc import PlcState, estimate_pitch


def periodic_frame(period=120, amplitude=0.4):
    base = np.arange(FRAME_SAMPLES * 4, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * base / period)).astype(np.float32)


class PlcTests(unittest.TestCase):
    def test_pitch_estimate_finds_known_period(self):
        history = periodic_frame(period=120)
        lag, confidence = estimate_pitch(history)
        self.assertIsNotNone(lag)
        self.assertGreater(confidence, 0.45)
        self.assertLess(abs(lag - 120), 20)

    def test_conceal_produces_smooth_frame_within_budget(self):
        state = PlcState(0)
        signal = periodic_frame()
        for frame_index in range(4):
            frame = signal[
                frame_index * FRAME_SAMPLES:(frame_index + 1) * FRAME_SAMPLES
            ]
            state.accept_good_frame(frame)

        previous_tail = state.history[-8:].copy()
        last_history_sample = state.history[-1]
        result = state.conceal()
        self.assertTrue(result.concealed)
        self.assertEqual(result.frame.shape, (FRAME_SAMPLES,))
        # No click at the join: the first concealed sample stays close to the
        # last good sample and the step is no larger than the natural slope.
        join_gap = abs(float(result.frame[0]) - float(last_history_sample))
        natural_step = float(
            np.max(np.abs(np.diff(previous_tail)))
        )
        self.assertLess(join_gap, natural_step * 1.5 + 1e-3)

    def test_long_loss_run_becomes_silence_and_recovers(self):
        state = PlcState(0)
        state.accept_good_frame(periodic_frame()[:FRAME_SAMPLES])
        methods = []
        for _ in range(MAX_PLC_FRAMES):
            methods.append(state.conceal().method)
        silence = state.conceal()
        self.assertFalse(silence.concealed)
        self.assertEqual(silence.method, "silence")
        self.assertTrue(np.all(silence.frame == 0.0))

        recovered = state.accept_good_frame(
            np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)
        )
        self.assertIsNotNone(recovered)
        # Fade-in starts from zero.
        self.assertAlmostEqual(float(recovered[0]), 0.0, places=6)
        self.assertGreater(float(recovered[-1]), 0.4)

    def test_linear_prediction_fallback_for_noise_like_history(self):
        state = PlcState(0)
        rng = np.random.default_rng(7)
        state.accept_good_frame(rng.standard_normal(FRAME_SAMPLES).astype(np.float32) * 0.1)
        result = state.conceal()
        self.assertEqual(result.frame.shape, (FRAME_SAMPLES,))
        self.assertIn(result.method, {"pitch", "linear_prediction"})

    def test_silence_after_three_concealed_frames_then_recovery_point(self):
        state = PlcState(0)
        state.accept_good_frame(periodic_frame()[:FRAME_SAMPLES])
        for _ in range(MAX_PLC_FRAMES):
            self.assertTrue(state.conceal().concealed)
        fourth = state.conceal()
        self.assertEqual(fourth.method, "silence")
        self.assertFalse(fourth.concealed)
        self.assertTrue(np.all(fourth.frame == 0.0))
        # Every further loss in the same run stays silent.
        self.assertEqual(state.conceal().method, "silence")
        # First real frame afterwards is the marked recovery point.
        recovered = state.accept_good_frame(
            np.full(FRAME_SAMPLES, 0.3, dtype=np.float32)
        )
        self.assertIsNotNone(recovered)
        self.assertAlmostEqual(float(recovered[0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
