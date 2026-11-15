import math
import unittest

import numpy as np

from cinema_master.dsp import (
    active_mask_from_segments,
    apply_lookahead_limiter,
    compressor_gain_db,
    detect_dialogue_activity,
    estimate_true_peak,
    integrated_loudness,
    resample_audio,
)


class DspTests(unittest.TestCase):
    def test_resampling_preserves_timeline_length(self):
        sample_rate = 44100
        source = np.sin(
            2.0
            * math.pi
            * 1000.0
            * np.arange(4410, dtype=np.float64)
            / sample_rate
        )[:, np.newaxis]
        output, frames = resample_audio(source, 44100, 48000)
        self.assertEqual(frames, 4800)
        self.assertEqual(output.shape, (4800, 1))
        self.assertGreater(float(np.max(np.abs(output))), 0.8)

    def test_resampling_has_no_leading_sample_offset(self):
        source = np.zeros((4410, 1), dtype=np.float64)
        source[100, 0] = 1.0
        output, _ = resample_audio(source, 44100, 48000)
        expected = 100.0 * 48000.0 / 44100.0
        self.assertLessEqual(abs(int(np.argmax(output[:, 0])) - expected), 1.0)

    def test_downsampling_is_antialiased(self):
        sample_rate = 48000
        frequency = 19000.0
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        source = np.sin(2.0 * math.pi * frequency * time)[:, np.newaxis]
        output, _ = resample_audio(source, 48000, 32000)
        interior = output[1000:-1000, 0]
        rms = math.sqrt(float(np.mean(interior * interior)))
        self.assertLess(20.0 * math.log10(max(rms, 1.0e-12)), -40.0)

    def test_dialogue_detection_merges_and_filters(self):
        sample_rate = 48000
        frames = sample_rate * 3
        signal = np.zeros((frames, 1), dtype=np.float64)
        first = int(0.4 * sample_rate)
        last = int(1.4 * sample_rate)
        tone = np.sin(
            2.0 * math.pi * 300.0 * np.arange(last - first) / sample_rate
        )
        signal[first:last, 0] = 0.2 * tone
        short_start = int(2.3 * sample_rate)
        short_end = int(2.40 * sample_rate)
        signal[short_start:short_end, 0] = 0.2
        result = detect_dialogue_activity(
            signal, sample_rate, 400.0, 0.75, -35.0, 1000.0
        )
        self.assertEqual(len(result["segments"]), 1)
        segment = result["segments"][0]
        self.assertLessEqual(segment["start_sample"], first)
        self.assertGreaterEqual(segment["end_sample"], last - 1)
        mask = active_mask_from_segments(result["segments"], frames)
        self.assertFalse(mask[0])
        self.assertTrue(mask[first + 1000])

    def test_compression_is_gated_to_dialogue_activity(self):
        sample_rate = 48000
        frames = sample_rate * 2
        time = np.arange(frames, dtype=np.float64) / sample_rate
        dialogue = np.zeros((frames, 1), dtype=np.float64)
        active = np.zeros(frames, dtype=bool)
        active[int(0.5 * sample_rate) : int(1.5 * sample_rate)] = True
        dialogue[active, 0] = 0.5 * np.sin(2.0 * math.pi * 500.0 * time[active])
        gain, stats = compressor_gain_db(
            dialogue,
            active,
            sample_rate,
            -24.0,
            3.0,
            10.0,
            100.0,
            6.0,
            True,
        )
        self.assertAlmostEqual(float(gain[0]), 0.0)
        self.assertLess(float(np.min(gain[active])), -1.0)
        self.assertGreater(stats["max_reduction_db"], 1.0)

    def test_limiter_guarantees_true_peak_ceiling(self):
        sample_rate = 48000
        time = np.arange(int(0.5 * sample_rate), dtype=np.float64) / sample_rate
        signal = np.stack(
            [
                0.95 * np.sin(2.0 * math.pi * 997.0 * time),
                0.85 * np.sin(2.0 * math.pi * 1301.0 * time + 0.4),
            ],
            axis=1,
        )
        limited, gain, stats = apply_lookahead_limiter(
            signal, sample_rate, -1.0, 8.0, 80.0, 4, 0.25
        )
        _, peak_db = estimate_true_peak(limited, sample_rate, 4)
        self.assertLessEqual(peak_db, -1.0 + 1.0e-7)
        self.assertTrue(stats["triggered"])
        self.assertLess(float(np.min(gain)), 1.0)

    def test_integrated_loudness_is_finite_for_tone(self):
        sample_rate = 48000
        time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
        tone = 0.05 * np.sin(2.0 * math.pi * 1000.0 * time)
        data = np.stack([tone, tone], axis=1)
        loudness = integrated_loudness(data, sample_rate, ["FL", "FR"])
        self.assertTrue(math.isfinite(loudness))
        self.assertLess(loudness, 0.0)


if __name__ == "__main__":
    unittest.main()

