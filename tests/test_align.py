import unittest

import numpy as np

from lipsync_tool import align


class AlignTests(unittest.TestCase):
    def test_frame_to_time_mapping(self):
        events = [
            {"id": "a", "phoneme": "AA", "start_frame": 24, "duration_frames": 12},
            {"id": "b", "phoneme": "M", "start_frame": 48, "duration_frames": 6},
        ]
        mapped = align.map_events_to_audio_time(events, frame_rate=24.0)
        self.assertAlmostEqual(mapped[0]["start_time"], 1.0)
        self.assertAlmostEqual(mapped[0]["duration"], 0.5)
        self.assertAlmostEqual(mapped[1]["start_time"], 2.0)

    def test_energy_peak_alignment_within_tolerance(self):
        rate = 16000
        signal = np.zeros(rate * 4, dtype=np.float64)
        # Put a strong burst 30 ms after the nominal event time.
        nominal = 1.0
        shift = 0.03
        burst_start = int((nominal + shift) * rate)
        time_axis = np.arange(int(0.12 * rate)) / rate
        signal[burst_start : burst_start + time_axis.size] += 0.6 * np.sin(
            2 * np.pi * 250 * time_axis
        )
        events = [
            {"id": "a", "phoneme": "AA", "start_time": nominal, "duration": 0.12,
             "end_time": nominal + 0.12, "original_start_frame": 24}
        ]
        aligned, stats = align.align_events(events, signal, rate, tolerance_ms=50.0)
        self.assertAlmostEqual(aligned[0]["start_time"], nominal + shift, delta=0.01)
        self.assertLessEqual(abs(aligned[0]["offset_ms"]), 50.0 + 1e-6)
        self.assertGreater(stats["max_offset_ms"], 20.0)

    def test_never_exceeds_tolerance(self):
        rate = 16000
        rng = np.random.default_rng(5)
        signal = rng.normal(0, 0.01, size=rate * 5)
        events = [
            {
                "id": f"e{i}",
                "phoneme": "AA",
                "start_time": 0.5 + i * 0.4,
                "duration": 0.1,
                "end_time": 0.6 + i * 0.4,
                "original_start_frame": i,
            }
            for i in range(10)
        ]
        aligned, _ = align.align_events(events, signal, rate, tolerance_ms=50.0)
        for event in aligned:
            self.assertLessEqual(abs(event["offset_ms"]), 50.0 + 1e-6)

    def test_silence_detection_and_coverage(self):
        rate = 16000
        signal = np.zeros(rate * 3, dtype=np.float64)
        signal[: rate] = 0.3
        # 1 second of pure silence in the middle
        signal[rate * 2 :] = 0.3
        segments = align.detect_silence(signal, rate, min_silence_ms=500.0)
        self.assertTrue(any(segment.duration >= 0.5 for segment in segments))

        events = [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 2.0, "end_time": 3.0},
        ]
        ratio = align.coverage_ratio(events, 3.0)
        self.assertAlmostEqual(ratio, 2.0 / 3.0)
        gaps = align.uncovered_segments(events, 3.0)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["type"], "uncovered_gap")


if __name__ == "__main__":
    unittest.main()
