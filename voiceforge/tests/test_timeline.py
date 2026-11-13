import unittest

from voiceforge import OUTPUT_SAMPLE_RATE
from voiceforge.timeline import build_timeline


def _interval(interval_id, player_id, start_ms, end_ms):
    from voiceforge.resampling import ms_to_samples
    return {
        "interval_id": interval_id,
        "player_id": player_id,
        "segment_id": interval_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "threshold": 0.1,
        "gain": 1.0,
        "_start_sample": ms_to_samples(start_ms),
        "_end_sample": ms_to_samples(end_ms),
    }


class TimelineTests(unittest.TestCase):
    def test_overlapping_intervals_become_disjoint_events(self):
        intervals = [
            _interval("a", "P1", 0.0, 100.0),
            _interval("b", "P2", 50.0, 150.0),
        ]
        loudness = {
            "a": {"peak": 0.2, "rms": 0.1, "new_clipped_samples": 0},
            "b": {"peak": 0.3, "rms": 0.2, "new_clipped_samples": 5},
        }
        events = build_timeline(intervals, set(), loudness,
                                OUTPUT_SAMPLE_RATE // 5, OUTPUT_SAMPLE_RATE)
        boundaries = [0]
        for event in events:
            self.assertEqual(event["start_sample"], boundaries[-1])
            boundaries.append(event["end_sample"])
        self.assertEqual(len(events), 3)
        self.assertFalse(events[0]["simultaneous"])
        self.assertTrue(events[1]["simultaneous"])
        self.assertEqual(events[1]["players"], ["P1", "P2"])
        self.assertEqual(events[1]["participants"][0]["interval_id"], "a")
        self.assertTrue(events[1]["any_clipped"])

    def test_missing_player_is_flagged(self):
        intervals = [_interval("g", "PX", 0.0, 50.0)]
        loudness = {"g": {"peak": 0.0, "rms": 0.0,
                          "new_clipped_samples": 0}}
        events = build_timeline(intervals, {"PX"}, loudness,
                                OUTPUT_SAMPLE_RATE // 10, OUTPUT_SAMPLE_RATE)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["any_missing_source"])
        self.assertTrue(events[0]["participants"][0]["missing_source"])


if __name__ == "__main__":
    unittest.main()

