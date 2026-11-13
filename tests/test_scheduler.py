import unittest

import numpy as np

from audiopack.assets import LoadedClip
from audiopack.config import Config
from audiopack.manifest import Event, parse_manifest
from audiopack.scheduler import schedule_language


def clip(voice_key: str, frames: int, tmpdir: str) -> LoadedClip:
    return LoadedClip(
        path=f"{tmpdir}/{voice_key}_zh.wav",
        samples=np.zeros(frames, dtype=np.float32),
        sample_rate=48000,
        frames=frames,
        content_hash=f"hash-{voice_key}",
        source_name=f"{voice_key}_zh.wav",
    )


def event(event_id, time_ms, priority, voice_key, max_delay_ms=500):
    return Event(
        event_id=event_id,
        event_type="t",
        time_ms=time_ms,
        priority=priority,
        voice_key=voice_key,
        langs=("zh",),
        max_delay_ms=max_delay_ms,
        tone=None,
    )


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        raw = {"events": [
            {"id": "A", "type": "t", "time_ms": 0, "priority": 1},
        ]}
        self.manifest = parse_manifest(raw, self.config)

    def test_high_priority_keeps_natural_position(self) -> None:
        events = [
            # LOW naturally spans 0..100ms; HIGH at 50ms overlaps it.
            event("LOW", 0, 1, "a", max_delay_ms=1000),
            event("HIGH", 50, 10, "b"),
        ]
        clips = {"a": clip("a", 4800, "/tmp"), "b": clip("b", 2400, "/tmp")}
        result = schedule_language("zh", events, self.manifest, clips, self.config)
        by_id = {p.event_id: p for p in result.placements}
        # HIGH starts at its natural 100ms point.
        self.assertEqual(by_id["HIGH"].start, 50 * 48)
        self.assertEqual(by_id["HIGH"].shift_samples, 0)
        # LOW (0..100ms natural) is pushed as a whole past HIGH's end.
        self.assertEqual(by_id["LOW"].start, by_id["HIGH"].end)
        self.assertEqual(by_id["LOW"].shift_samples, by_id["HIGH"].end)
        self.assertEqual(result.dropped, [])

    def test_shift_never_moves_higher_priority_event(self) -> None:
        events = [
            # H1 0..100ms; L at 40ms (overlaps H1); H2 at 200ms stays natural.
            event("H1", 0, 9, "a"),
            event("L", 40, 1, "c", max_delay_ms=2000),
            event("H2", 200, 9, "b"),
        ]
        clips = {
            "a": clip("a", 4800, "/tmp"),
            "b": clip("b", 4800, "/tmp"),
            "c": clip("c", 2400, "/tmp"),
        }
        result = schedule_language("zh", events, self.manifest, clips, self.config)
        by_id = {p.event_id: p for p in result.placements}
        self.assertEqual(by_id["H1"].start, 0)
        self.assertEqual(by_id["H2"].start, 200 * 48)
        # Low event lands after the high-priority chain, never before/over it.
        self.assertGreaterEqual(by_id["L"].start, by_id["H1"].end)

    def test_event_beyond_max_delay_is_dropped(self) -> None:
        events = [
            event("LOW", 0, 1, "a", max_delay_ms=50),
            event("HIGH", 10, 10, "b"),
        ]
        clips = {"a": clip("a", 4800, "/tmp"), "b": clip("b", 9600, "/tmp")}
        result = schedule_language("zh", events, self.manifest, clips, self.config)
        self.assertEqual([p.event_id for p in result.placements], ["HIGH"])
        self.assertEqual(len(result.dropped), 1)
        dropped = result.dropped[0]
        self.assertEqual(dropped.event_id, "LOW")
        self.assertEqual(dropped.reason, "exceeds_max_delay")
        self.assertGreater(dropped.required_delay_samples,
                           dropped.max_delay_samples)

    def test_no_overlap_when_gap_exists(self) -> None:
        events = [
            event("A", 0, 5, "a"),
            event("B", 500, 5, "b"),
        ]
        clips = {"a": clip("a", 4800, "/tmp"), "b": clip("b", 4800, "/tmp")}
        result = schedule_language("zh", events, self.manifest, clips, self.config)
        by_id = {p.event_id: p for p in result.placements}
        self.assertEqual(by_id["A"].shift_samples, 0)
        self.assertEqual(by_id["B"].shift_samples, 0)

    def test_ties_are_deterministic_by_time_then_id(self) -> None:
        events = [
            event("Z", 100, 5, "b"),
            event("A", 100, 5, "a"),
        ]
        clips = {"a": clip("a", 2400, "/tmp"), "b": clip("b", 2400, "/tmp")}
        first = schedule_language("zh", events, self.manifest, clips, self.config)
        second = schedule_language("zh", list(reversed(events)),
                                   self.manifest, clips, self.config)
        order_first = [p.event_id for p in first.placements]
        order_second = [p.event_id for p in second.placements]
        self.assertEqual(order_first, order_second)
        self.assertEqual(order_first[0], "A")

    def test_placements_never_overlap(self) -> None:
        events = [
            event("A", 0, 3, "a"),
            event("B", 20, 7, "b"),
            event("C", 40, 2, "c", max_delay_ms=5000),
        ]
        clips = {
            "a": clip("a", 4800, "/tmp"),
            "b": clip("b", 4800, "/tmp"),
            "c": clip("c", 4800, "/tmp"),
        }
        result = schedule_language("zh", events, self.manifest, clips, self.config)
        placements = sorted(result.placements, key=lambda p: p.start)
        for left, right in zip(placements, placements[1:]):
            self.assertLessEqual(left.end, right.start)


if __name__ == "__main__":
    unittest.main()
