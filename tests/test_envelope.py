from __future__ import annotations

import unittest

import numpy as np

from dialogue_mixer.envelope import (
    DuckEvent,
    build_track_envelopes,
    db_to_linear,
    find_interval_overlaps,
)


class DuckEnvelopeTests(unittest.TestCase):
    def test_attack_sustain_and_release_are_sample_accurate(self) -> None:
        event = DuckEvent(
            dialogue_id="line",
            target_track="music",
            start_sample=100,
            end_sample=200,
            attack_samples=10,
            release_samples=20,
            depth_db=-12.0,
            attack_curve="linear",
            release_curve="linear",
        )
        envelope = np.ones(400, dtype=np.float64)
        event.apply(envelope)
        depth = db_to_linear(-12.0)

        self.assertEqual(envelope[99], 1.0)
        self.assertAlmostEqual(envelope[100], 1.0 + (depth - 1.0) * 0.1)
        self.assertAlmostEqual(envelope[109], depth)
        self.assertAlmostEqual(envelope[110], depth)
        self.assertAlmostEqual(envelope[199], depth)
        self.assertAlmostEqual(envelope[200], depth + (1.0 - depth) / 20.0)
        self.assertAlmostEqual(envelope[219], 1.0)
        self.assertEqual(envelope[220], 1.0)

    def test_overlap_merge_uses_per_sample_minimum(self) -> None:
        shallow = DuckEvent(
            "a",
            "music",
            100,
            300,
            0,
            0,
            -6.0,
            "linear",
            "linear",
        )
        deep = DuckEvent(
            "b",
            "music",
            200,
            400,
            0,
            10,
            -20.0,
            "linear",
            "linear",
        )
        envelopes = build_track_envelopes(
            [shallow, deep], ["music", "sfx"], total_samples=500
        )
        music = envelopes["music"]
        self.assertAlmostEqual(music[100], db_to_linear(-6.0))
        self.assertAlmostEqual(music[200], db_to_linear(-20.0))
        self.assertAlmostEqual(music[250], db_to_linear(-20.0))
        self.assertAlmostEqual(music[399], db_to_linear(-20.0))
        self.assertGreater(music[400], db_to_linear(-20.0))
        self.assertAlmostEqual(music[409], 1.0)
        self.assertAlmostEqual(music[410], 1.0)
        self.assertTrue(np.all(envelopes["sfx"] == 1.0))

    def test_interval_overlap_detection(self) -> None:
        overlaps = find_interval_overlaps(
            [
                ("a", 0, 100),
                ("b", 50, 150),
                ("c", 120, 160),
            ]
        )
        self.assertEqual(len(overlaps), 2)
        self.assertEqual(overlaps[0]["dialogue_a"], "a")
        self.assertEqual(overlaps[0]["dialogue_b"], "b")
        self.assertEqual(overlaps[1]["dialogue_a"], "b")
        self.assertEqual(overlaps[1]["dialogue_b"], "c")


if __name__ == "__main__":
    unittest.main()

