from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from cinemix.mixing import (
    _apply_cut_crossfades,
    _build_ambience,
    _build_music,
    build_duck_envelope,
    seconds_to_samples,
)
from cinemix.scene import (
    AmbienceSpec,
    DialogueIntervalSpec,
    MusicLayerSpec,
    SceneSpec,
)


class TimelineTests(unittest.TestCase):
    def test_seconds_to_samples_is_deterministic(self) -> None:
        self.assertEqual(seconds_to_samples(0.0, 48000), 0)
        self.assertEqual(seconds_to_samples(0.001, 48000), 48)
        self.assertEqual(seconds_to_samples(0.333, 48000), 15984)
        self.assertEqual(seconds_to_samples(7.5, 48000), 360000)

    def test_ambience_and_music_loop_points_do_not_drift(self) -> None:
        sample_rate = 48000
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            loop_asset = np.zeros((4800, 1), dtype=np.float64)
            loop_asset[0, 0] = 1.0
            sf.write(
                str(root / "loop.wav"),
                loop_asset,
                sample_rate,
                subtype="PCM_16",
            )
            sf.write(
                str(root / "music.wav"),
                loop_asset,
                sample_rate,
                subtype="PCM_16",
            )
            scene = SceneSpec(
                path=root / "scene.json",
                scene_id="loops",
                duration=0.5,
                ambience=AmbienceSpec(
                    file=root / "loop.wav",
                    loop_start=0.0,
                    loop_end=0.1,
                    gain=1.0,
                ),
                events=(),
                music_layers=(
                    MusicLayerSpec(
                        file=root / "music.wav",
                        start=0.05,
                        loops=3,
                        gain=1.0,
                    ),
                ),
                dialogue_intervals=(),
                camera_cuts=(),
            )
            cache = {}
            ambience = _build_ambience(scene, sample_rate, 24000, cache)
            music = _build_music(scene, sample_rate, 24000, cache)
            ambience_hits = np.flatnonzero(np.abs(ambience[:, 0]) > 0.5)
            music_hits = np.flatnonzero(np.abs(music[:, 0]) > 0.5)
            self.assertEqual(ambience_hits.tolist(), list(range(0, 24000, 4800)))
            self.assertEqual(music_hits.tolist(), [2400, 7200, 12000])


class DuckingTests(unittest.TestCase):
    def test_equal_power_ducking_is_continuous_at_boundaries(self) -> None:
        sample_rate = 48000
        total = sample_rate * 4
        interval = DialogueIntervalSpec(1.0, 2.5, 12.0)
        envelope, reports = build_duck_envelope(
            [interval], total, sample_rate, transition_ms=50.0
        )
        start = 48000
        end = 120000
        fade = 2400
        duck = 10.0 ** (-12.0 / 20.0)

        self.assertEqual(envelope.shape, (total,))
        self.assertTrue(np.allclose(envelope[:start], 1.0))
        self.assertAlmostEqual(float(envelope[start]), 1.0, places=15)
        self.assertAlmostEqual(float(envelope[start + fade]), duck, places=12)
        self.assertAlmostEqual(float(envelope[end - fade]), duck, places=12)
        self.assertAlmostEqual(float(envelope[end - 1]), 1.0, places=12)
        self.assertEqual(float(envelope[end]), 1.0)
        self.assertLess(float(np.max(np.abs(np.diff(envelope)))), 0.001)
        self.assertEqual(reports[0].attack_samples, fade)
        self.assertEqual(reports[0].release_samples, fade)
        self.assertAlmostEqual(reports[0].gain_at_start, 1.0, places=15)
        self.assertAlmostEqual(reports[0].gain_at_end, 1.0, places=15)

    def test_overlapping_dialogues_take_deeper_duck(self) -> None:
        sample_rate = 48000
        total = sample_rate * 3
        intervals = [
            DialogueIntervalSpec(0.5, 1.5, 6.0),
            DialogueIntervalSpec(1.2, 2.2, 15.0),
        ]
        envelope, _ = build_duck_envelope(intervals, total, sample_rate)
        deepest = 10.0 ** (-15.0 / 20.0)
        self.assertAlmostEqual(float(envelope[72000]), deepest, places=12)


class CutCrossfadeTests(unittest.TestCase):
    def test_cut_bridge_is_symmetric_and_preserves_length_and_edges(self) -> None:
        sample_rate = 48000
        total = sample_rate * 2
        center = 48000
        signal = np.zeros((total, 2), dtype=np.float64)
        signal[:center, 0] = -0.4
        signal[center:, 0] = 0.45
        signal[:center, 1] = 0.2
        signal[center:, 1] = -0.25
        original = signal.copy()

        repaired, checks = _apply_cut_crossfades(
            signal, [1.0], sample_rate, transition_ms=20.0
        )

        self.assertEqual(repaired.shape, original.shape)
        self.assertEqual(len(checks), 1)
        check = checks[0]
        self.assertEqual(check.fade_samples, 960)
        self.assertEqual(check.window_start_sample, center - 480)
        self.assertEqual(check.window_end_sample, center + 480)
        self.assertEqual(check.pre_samples, 480)
        self.assertEqual(check.post_samples, 480)
        self.assertTrue(check.passed)
        self.assertLessEqual(check.equal_power_max_error, 1e-12)
        self.assertLessEqual(check.boundary_continuity_error, 1e-12)
        self.assertAlmostEqual(
            float(repaired[center - 480, 0]), float(original[center - 480, 0]), places=15
        )
        self.assertAlmostEqual(
            float(repaired[center + 479, 0]),
            float(original[center + 479, 0]),
            places=15,
        )
        self.assertTrue(np.allclose(repaired[: center - 480], original[: center - 480]))
        self.assertTrue(np.allclose(repaired[center + 480 :], original[center + 480 :]))
        self.assertLess(check.max_abs_step_after, 0.05)


if __name__ == "__main__":
    unittest.main()
