from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from podcast_post.chapters import align_chapters
from podcast_post.ducking import DuckingConfig, apply_ducking
from podcast_post.loudness import normalize, true_peak_dbtp
from podcast_post.manifest import Chapter, parse_manifest
from podcast_post.timeline import (
    FadeInfo,
    equal_power_gains,
    mix_tracks,
    render_timeline,
)

from .helpers import FS, manifest_document, write_asset, write_manifest


def _single_clip(
    clip_id: str,
    track: str,
    source_in: int,
    source_out: int,
    timeline_in: int,
    timeline_out: int,
    gain_db: float = 0.0,
) -> dict:
    return {
        "id": clip_id,
        "track": track,
        "asset": "asset.wav",
        "source_in": source_in,
        "source_out": source_out,
        "timeline_in": timeline_in,
        "timeline_out": timeline_out,
        "gain_db": gain_db,
    }


class CrossfadeTests(unittest.TestCase):
    def test_equal_power_curve_is_sample_exact(self) -> None:
        for length in (1, 2, 17, 4800):
            outgoing, incoming = equal_power_gains(length)
            np.testing.assert_allclose(
                np.square(outgoing) + np.square(incoming),
                np.ones(length),
                rtol=0.0,
                atol=1e-12,
            )
            if length > 1:
                self.assertAlmostEqual(float(outgoing[-1]), 0.0, places=12)
                self.assertAlmostEqual(float(incoming[-1]), 1.0, places=12)

    def test_later_clip_overwrites_same_track_by_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_asset(root / "asset.wav", 1000, value=0.5)
            document = manifest_document(
                1000,
                [
                    _single_clip("a", "host", 0, 600, 0, 600),
                    _single_clip("b", "host", 0, 500, 500, 1000),
                ],
            )
            manifest = parse_manifest(write_manifest(root / "manifest.json", document))
            rendered = render_timeline(manifest)
            self.assertTrue(np.allclose(rendered.tracks["host"][500:600], 0.5))

    def test_crossfade_overrides_overlap_and_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_asset(root / "asset.wav", 1200, value=1.0)
            document = manifest_document(
                1200,
                [
                    _single_clip("a", "host", 0, 700, 0, 700),
                    _single_clip("b", "host", 100, 600, 600, 1100),
                ],
                edit_points=[
                    {
                        "id": "cut",
                        "track": "host",
                        "outgoing_clip": "a",
                        "incoming_clip": "b",
                        "at_sample": 700,
                        "fade_samples": 300,
                    }
                ],
            )
            manifest = parse_manifest(write_manifest(root / "manifest.json", document))
            rendered = render_timeline(manifest)
            self.assertEqual(len(rendered.fades), 1)
            fade = rendered.fades[0]
            self.assertEqual(fade.effective_samples, 100)
            self.assertTrue(fade.truncated)
            outgoing, incoming = equal_power_gains(100)
            np.testing.assert_allclose(
                rendered.tracks["host"][600:700, 0],
                outgoing + incoming,
                atol=2e-7,
            )

    def test_cross_track_mix_sums_float64(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_asset(root / "asset.wav", 200, value=0.25)
            document = manifest_document(
                200,
                [
                    _single_clip("h", "host", 0, 200, 0, 200),
                    _single_clip("g", "guest", 0, 200, 0, 200),
                ],
            )
            manifest = parse_manifest(write_manifest(root / "manifest.json", document))
            rendered = render_timeline(manifest)
            mixture = mix_tracks(rendered.tracks)
            self.assertEqual(mixture.dtype, np.float64)
            np.testing.assert_allclose(mixture, 0.5)

    def test_later_clip_still_wins_inside_an_earlier_transition(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_asset(root / "asset.wav", 1200, value=1.0)
            document = manifest_document(
                1200,
                [
                    _single_clip("a", "host", 0, 700, 0, 700),
                    _single_clip("b", "host", 100, 600, 600, 1100),
                    _single_clip("c", "host", 0, 100, 620, 720),
                ],
                edit_points=[
                    {
                        "id": "cut",
                        "track": "host",
                        "outgoing_clip": "a",
                        "incoming_clip": "b",
                        "at_sample": 700,
                        "fade_samples": 100,
                    }
                ],
            )
            manifest = parse_manifest(write_manifest(root / "manifest.json", document))
            rendered = render_timeline(manifest)
            self.assertTrue(np.allclose(rendered.tracks["host"][620:720], 1.0))


class DuckingTests(unittest.TestCase):
    def test_gate_counts_once_and_one_pole_attack_release(self) -> None:
        frames = FS
        host = np.zeros((frames, 1), dtype=np.float64)
        host[10000:30000] = 0.5
        music = np.ones((frames, 1), dtype=np.float64)
        config = DuckingConfig(
            enabled=True,
            trigger_track="host",
            target_track="music",
            threshold_dbfs=-25.0,
            energy_window_ms=10.0,
            depth_db=12.0,
            attack_ms=5.0,
            release_ms=50.0,
        )
        result = apply_ducking(
            {"host": host, "guest": np.zeros_like(host), "music": music},
            config,
            FS,
        )
        self.assertEqual(result.trigger_count, 1)
        self.assertEqual(result.attack_samples, 240)
        self.assertEqual(result.release_samples, 2400)
        self.assertGreater(float(np.min(music[10000:30000])), 0.20)
        self.assertLess(float(np.min(music[10000:30000])), 0.30)
        self.assertGreater(float(music[-1, 0]), 0.99)


class ChapterTests(unittest.TestCase):
    def test_boundary_inside_fade_moves_forward_and_spans_are_contiguous(self) -> None:
        manifest = SimpleNamespace(total_samples=1000)
        chapters = [
            Chapter("A", 0),
            Chapter("B", 150),
            Chapter("C", 400),
        ]
        fade = FadeInfo(
            edit_id="cut",
            track="host",
            outgoing_clip="a",
            incoming_clip="b",
            at_sample=200,
            requested_samples=100,
            effective_samples=100,
            fade_start=100,
            fade_end=200,
        )
        result = align_chapters(manifest, chapters, [fade])
        self.assertEqual([span.start_sample for span in result.spans], [0, 200, 400])
        self.assertEqual([span.end_sample for span in result.spans], [200, 400, 1000])
        self.assertEqual(len(result.adjustments), 1)


class LoudnessTests(unittest.TestCase):
    def test_normalizes_to_target_when_peak_allows(self) -> None:
        frames = 3 * FS
        time = np.arange(frames) / FS
        mono = 0.08 * np.sin(2.0 * np.pi * 997.0 * time)
        samples = np.column_stack((mono, mono))
        result = normalize(samples, FS, -16.0, -1.0)
        self.assertFalse(result.peak_limited)
        self.assertAlmostEqual(result.final_lufs, -16.0, delta=0.02)
        self.assertLessEqual(result.final_true_peak_dbtp, -1.0 + 1e-6)

    def test_peak_constraint_takes_priority(self) -> None:
        frames = 3 * FS
        time = np.arange(frames) / FS
        mono = 0.018 * np.sin(2.0 * np.pi * 411.0 * time)
        samples = np.column_stack((mono, mono))
        samples[FS, 0] = 0.99
        samples[2 * FS + 31, 1] = -0.98
        result = normalize(samples, FS, -16.0, -1.0)
        self.assertTrue(result.peak_limited)
        self.assertLess(result.final_lufs, -16.0)
        self.assertLessEqual(result.final_true_peak_dbtp, -1.0 + 1e-6)

    def test_true_peak_detects_inter_sample_peak(self) -> None:
        frames = 1000
        time = np.arange(frames) / FS
        samples = np.sin(2.0 * np.pi * 12000.0 * time + np.pi / 4.0) * 0.5
        sample_peak = 20.0 * np.log10(float(np.max(np.abs(samples))))
        measured = true_peak_dbtp(samples[:, None])
        self.assertGreater(measured, sample_peak)
        self.assertLess(measured, -5.9)


if __name__ == "__main__":
    unittest.main()

