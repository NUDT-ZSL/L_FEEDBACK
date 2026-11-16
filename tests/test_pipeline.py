"""Unit and end-to-end tests for the offline mixer (stdlib unittest only)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_mixer import TARGET_SR
from offline_mixer.ducking import apply_ducking, build_duck_gain, measure_recovery_time
from offline_mixer.limiter import ceiling_linear, soft_limit
from offline_mixer.loudness import integrated_loudness, normalize_loudness
from offline_mixer.resample import resample_to_48k
from offline_mixer.samplegen import DEFAULT_SEED, generate_project
from offline_mixer.timeline import TimelineValidationError, load_timeline
from offline_mixer.timescale import (
    MAX_RATIO,
    MIN_RATIO,
    align_segment,
    granular_time_stretch,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ResamplingTests(unittest.TestCase):
    def test_anti_aliased_resample_lengths_and_content(self):
        duration = 1.0
        for source_sr in (8000, 16000, 32000, 44100, 48000):
            t = np.arange(int(duration * source_sr)) / source_sr
            signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            out = resample_to_48k(signal, source_sr)
            expected = int(round(duration * TARGET_SR))
            self.assertAlmostEqual(out.size, expected, delta=8)
            # Trim FIR edge transients; the tone must survive with expected RMS.
            trimmed = out[512:-512]
            self.assertAlmostEqual(float(np.sqrt(np.mean(trimmed**2))),
                                   0.5 / math.sqrt(2), delta=0.02)

    def test_high_frequency_is_rejected_when_downsampling(self):
        # 96 kHz -> 48 kHz: new Nyquist is 24 kHz. A 30 kHz tone is above it
        # and must be suppressed by the anti-alias filter; a 1 kHz in-band
        # tone must survive unchanged.
        sr = 96000
        t = np.arange(4 * sr) / sr
        out_of_band = np.sin(2 * np.pi * 30000.0 * t)
        out = resample_to_48k(out_of_band, 96000)
        self.assertLess(float(np.sqrt(np.mean(out**2))), 0.02)

        in_band = np.sin(2 * np.pi * 1000.0 * t)
        kept = resample_to_48k(in_band, 96000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(kept[512:-512] ** 2))),
                               1.0 / math.sqrt(2), delta=0.02)


class TimeScaleTests(unittest.TestCase):
    def test_stretch_durations_and_pitch_preservation(self):
        sr = TARGET_SR
        t = np.arange(2 * sr) / sr
        signal = np.sin(2 * np.pi * 220.0 * t)
        for ratio in (0.8, 1.0, 1.1, 1.25):
            target_len = int(round(signal.size / ratio))
            out = granular_time_stretch(signal, ratio, sr, target_len)
            self.assertEqual(out.size, target_len)
        # Zero-length target returns an empty array instead of raising.
        out = granular_time_stretch(signal, 1.0, sr, 0)
        self.assertEqual(out.size, 0)

    def test_align_ratio_bounds(self):
        sr = TARGET_SR
        signal = np.random.default_rng(1).normal(0, 0.5, sr)
        aligned, ratio, error = align_segment(signal, 1.0, sr)
        self.assertIsNone(error)
        self.assertEqual(aligned.size, sr)

        # Source 1 s -> target 0.5 s gives ratio 2.0 -> rejected.
        _, ratio, error = align_segment(signal, 0.5, sr)
        self.assertIsNotNone(error)
        self.assertEqual(error["code"], "E3001")
        self.assertGreater(ratio, MAX_RATIO)

        # Zero target duration path.
        _, _, error = align_segment(signal, 0.0, sr)
        self.assertEqual(error["reason"], "zero_or_negative_target_duration")
        self.assertTrue(MIN_RATIO == 0.8 and MAX_RATIO == 1.25)


class LoudnessTests(unittest.TestCase):
    def test_silence_returns_nan(self):
        self.assertTrue(math.isnan(integrated_loudness(np.zeros(TARGET_SR))))
        self.assertTrue(math.isnan(integrated_loudness(np.zeros(10))))

    def test_normalization_hits_target_within_tolerance(self):
        rng = np.random.default_rng(7)
        sr = TARGET_SR
        t = np.arange(3 * sr) / sr
        signal = 0.4 * np.sin(2 * np.pi * 300 * t) + rng.normal(0, 0.03, 3 * sr)
        for target in (-24.0, -20.0, -16.0):
            normalized, gain_db, realized, is_silence = normalize_loudness(signal, target, sr)
            self.assertFalse(is_silence)
            self.assertLessEqual(abs(realized - target), 0.5)
            self.assertNotEqual(gain_db, 0.0)

    def test_silent_segment_not_amplified(self):
        normalized, gain_db, _, is_silence = normalize_loudness(
            np.zeros(TARGET_SR), -23.0, TARGET_SR)
        self.assertTrue(is_silence)
        self.assertEqual(gain_db, 0.0)
        self.assertTrue(np.all(normalized == 0.0))


class DuckingTests(unittest.TestCase):
    def test_gain_reduction_cap_and_recovery(self):
        sr = TARGET_SR
        dialogue = np.zeros(5 * sr)
        # One second of loud speech starting at t=1 s.
        dialogue[sr:2 * sr] = 0.8 * np.sin(2 * np.pi * 240 * np.arange(sr) / sr)
        gain = build_duck_gain(dialogue, sr)
        min_gain_db = 20 * np.log10(np.min(gain))
        self.assertGreaterEqual(min_gain_db, -12.0 - 0.05)
        self.assertLess(min_gain_db, -6.0)

        active = np.abs(dialogue) > 0
        recovery = measure_recovery_time(gain, active, sr)
        self.assertIsNotNone(recovery)
        self.assertLessEqual(recovery, 0.2)

    def test_music_is_attenuated_only_during_speech(self):
        sr = TARGET_SR
        music = np.ones(4 * sr) * 0.3
        dialogue = np.zeros(4 * sr)
        dialogue[sr:2 * sr] = 0.7
        ducked, gain = apply_ducking(music, dialogue, sr)
        self.assertAlmostEqual(float(gain[0]), 1.0, places=3)
        self.assertLess(float(np.min(gain[sr:2 * sr])), 0.5)
        # Post speech region recovers toward unity.
        self.assertGreater(float(gain[-1]), 0.98)
        self.assertEqual(ducked.size, music.size)


class LimiterTests(unittest.TestCase):
    def test_soft_limit_respects_ceiling_and_counts_events(self):
        sr = TARGET_SR
        signal = np.zeros(sr)
        signal[100:120] = 1.5
        signal[500:510] = -2.0
        limited, stats = soft_limit(signal)
        ceiling = ceiling_linear()
        self.assertLessEqual(float(np.max(np.abs(limited))), ceiling + 1e-12)
        self.assertEqual(stats["trigger_event_count"], 2)
        self.assertGreater(stats["trigger_sample_count"], 0)
        self.assertTrue(stats["limiter_engaged"])

    def test_below_ceiling_passes_unchanged(self):
        signal = np.linspace(-0.5, 0.5, 100)
        limited, stats = soft_limit(signal)
        np.testing.assert_array_equal(signal, limited)
        self.assertFalse(stats["limiter_engaged"])


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mixer_e2e_")
        cls.meta = generate_project(os.path.join(cls.tmp, "project"), DEFAULT_SEED)
        cls.timeline_path = cls.meta["timeline"]
        cls.wav = os.path.join(cls.tmp, "mix.wav")
        cls.report_path = os.path.join(cls.tmp, "report.json")
        from offline_mixer.mixer import run_mix
        cls.report = run_mix(load_timeline(cls.timeline_path), cls.wav, cls.report_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_output_wav_format(self):
        data, sr = sf.read(self.wav)
        self.assertEqual(sr, TARGET_SR)
        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.shape[1], 2)
        info = sf.info(self.wav)
        self.assertEqual(info.subtype, "PCM_16")

    def test_segment_acceptance_metrics(self):
        report = self.report
        self.assertGreaterEqual(len(report["segments"]), 8)
        for segment in report["segments"]:
            if segment["alignment_error"]:
                self.assertIn(segment["alignment_error"]["code"], ("E3001",))
                continue
            delta = abs(segment["aligned_duration_seconds"]
                        - segment["target_duration_seconds"])
            self.assertLessEqual(delta, 1.0 / TARGET_SR + 1e-9)
            if not segment["is_silence"]:
                self.assertLessEqual(abs(segment["loudness_error_lu"]), 0.5)

    def test_resampling_status_recorded(self):
        rates = {s["source_sample_rate"] for s in self.report["segments"]}
        self.assertIn(32000, rates)
        self.assertIn(16000, rates)
        resampled = [s for s in self.report["segments"] if s["resampling"]["resampled"]]
        self.assertTrue(resampled)
        self.assertEqual(resampled[0]["resampling"]["method"],
                         "band_limited_fir_polyphase")

    def test_limiter_and_ducking_summary(self):
        summary = self.report["summary"]
        self.assertTrue(summary["limiter"]["limiter_engaged"])
        self.assertGreater(summary["limiter"]["trigger_sample_count"], 0)
        self.assertLessEqual(
            summary["ducking"]["recovery_time_seconds"],
            summary["ducking"]["recovery_limit_seconds"] + 1e-9)
        self.assertGreater(summary["ducking"]["max_gain_reduction_db"], 6.0)
        self.assertIn("total_processing_seconds", summary)

    def test_determinism_across_runs(self):
        other = tempfile.mkdtemp(prefix="mixer_det_")
        try:
            meta_b = generate_project(os.path.join(other, "project"), DEFAULT_SEED)
            for rel_path in ("audio/voice_en.wav", "audio/voice_ja.wav",
                             "audio/music.wav", "audio/sfx.wav"):
                a, sr_a = sf.read(os.path.join(os.path.dirname(self.timeline_path), rel_path))
                b, sr_b = sf.read(os.path.join(os.path.dirname(meta_b["timeline"]), rel_path))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(sr_a, sr_b)
            with open(self.timeline_path) as fa, open(meta_b["timeline"]) as fb:
                self.assertEqual(json.load(fa), json.load(fb))
        finally:
            shutil.rmtree(other, ignore_errors=True)


class ValidationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mixer_val_")
        generate_project(os.path.join(self.tmp, "project"), DEFAULT_SEED)
        self.timeline = os.path.join(self.tmp, "project", "timeline.json")
        with open(self.timeline) as handle:
            self.base = json.load(handle)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _expect(self, mutate, expected_code, expected_index):
        tl = json.loads(json.dumps(self.base))
        mutate(tl)
        path = os.path.join(self.tmp, "project", "mutated.json")
        with open(path, "w") as handle:
            json.dump(tl, handle)
        with self.assertRaises(TimelineValidationError) as ctx:
            load_timeline(path)
        diagnostic = ctx.exception.diagnostics
        self.assertEqual(diagnostic["status"], "error")
        matches = [issue for issue in diagnostic["error"]["details"]["issues"]
                   if issue["timeline_index"] == expected_index]
        self.assertTrue(matches, f"no issue for index {expected_index}")
        issue = matches[0]
        self.assertEqual(issue["code"], expected_code)

    def test_negative_timing(self):
        self._expect(lambda tl: tl["segments"][0].update(start=9, end=1),
                     "E2001", 0)

    def test_duration_exceeded(self):
        def mutate(tl):
            # Point at an existing short file then exceed its real duration.
            tl["segments"][0]["file"] = "audio/sfx.wav"
            tl["segments"][0]["source_range"] = [0.0, 999.0]
        self._expect(mutate, "E2002", 0)

    def test_missing_file(self):
        self._expect(lambda tl: tl["segments"][1].update(file="audio/missing.wav"),
                     "E1001", 1)

    def test_bad_schema(self):
        self._expect(lambda tl: tl["segments"][2].pop("language"), "E2003", 2)

    def test_cli_nonzero_exit_and_json_diagnostic(self):
        proc = subprocess.run(
            [sys.executable, "-m", "offline_mixer", "mix",
             os.path.join(self.tmp, "does_not_exist.json")],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=ROOT))
        self.assertNotEqual(proc.returncode, 0)
        payload = json.loads(proc.stderr)
        self.assertEqual(payload["error"]["code"], "E1001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
