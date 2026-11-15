"""End-to-end and unit tests for the cutscene pipeline.

Runs with the stdlib unittest runner:
    python3 -m unittest discover -s tests -v
Requires ffmpeg/ffprobe on PATH for the mux-related checks.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cutscene_pipeline import cli, dsp, synth, wavfile

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "project.json"
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


class TestDspPrimitives(unittest.TestCase):
    def test_synth_deterministic(self):
        a = synth.synth_event("evt", "src", 1.0)
        b = synth.synth_event("evt", "src", 1.0)
        np.testing.assert_array_equal(a, b)
        c = synth.synth_event("evt2", "src", 1.0)
        self.assertFalse(np.array_equal(a, c))

    def test_resample_length_and_position(self):
        # Impulse at sample 4410 (0.1 s @ 44.1 kHz) must land at 4800 (0.1 s @ 48 kHz).
        x = np.zeros(44100)
        x[4410] = 1.0
        y = dsp.resample_rational(x, 160, 147)
        self.assertAlmostEqual(len(y), 48000, delta=2)
        peak = int(np.argmax(np.abs(y)))
        self.assertLessEqual(abs(peak - 4800), 1)

    def test_true_peak_known_sine(self):
        sr = 48000
        t = np.arange(sr) / sr
        sig = 0.5 * np.sin(2 * np.pi * 997 * t)
        tp = dsp.true_peak_dbtp(sig[:, None])
        self.assertAlmostEqual(tp, 20 * math.log10(0.5), delta=0.2)

    def test_loudness_silence(self):
        self.assertEqual(dsp.integrated_loudness(np.zeros((48000, 2))), float("-inf"))

    def test_peak_limiter(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((48000, 2)) * 0.9
        y = dsp.peak_limit(x, 0.5)
        self.assertLessEqual(float(np.max(np.abs(y))), 0.5 + 1e-9)


class TestCliFailure(unittest.TestCase):
    def test_malformed_json_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(cli.run(str(bad), str(Path(td) / "out"), False), 2)

    def test_empty_timeline_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            proj = {
                "tracks": {"a": {"gain_db": 0.0}},
                "video": {"width": 64, "height": 64, "fps": 10},
                "timeline": {"events": []},
            }
            p = Path(td) / "p.json"
            p.write_text(json.dumps(proj), encoding="utf-8")
            self.assertEqual(cli.run(str(p), str(Path(td) / "out"), False), 2)


class TestEndToEnd(unittest.TestCase):
    """Runs the example project once (normal mode) and checks the
    acceptance criteria: alignment <= 1 ms, -16 +/- 0.5 LUFS,
    TP <= -1 dBTP, skip accounting, chapters and the merged report."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name) / "normal"
        cls.exit_code = cli.run(str(EXAMPLE), str(cls.out), dry_run=False)
        cls.report = json.loads((cls.out / "report.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_exit_code_partial(self):
        self.assertEqual(self.exit_code, 1)  # warnings/skips present

    def test_skip_accounting(self):
        r = self.report
        self.assertEqual(r["input_event_count"], 11)
        self.assertEqual(r["processed_event_count"], 7)
        self.assertEqual(r["skipped_event_count"], 4)
        reasons = {e["reason"] for e in r["skipped_events"]}
        self.assertEqual(reasons, {"negative_duration", "unknown_track",
                                   "missing_source", "format_mismatch"})
        warn_types = {w["type"] for w in r["warnings"]}
        self.assertIn("overlap", warn_types)
        self.assertIn("chapter_out_of_range", warn_types)

    def test_alignment_error_within_1ms(self):
        for entry in self.report["timeline"]["alignment"]:
            self.assertLessEqual(abs(entry["offset_error_ms"]), 1.0)
        # Independently verify onsets in the written per-track WAVs.
        data, rate, channels, _ = wavfile.read_wav(self.out / "aligned" / "track_dialogue.wav")
        self.assertEqual(rate, 48000)
        self.assertEqual(channels, 1)
        mono = data if data.ndim == 1 else data[:, 0]
        onset = int(np.argmax(np.abs(mono) > 0.01 * np.max(np.abs(mono))))
        expected = round(0.5 * 48000)  # first dialogue event starts at 0.5 s
        self.assertLessEqual(abs(onset - expected), 48)  # 1 ms @ 48 kHz

    def test_mix_output_format_and_peaks(self):
        _, rate, channels, width = wavfile.read_wav(self.out / "mix_48k_24bit_stereo.wav")
        self.assertEqual((rate, channels, width), (48000, 2, 3))
        mix = self.report["mix"]
        self.assertLess(mix["mix_peak_linear"], 1.0)  # no clipping
        self.assertEqual(set(mix["tracks"]), {"dialogue", "sfx", "music"})
        for info in mix["tracks"].values():
            self.assertIn("peak_dbfs", info)

    def test_loudness_targets(self):
        after = self.report["loudness"]["after"]
        self.assertAlmostEqual(after["integrated_lufs"], -16.0, delta=0.5)
        self.assertLessEqual(after["true_peak_dbtp"], -1.0)
        # Re-measure from the written 24-bit WAV.
        data, rate, _, _ = wavfile.read_wav(self.out / "normalized_48k_24bit_stereo.wav")
        stereo = data if data.ndim == 2 else np.column_stack([data, data])
        lufs = dsp.integrated_loudness(stereo, rate)
        tp = dsp.true_peak_dbtp(stereo)
        self.assertAlmostEqual(lufs, -16.0, delta=0.5)
        self.assertLessEqual(tp, -1.0)

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg not on PATH")
    def test_loudness_crosscheck_ffmpeg(self):
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i",
             str(self.out / "normalized_48k_24bit_stereo.wav"),
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True)
        tail = proc.stderr.split("Summary:")[-1]
        lufs = None
        tp = None
        for line in tail.splitlines():
            s = line.strip()
            if s.startswith("I:"):
                lufs = float(s.split()[1])
            if s.startswith("Peak:"):
                tp = float(s.split()[1])
        self.assertIsNotNone(lufs)
        self.assertIsNotNone(tp)
        self.assertAlmostEqual(lufs, -16.0, delta=0.5)
        self.assertLessEqual(tp, -1.0 + 0.3)  # ffmpeg oversampling differs slightly

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg not on PATH")
    def test_mp4_and_chapters(self):
        mp4 = self.out / "cutscene.mp4"
        self.assertTrue(mp4.exists())
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_chapters", "-show_format", str(mp4)],
            capture_output=True, text=True)
        info = json.loads(proc.stdout)
        chapters = info.get("chapters", [])
        self.assertEqual(len(chapters), 3)  # out-of-range chapter dropped
        titles = [c["tags"]["title"] for c in chapters]
        self.assertEqual(titles, ["Opening", "Confrontation", "Resolution"])
        duration = float(info["format"]["duration"])
        self.assertAlmostEqual(duration, 12.0, delta=0.2)
        for c in chapters:
            self.assertLess(float(c["start_time"]), duration)

    def test_report_structure(self):
        for key in ("input_event_count", "skipped_event_count", "stage_timings_s",
                    "loudness", "mix", "chapters", "warnings", "timeline",
                    "validation", "files", "mode"):
            self.assertIn(key, self.report)
        for stage in ("parse_validate", "align", "mix", "loudness", "mux", "total"):
            self.assertIn(stage, self.report["stage_timings_s"])


class TestDryRun(unittest.TestCase):
    def test_dry_run_writes_no_media(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dry"
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli.run(str(EXAMPLE), str(out), dry_run=True)
            self.assertEqual(code, 1)
            media = [p for p in out.rglob("*") if p.is_file()] if out.exists() else []
            self.assertEqual(media, [])  # no files at all in dry-run
            report = json.loads(buf.getvalue())
            self.assertEqual(report["mode"], "dry-run")
            # Structure identical to normal mode: same top-level keys.
            self.assertEqual(set(report), {
                "report_version", "tool", "mode", "project",
                "input_event_count", "processed_event_count",
                "skipped_event_count", "skipped_events", "validation",
                "timeline", "mix", "loudness", "chapters", "mux",
                "stage_timings_s", "warnings", "files"})
            self.assertIsNotNone(report["loudness"]["after"]["integrated_lufs"])


if __name__ == "__main__":
    unittest.main()
