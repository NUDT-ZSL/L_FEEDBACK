import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiopipe import wavio
from audiopipe.cli import main
from audiopipe.checkpoint import CheckpointStore, parameter_fingerprint, CHECKPOINT_VERSION
from audiopipe.errors import WavFormatError
from audiopipe.generate import generate
from audiopipe.mixer import ms_to_samples, db_to_linear, pcm16_from_float
from audiopipe.resample import resample, expected_output_len

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TempCase(unittest.TestCase):
    def setUp(self):
        os.makedirs(os.path.join(ROOT, ".work"), exist_ok=True)
        self.base = tempfile.mkdtemp(prefix="audiopipe_", dir=os.path.join(ROOT, ".work"))
        self.mat = os.path.join(self.base, "mat")
        self.out = os.path.join(self.base, "out")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def gen(self, **kw):
        return generate(self.mat, languages=("en", "ja"), clip_count=3, **kw)


class GeneratorTests(TempCase):
    def test_byte_identical_regeneration(self):
        self.gen(seed=42)
        first = {}
        for root, _d, files in os.walk(self.mat):
            for name in files:
                p = os.path.join(root, name)
                first[os.path.relpath(p, self.mat)] = wavio.file_sha256(p)
        shutil.rmtree(self.mat)
        self.gen(seed=42)
        second = {}
        for root, _d, files in os.walk(self.mat):
            for name in files:
                p = os.path.join(root, name)
                second[os.path.relpath(p, self.mat)] = wavio.file_sha256(p)
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_different_seed_changes_content(self):
        self.gen(seed=1)
        a = wavio.file_sha256(next(iter(self._wavs())))
        shutil.rmtree(self.mat)
        self.gen(seed=2)
        b = wavio.file_sha256(next(iter(self._wavs())))
        self.assertNotEqual(a, b)

    def test_language_filter(self):
        generate(self.mat, languages=("en",), clip_count=2)
        csvs = os.listdir(os.path.join(self.mat, "events"))
        self.assertTrue(csvs)
        self.assertTrue(all("_en_" in n for n in csvs))

    def _wavs(self):
        d = os.path.join(self.mat, "wav")
        return [os.path.join(d, n) for n in sorted(os.listdir(d))]


class WavValidationTests(TempCase):
    def _good(self, path, rate=48000, samples=None):
        samples = samples if samples is not None else [1, -2, 300, -300]
        wavio.write_wav(path, rate, samples)
        return samples

    def test_valid_roundtrip(self):
        p = os.path.join(self.base, "a.wav")
        samples = self._good(p)
        w = wavio.read_wav(p)
        self.assertEqual(w.sample_rate, 48000)
        self.assertEqual(w.samples, samples)

    def test_bad_signature(self):
        p = os.path.join(self.base, "b.wav")
        self._good(p)
        with open(p, "rb") as f:
            data = bytearray(f.read())
        data[0:4] = b"RIFX"
        with open(p, "wb") as f:
            f.write(data)
        with self.assertRaises(WavFormatError) as ctx:
            wavio.read_wav(p)
        self.assertEqual(ctx.exception.code, "E_WAV_FORMAT")

    def test_non_pcm_rejected(self):
        p = os.path.join(self.base, "c.wav")
        # Hand-craft a float-format (tag=3) fmt chunk.
        body = struct.pack("<HHIIHH", 3, 1, 48000, 96000, 2, 16)
        payload = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + body + b"data" + struct.pack("<I", 0)
        with open(p, "wb") as f:
            f.write(payload)
        with self.assertRaises(WavFormatError):
            wavio.read_wav(p)

    def test_illegal_riff_length(self):
        p = os.path.join(self.base, "d.wav")
        self._good(p)
        with open(p, "rb") as f:
            data = bytearray(f.read())
        struct.pack_into("<I", data, 4, 999999)
        with open(p, "wb") as f:
            f.write(data)
        with self.assertRaises(WavFormatError):
            wavio.read_wav(p)

    def test_odd_data_bytes_rejected(self):
        p = os.path.join(self.base, "e.wav")
        payload = (b"RIFF" + struct.pack("<I", 37) + b"WAVE" + b"fmt " +
                   struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16) +
                   b"data" + struct.pack("<I", 1) + b"\x00")
        with open(p, "wb") as f:
            f.write(payload)
        with self.assertRaises(WavFormatError):
            wavio.read_wav(p)

    def test_zero_length_data_is_legal(self):
        p = os.path.join(self.base, "zero.wav")
        wavio.write_wav(p, 48000, [])
        w = wavio.read_wav(p)
        self.assertEqual(w.samples, [])
        self.assertEqual(w.sample_rate, 48000)

    def test_fact_sample_count_mismatch_rejected(self):
        p = os.path.join(self.base, "fact.wav")
        samples = [10, -10, 20]
        wavio.write_wav(p, 48000, samples)
        # Inject a fact chunk claiming a different sample count.
        with open(p, "rb") as f:
            data = f.read()
        fact = b"fact" + struct.pack("<I", 4) + struct.pack("<I", 99)
        riff_size = struct.unpack("<I", data[4:8])[0] + len(fact)
        injected = data[:4] + struct.pack("<I", riff_size) + data[8:12] + fact + data[12:]
        with open(p, "wb") as f:
            f.write(injected)
        with self.assertRaises(WavFormatError):
            wavio.read_wav(p)

    def test_chunk_payload_beyond_file_rejected(self):
        p = os.path.join(self.base, "beyond.wav")
        wavio.write_wav(p, 48000, [1, 2, 3])
        with open(p, "rb") as f:
            data = bytearray(f.read())
        # Lie that the data chunk is larger than the file.
        data_idx = data.index(b"data")
        struct.pack_into("<I", data, data_idx + 4, 99999)
        struct.pack_into("<I", data, 4, 99999 + 36)
        with open(p, "wb") as f:
            f.write(data)
        with self.assertRaises(WavFormatError):
            wavio.read_wav(p)

    def test_cli_nonzero_exit_and_json_stderr(self):
        p = os.path.join(self.base, "bad.wav")
        with open(p, "wb") as f:
            f.write(b"not a wav at all really")
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with redirect_stderr(buf):
            code = main(["verify-wav", p])
        self.assertNotEqual(code, 0)
        rec = json.loads(buf.getvalue().strip())
        self.assertTrue(rec["code"].startswith("E_WAV"))


class ResampleTests(unittest.TestCase):
    def test_length_within_one_sample(self):
        import math
        for ir, orr in [(44100, 48000), (48000, 44100), (44100, 44100)]:
            for n in (1, 7, 333, 4410, 10000):
                x = [int(8000 * math.sin(2 * math.pi * 300 * i / ir)) for i in range(n)]
                y, meta = resample(x, ir, orr)
                self.assertLessEqual(abs(len(y) - expected_output_len(n, ir, orr)), 1)
                self.assertEqual(meta["length_delta"], len(y) - meta["output_samples"] if ir != orr else 0)

    def test_dc_and_determinism(self):
        import math
        dc = [12000] * 4000
        y, _ = resample(dc, 44100, 48000)
        self.assertLess(max(abs(v - 12000) for v in y[800:-800]), 1e-6)
        x = [int(9000 * math.sin(2 * math.pi * 440 * i / 44100)) for i in range(6000)]
        a, _ = resample(x, 44100, 48000)
        b, _ = resample(x, 44100, 48000)
        self.assertEqual(a, b)


class MixerMathTests(unittest.TestCase):
    def test_ms_rounding_consistent(self):
        self.assertEqual(ms_to_samples(1, 48000), 48)  # 48.0
        self.assertEqual(ms_to_samples(1, 44100), 44)  # 44.1 -> 44
        self.assertEqual(ms_to_samples(10, 44100), 441)

    def test_gain_and_headroom_limit(self):
        self.assertAlmostEqual(db_to_linear(6.0), 1.9952623, places=5)
        vals = [40000.0, -40000.0, 100.0]
        out, clipped, limit = pcm16_from_float(vals, headroom_db=-1.0)
        self.assertEqual(clipped, 2)
        self.assertLess(limit, 32767.0)
        self.assertLessEqual(max(abs(v) for v in out), int(limit) + 1)


class MixerBehaviorTests(TempCase):
    def _clip(self, rows, rate=48000):
        from audiopipe.mixer import DegradationReport, mix_clip
        os.makedirs(os.path.join(self.base, "wav"), exist_ok=True)
        report = DegradationReport()
        clip_ms = max(int(r["start_ms"]) + int(r["duration_ms"]) for r in rows)
        pcm, meta, total, clipped, limit = mix_clip(
            rows, os.path.join(self.base, "wav"), rate, clip_ms, -1.0, report
        )
        return pcm, meta, report

    def _wav(self, name, rate, samples):
        os.makedirs(os.path.join(self.base, "wav"), exist_ok=True)
        wavio.write_wav(os.path.join(self.base, "wav", name), rate, samples)

    def _row(self, **over):
        row = dict(clip_id="c0", language="en", track=1, event_index=1,
                   start_ms=0, duration_ms=100, gain_db=0.0, missing=0,
                   force_mute=0, audio_file="", declared_rate=48000,
                   actual_samples=0, duration_mismatch_pct=0)
        row.update(over)
        return row

    def test_same_track_placeholder_overrides_earlier_audio(self):
        self._wav("a.wav", 48000, [10000] * 4800)
        rows = [
            self._row(event_index=1, audio_file="a.wav", actual_samples=4800),
            self._row(event_index=2, start_ms=0, duration_ms=100, missing=1),
        ]
        pcm, meta, report = self._clip(rows)
        # The later missing event's placeholder silences the whole window.
        self.assertTrue(all(v == 0 for v in pcm[: ms_to_samples(100, 48000)]))
        self.assertTrue(any(r["kind"] == "missing_declared" for r in report.recoverable))

    def test_event_fully_outside_window_is_recoverable(self):
        self._wav("a.wav", 48000, [10000] * 4800)
        rows = [
            self._row(event_index=1, start_ms=0, duration_ms=100,
                      audio_file="a.wav", actual_samples=4800),
            # Entirely before the clip origin (negative offset fully outside).
            self._row(event_index=2, track=1, start_ms=-200, duration_ms=100,
                      audio_file="a.wav", actual_samples=4800),
        ]
        pcm, meta, report = self._clip(rows)
        # The window spans the first 100ms event; the second never enters it.
        self.assertEqual(len(pcm), ms_to_samples(100, 48000))
        self.assertTrue(any(r["kind"] == "fully_outside_window"
                            for r in report.recoverable))

    def test_forced_mute_is_recoverable_and_silent(self):
        self._wav("a.wav", 48000, [10000] * 4800)
        rows = [self._row(force_mute=1, audio_file="a.wav", actual_samples=4800)]
        pcm, meta, report = self._clip(rows)
        self.assertTrue(all(v == 0 for v in pcm))
        self.assertTrue(any(r["kind"] == "forced_mute" for r in report.recoverable))


class PipelineTests(TempCase):
    def test_end_to_end_and_manifest_stability(self):
        self.gen()
        results, code = self.run_mix()
        self.assertEqual(code, 0)
        manifest1 = os.path.join(self.out, "manifest.json")
        self.assertTrue(os.path.exists(manifest1))
        with open(manifest1, "rb") as f:
            m1 = f.read()
        # Re-run with unchanged inputs: identical whole-directory bytes.
        results2, code2 = self.run_mix()
        self.assertEqual(code2, 0)
        self.assertTrue(all(r.get("reused") for r in results2["completed"]))
        with open(manifest1, "rb") as f:
            self.assertEqual(f.read(), m1)
        self._assert_whole_dir_unchanged()

    def _snapshot(self):
        snap = {}
        for root, _d, files in os.walk(self.out):
            for name in files:
                p = os.path.join(root, name)
                snap[os.path.relpath(p, self.out)] = wavio.file_sha256(p)
        return snap

    def _assert_whole_dir_unchanged(self):
        # The second run reused checkpoints; take a fresh third snapshot and
        # confirm outputs are stable even after another full pass.
        snap_before = self._snapshot()
        self.run_mix()
        self.assertEqual(snap_before, self._snapshot())

    def test_degradation_report_categories(self):
        self.gen()
        results, code = self.run_mix()
        report_path = os.path.join(self.out, "reports", "en.degradation.json")
        with open(report_path) as rf:
            report = json.load(rf)
        kinds = {r["kind"] for r in report["recoverable"]}
        # Deterministic generator guarantees some missing and/or mismatched data.
        self.assertTrue(kinds & {"missing_declared", "duration_exceeds_window",
                                 "duration_short", "zero_duration"})
        self.assertEqual(report["fatal"], [])
        # Metadata marks placeholders and final sample spans.
        for meta_file in self._meta_files():
            with open(meta_file) as f:
                data = json.load(f)
            for ev in data["events"]:
                self.assertIn("target_start_sample", ev)
                self.assertIn("target_end_sample", ev)
                self.assertIn("placeholder", ev)

    def test_fatal_clip_isolates_others(self):
        self.gen()
        # Corrupt one WAV structurally -> that clip fails fatally, others finish.
        wav_dir = os.path.join(self.mat, "wav")
        target = sorted(n for n in os.listdir(wav_dir) if n.endswith(".wav"))[0]
        with open(os.path.join(wav_dir, target), "r+b") as f:
            f.seek(20)
            f.write(b"\x03\x00")  # corrupt format tag -> non-PCM
        results, code = self.run_mix()
        self.assertNotEqual(code, 0)
        self.assertTrue(results["fatal"])
        # Other language artifacts must still be produced.
        self.assertTrue(os.path.isdir(os.path.join(self.out, "ja")))
        self.assertTrue(self._meta_files())

    def test_checkpoint_version_rejection(self):
        self.gen()
        self.run_mix()
        legacy = os.path.join(self.out, "_checkpoints", "checkpoints.v1.json")
        with open(legacy, "w") as f:
            json.dump({"version": 1, "clips": {}}, f)
        from audiopipe.errors import CheckpointVersionError
        with self.assertRaises(CheckpointVersionError):
            self.run_mix()

    def test_fingerprint_mismatch_forces_recompute(self):
        self.gen()
        results, _ = self.run_mix()
        token = results["completed"][0]["token"]
        store = CheckpointStore(os.path.join(self.out, "_checkpoints"))
        good_fp = parameter_fingerprint({
            "target_rate": 48000, "headroom_db": -1.0,
            "mixer": "audiopipe.v1", "crossfade_ms": 5, "rounding": "round_half_up"})
        self.assertTrue(store.is_current(token, store._data["clips"][token]["input_hash"], good_fp))
        self.assertFalse(store.is_current(token, "deadbeef", good_fp))
        self.assertFalse(store.is_current(token, store._data["clips"][token]["input_hash"], "wrongfp"))

    def run_mix(self):
        from audiopipe.pipeline import run_pipeline
        return run_pipeline(self.mat, self.out, target_rate=48000,
                            headroom_db=-1.0, max_workers=2, max_open_files=4)

    def _meta_files(self):
        out = []
        for root, _d, files in os.walk(self.out):
            out += [os.path.join(root, n) for n in files if n.endswith(".metadata.json")]
        return out


class InterruptTests(TempCase):
    def test_interrupt_checkpoint_recovery(self):
        self.gen()
        env = dict(os.environ)
        env["AUDIOPIPE_SLEEP_STEPS"] = "1"
        # Launch a subprocess that processes clips slowly and gets killed.
        cmd = [sys.executable, "-m", "audiopipe", "mix", self.mat,
               os.path.join(self.base, "int_out"), "--workers", "1"]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2.0)
        proc.kill()
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()
        ckpt = os.path.join(self.base, "int_out", "_checkpoints",
                            "checkpoints.v%d.json" % CHECKPOINT_VERSION)
        self.assertTrue(os.path.exists(ckpt))
        with open(ckpt) as cf:
            saved = json.load(cf)
        done_before = len(saved["clips"])
        self.assertGreaterEqual(done_before, 1)
        total_tasks = len([
            n for n in os.listdir(os.path.join(self.mat, "events"))
            if n.endswith("_events.csv")
        ])
        # Restart: it must reuse matching fingerprints and finish all clips.
        env2 = dict(os.environ)
        env2.pop("AUDIOPIPE_SLEEP_STEPS", None)
        proc2 = subprocess.Popen(cmd, cwd=ROOT, env=env2,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc2.communicate(timeout=60)
        self.assertEqual(proc2.returncode, 0)
        with open(ckpt) as cf:
            saved2 = json.load(cf)
        self.assertEqual(len(saved2["clips"]), total_tasks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
