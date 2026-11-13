from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from replaymix.checkpoint import load_checkpoint_file
from replaymix.gain import adaptive_gain, dbfs_to_linear
from replaymix.generator import generate_scenario
from replaymix.pipeline import MixConfig, _bounded_map, process_session
from replaymix.resample import (
    fit_to_target,
    natural_output_frames,
    resample_linear_to_48k,
)
from replaymix.timeline import ms_to_frames
from replaymix.wavio import read_wav_mono_float64, sha256_file, write_wav_pcm16


ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "artifacts" / "test-tmp"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ReplayMixTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="case-", dir=TEST_TMP))

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def _generate(self, name: str = "samples", faults: bool = False) -> Path:
        out = self.work / name
        generate_scenario(out, include_faults=faults)
        return out

    def test_generator_is_byte_deterministic(self) -> None:
        first = self._generate("first")
        second = self._generate("second")
        self.assertEqual(
            (first / "event_table.json").read_bytes(),
            (second / "event_table.json").read_bytes(),
        )
        for path in sorted(first.glob("*.wav")):
            self.assertEqual(
                path.read_bytes(),
                (second / path.name).read_bytes(),
                path.name,
            )

    def test_resampler_is_deterministic_and_rational(self) -> None:
        source = np.arange(800, dtype=np.float64) / 800.0
        first = resample_linear_to_48k(source, 8_000)
        second = resample_linear_to_48k(source, 8_000)
        self.assertEqual(first.size, 4_800)
        self.assertEqual(natural_output_frames(800, 8_000), 4_800)
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(first[0], source[0])
        self.assertAlmostEqual(first[6], source[1])

        natural = np.arange(4_799, dtype=np.float64)
        padded, fit = fit_to_target(natural, 4_800)
        self.assertEqual(padded.size, 4_800)
        self.assertEqual(padded[-1], 0.0)
        self.assertEqual(fit.zero_padded_frames, 1)
        self.assertEqual(fit.frame_delta, -1)

    def test_gain_caps_and_never_attenuates(self) -> None:
        tiny = np.full(100, 1.0e-6, dtype=np.float64)
        _, tiny_gain = adaptive_gain(tiny, target_peak_dbfs=-1.0, max_gain_db=20.0)
        self.assertAlmostEqual(tiny_gain.gain_db, 20.0)
        self.assertAlmostEqual(tiny_gain.gain_linear, 10.0)

        quiet = np.full(100, 0.1, dtype=np.float64)
        gained, quiet_gain = adaptive_gain(
            quiet, target_peak_dbfs=-1.0, max_gain_db=20.0
        )
        self.assertGreaterEqual(quiet_gain.gain_db, 0.0)
        self.assertLessEqual(quiet_gain.gain_db, 20.0)
        self.assertLessEqual(
            float(np.max(np.abs(gained))), dbfs_to_linear(-1.0) + 1e-15
        )

        loud = np.full(100, 0.99, dtype=np.float64)
        gained_loud, loud_gain = adaptive_gain(
            loud, target_peak_dbfs=-1.0, max_gain_db=20.0
        )
        self.assertEqual(loud_gain.gain_db, 0.0)
        self.assertTrue(np.array_equal(gained_loud, loud))
        self.assertTrue(loud_gain.input_peak_above_target)

    def test_clean_mix_report_and_frame_count(self) -> None:
        samples = self._generate()
        out = self.work / "mix"
        result = process_session(
            samples / "event_table.json", out, config=MixConfig(concurrency=2)
        )
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        session_frames = ms_to_frames(12_000)
        self.assertEqual(session_frames, 576_000)
        self.assertEqual(report["mix"]["total_frames"], session_frames)
        self.assertEqual(report["session"]["mix_frames"], session_frames)
        self.assertEqual(report["output_wav"]["frames"], session_frames)
        self.assertEqual(report["output_wav"]["sample_rate_hz"], 48_000)
        self.assertEqual(report["output_wav"]["channels"], 1)
        self.assertEqual([track["track_id"] for track in report["tracks"]], [1, 2, 3, 4])
        self.assertEqual(report["errors"], [])
        self.assertLessEqual(report["settings"]["max_observed_concurrency"], 2)
        self.assertGreaterEqual(report["settings"]["max_observed_concurrency"], 1)

        for track in report["tracks"]:
            self.assertIn(track["status"], ("processed", "processed_with_errors"))
            self.assertGreater(track["zero_fill"]["gap_fill_frames"], 0)
            for segment in track["segments"]:
                if segment["kind"] != "active":
                    continue
                self.assertGreaterEqual(segment["gain_db"], 0.0)
                self.assertLessEqual(segment["gain_db"], 20.0)
                if not segment["input_peak_above_target"]:
                    self.assertLessEqual(
                        segment["output_peak_linear"],
                        dbfs_to_linear(-1.0) + 1e-12,
                    )

        samples_out, info = read_wav_mono_float64(result.wav_path)
        self.assertEqual(info.frames, session_frames)
        self.assertLessEqual(float(np.max(np.abs(samples_out))), 1.0)

        # A clean rerun in a fresh directory must produce byte-identical audio.
        second = process_session(
            samples / "event_table.json",
            self.work / "mix-second",
            config=MixConfig(concurrency=2),
        )
        self.assertEqual(sha256_file(result.wav_path), sha256_file(second.wav_path))

    def test_gap_and_silence_are_explicit_zero_segments(self) -> None:
        samples_dir = self.work / "zero-case"
        samples_dir.mkdir(parents=True)
        rate = 48_000
        frames = 48_000
        t = np.arange(frames, dtype=np.float64) / rate
        tone = 0.25 * np.sin(2.0 * np.pi * 220.0 * t)
        write_wav_pcm16(samples_dir / "track.wav", tone, sample_rate_hz=rate)
        table = {
            "session_id": "zero-case",
            "session_ms": 1_000,
            "tracks": [
                {
                    "track_id": 1,
                    "file": "track.wav",
                    "sample_rate_hz": rate,
                    "channels": 1,
                    "frames": frames,
                    "write_start_ms": 0,
                    "write_end_ms": 1_000,
                    "silence_ms": [[600, 700]],
                    "gaps_ms": [[200, 400]],
                }
            ],
            "events": [],
        }
        _write_json(samples_dir / "event_table.json", table)
        result = process_session(
            samples_dir / "event_table.json",
            self.work / "zero-out",
            config=MixConfig(checkpoint_enabled=False),
        )
        audio, _ = read_wav_mono_float64(result.wav_path)
        gap_start, gap_end = ms_to_frames(200), ms_to_frames(400)
        silence_start, silence_end = ms_to_frames(600), ms_to_frames(700)
        self.assertTrue(np.all(audio[gap_start:gap_end] == 0.0))
        self.assertTrue(np.all(audio[silence_start:silence_end] == 0.0))
        self.assertGreater(float(np.max(np.abs(audio[:gap_start]))), 0.0)
        track = result.report["tracks"][0]
        self.assertEqual(
            track["zero_fill"]["gap_fill_frames"], gap_end - gap_start
        )
        self.assertEqual(
            track["zero_fill"]["silence_fill_frames"],
            silence_end - silence_start,
        )
        kinds = [segment["kind"] for segment in track["segments"]]
        self.assertIn("gap", kinds)
        self.assertIn("silence", kinds)

    def test_track_input_order_does_not_change_output(self) -> None:
        samples = self._generate()
        table_path = samples / "event_table.json"
        table = json.loads(table_path.read_text(encoding="utf-8"))
        table["tracks"] = list(reversed(table["tracks"]))
        shuffled = samples / "event_table_shuffled.json"
        _write_json(shuffled, table)

        a = process_session(table_path, self.work / "a", config=MixConfig())
        b = process_session(shuffled, self.work / "b", config=MixConfig())
        self.assertEqual(sha256_file(a.wav_path), sha256_file(b.wav_path))
        self.assertEqual(
            [item["track_id"] for item in b.report["tracks"]], [1, 2, 3, 4]
        )

    def test_business_errors_are_isolated_per_track(self) -> None:
        samples = self._generate(faults=True)
        result = process_session(
            samples / "event_table.json",
            self.work / "faults",
            config=MixConfig(concurrency=2),
        )
        report = result.report
        by_id = {track["track_id"]: track for track in report["tracks"]}
        for track_id in (1, 2, 3, 4):
            self.assertEqual(by_id[track_id]["error_codes"], [])
        self.assertIn(
            "E_TRACK_TIMELINE_MISMATCH", by_id[5]["error_codes"]
        )
        self.assertEqual(
            by_id[6]["error_codes"], ["E_SAMPLE_RATE_UNSUPPORTED"]
        )
        self.assertEqual(by_id[7]["error_codes"], ["E_TRACK_EXCEEDS_SESSION"])
        self.assertEqual(by_id[8]["error_codes"], ["E_TRACK_FILE_MISSING"])
        self.assertIn(
            "E_TRACK_TIMELINE_MISMATCH", by_id[9]["error_codes"]
        )
        self.assertEqual(report["mix"]["total_frames"], ms_to_frames(12_000))
        self.assertEqual(report["mix"]["out_of_bounds_frames_discarded"], 0)
        _audio, info = read_wav_mono_float64(result.wav_path)
        self.assertEqual(info.frames, ms_to_frames(12_000))
        codes = {item["code"] for item in report["errors"]}
        self.assertEqual(
            codes,
            {
                "E_TRACK_TIMELINE_MISMATCH",
                "E_SAMPLE_RATE_UNSUPPORTED",
                "E_TRACK_EXCEEDS_SESSION",
                "E_TRACK_FILE_MISSING",
            },
        )

    def test_checkpoint_reuse_version_and_offset_rebuild(self) -> None:
        samples = self._generate()
        out = self.work / "checkpoint"
        first = process_session(samples / "event_table.json", out, config=MixConfig())
        first_hash = sha256_file(first.wav_path)
        self.assertFalse(first.report["checkpoint_reused"])

        checkpoint = out / "checkpoint.jsonl"
        data = checkpoint.read_bytes()
        lines = data.splitlines()
        header = json.loads(lines[0])
        footer = json.loads(lines[-1])
        self.assertEqual(header["version"], 1)
        self.assertEqual(set(footer["row_offsets"]), {"1", "2", "3", "4"})
        for offset in footer["row_offsets"].values():
            self.assertGreater(offset, 0)
            self.assertLess(offset, len(data))

        second = process_session(samples / "event_table.json", out, config=MixConfig())
        self.assertTrue(second.report["checkpoint_reused"])
        self.assertEqual(
            second.report["checkpoint"]["reused_track_ids"], [1, 2, 3, 4]
        )
        self.assertTrue(
            all(track["from_checkpoint"] for track in second.report["tracks"])
        )
        self.assertEqual(sha256_file(second.wav_path), first_hash)

        third = process_session(
            samples / "event_table.json",
            out,
            config=MixConfig(checkpoint_version=2),
        )
        self.assertFalse(third.report["checkpoint_reused"])
        self.assertEqual(
            third.report["checkpoint"]["rebuild_reason"], "version_changed"
        )
        self.assertEqual(sha256_file(third.wav_path), first_hash)

        lines = checkpoint.read_bytes().splitlines()
        corrupted_footer = json.loads(lines[-1])
        corrupted_footer["row_offsets"]["2"] += 1
        lines[-1] = json.dumps(
            corrupted_footer, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        checkpoint.write_bytes(b"\n".join(lines) + b"\n")
        fourth = process_session(
            samples / "event_table.json",
            out,
            config=MixConfig(checkpoint_version=2),
        )
        self.assertFalse(fourth.report["checkpoint_reused"])
        self.assertEqual(
            fourth.report["checkpoint"]["rebuild_reason"], "row_offset_mismatch"
        )
        self.assertEqual(sha256_file(fourth.wav_path), first_hash)

    def test_interrupted_run_resumes_from_checkpoint(self) -> None:
        samples = self._generate()
        reference = process_session(
            samples / "event_table.json",
            self.work / "reference",
            config=MixConfig(),
        )
        expected_hash = sha256_file(reference.wav_path)
        out = self.work / "interrupted"
        seen: list[int] = []

        def stop_after_first(result: object) -> None:
            seen.append(result.track_id)  # type: ignore[attr-defined]
            raise KeyboardInterrupt("simulated power loss")

        with self.assertRaises(KeyboardInterrupt):
            process_session(
                samples / "event_table.json",
                out,
                config=MixConfig(),
                on_track_complete=stop_after_first,
            )
        self.assertEqual(len(seen), 1)
        loaded = load_checkpoint_file(
            out / "checkpoint.jsonl",
            expected_version=1,
            run_signature=json.loads(
                (out / "checkpoint.jsonl").read_bytes().splitlines()[0]
            )["run_signature"],
        )
        self.assertTrue(loaded.valid)
        self.assertEqual(len(loaded.rows), 1)

        resumed = process_session(
            samples / "event_table.json",
            out,
            config=MixConfig(),
        )
        self.assertTrue(resumed.report["checkpoint_reused"])
        self.assertEqual(
            len(resumed.report["checkpoint"]["reused_track_ids"]), 1
        )
        self.assertEqual(sha256_file(resumed.wav_path), expected_hash)

    def test_bounded_map_never_exceeds_configured_concurrency(self) -> None:
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(item: int) -> int:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.025)
            with lock:
                active -= 1
            return item

        items = list(range(10))
        results, observed = _bounded_map(items, worker, 3)  # type: ignore[arg-type]
        self.assertEqual(results, items)
        self.assertLessEqual(maximum, 3)
        self.assertLessEqual(observed, 3)
        self.assertGreaterEqual(observed, 2)


if __name__ == "__main__":
    unittest.main()
