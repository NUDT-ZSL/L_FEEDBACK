from __future__ import annotations

import json
import hashlib
import math
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np

from gamepost.config import ExportConfig
from gamepost.dsp import (
    apply_silence_gate,
    detect_silence,
    repair_clipping,
    resample_narration,
)
from gamepost.errors import CancelledError, EventValidationError
from gamepost.events import EventSpec, parse_events
from gamepost.pipeline import ExportRequest, run_export
from gamepost.resources import CancellationToken
from gamepost.samples import generate_sample_set
from gamepost.timeline import build_timeline
from gamepost.wav import read_wav, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class PipelineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(
            prefix="gamepost-tests-", dir=ROOT
        )
        cls.root = Path(cls.temp.name)
        cls.samples = generate_sample_set(cls.root / "samples")
        cls.config = ExportConfig(chunk_frames=4_800)
        cls.game = read_wav(cls.samples["game"])
        cls.mic = read_wav(cls.samples["mic"])
        cls.events = parse_events(
            cls.samples["events"], cls.mic.frames, cls.config
        )
        cls.timeline = build_timeline(cls.mic.frames, cls.events, cls.config)
        cls.reference_request = ExportRequest(
            game=cls.samples["game"],
            mic=cls.samples["mic"],
            events=cls.samples["events"],
            output=str(cls.root / "reference" / "mix.wav"),
            manifest=str(cls.root / "reference" / "manifest.json"),
            config=cls.config,
        )
        cls.reference_manifest = run_export(cls.reference_request)
        cls.reference_hash = sha256_file(cls.reference_request.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_event_validation_and_timeline_are_sample_exact(self) -> None:
        self.assertEqual(self.timeline.output_frames, 566_400)
        self.assertEqual(self.timeline.events[0].start_frame, 72_000)
        timeline_json = self.timeline.to_dict()
        self.assertEqual(
            [event["output_start_frame"] for event in timeline_json["events"]],
            [72_000, 182_400, 376_800],
        )
        self.assertEqual(timeline_json["segments"][0]["source_start"], 0)
        self.assertEqual(
            sum(segment.output_frames for segment in self.timeline.segments),
            self.timeline.output_frames,
        )
        source_cursor = 0
        output_cursor = 0
        for segment in self.timeline.segments:
            self.assertEqual(segment.source_start, source_cursor)
            self.assertEqual(segment.output_start, output_cursor)
            source_cursor = segment.source_end
            output_cursor = segment.output_end
        self.assertEqual(source_cursor, self.mic.frames)
        self.assertEqual(output_cursor, self.timeline.output_frames)

        chinese_events = self.root / "chinese_events.json"
        chinese_events.write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "timestamp_ms": 1000,
                            "duration": 100,
                            "type": "击杀",
                            "speed": 1.0,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        parsed_chinese = parse_events(
            chinese_events, self.mic.frames, self.config
        )
        self.assertEqual(parsed_chinese[0].category, "击杀")

        expected = (
            ("overlap", "EVENT_OVERLAP"),
            ("out_of_range", "EVENT_OUT_OF_RANGE"),
            ("unknown_category", "EVENT_UNKNOWN_CATEGORY"),
        )
        for key, code in expected:
            with self.subTest(category=key):
                with self.assertRaises(EventValidationError) as raised:
                    parse_events(
                        self.samples["invalid_events"][key],
                        self.mic.frames,
                        self.config,
                    )
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.exit_code, 2)

        report = self.root / "invalid_report.json"
        completed = subprocess.run(
            [
                PYTHON,
                "-m",
                "gamepost",
                "validate-events",
                "--events",
                self.samples["invalid_events"]["overlap"],
                "--mic",
                self.samples["mic"],
                "--report",
                str(report),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "invalid_event_table")
        self.assertEqual(payload["primary_error"]["code"], "EVENT_OVERLAP")
        self.assertIn("EVENT_OVERLAP", payload["business_codes"])

    def test_clipping_repair_only_changes_full_scale_runs(self) -> None:
        result = repair_clipping(self.mic.pcm, self.config)
        self.assertEqual(result.stats["run_count"], 3)
        self.assertEqual(result.stats["samples_repaired"], 108)
        self.assertEqual(result.stats["original_peak_int16"], 32_767)
        self.assertLess(result.stats["repaired_peak_int16"], 32_767)
        self.assertTrue(result.stats["non_clipped_samples_unchanged"])
        clipped = np.abs(self.mic.pcm.astype(np.int32)) >= self.config.clip_threshold
        self.assertTrue(np.array_equal(result.pcm[~clipped], self.mic.pcm[~clipped]))
        self.assertFalse(np.array_equal(result.pcm[clipped], self.mic.pcm[clipped]))

    def test_silence_gate_logs_ranges_and_has_smooth_edges(self) -> None:
        repaired = repair_clipping(self.mic.pcm, self.config).pcm
        mic_float = repaired.astype(np.float64) / 32768.0
        segments = detect_silence(mic_float, self.config.sample_rate, self.config)
        self.assertGreaterEqual(len(segments), 6)
        for segment in segments:
            self.assertIn("start_sample", segment)
            self.assertIn("end_sample", segment)
            self.assertEqual(segment["gain_linear"], 0.0)
            self.assertGreater(segment["samples"], 0)

        gated = apply_silence_gate(mic_float, segments)
        self.assertEqual(gated.shape, mic_float.shape)
        self.assertLessEqual(float(np.max(np.abs(gated))), 1.0)
        for segment in segments:
            center = segment["start_sample"] + segment["samples"] // 2
            self.assertEqual(float(gated[center, 0]), 0.0)
            start = segment["start_sample"]
            end = segment["end_sample"]
            fade = segment["fade_samples"]
            if start > 0:
                window = gated[start - 1 : start + fade + 2, 0]
                self.assertLess(float(np.max(np.abs(np.diff(window)))), 0.01)
            if end < gated.shape[0]:
                window = gated[end - fade - 1 : end + 2, 0]
                self.assertLess(float(np.max(np.abs(np.diff(window)))), 0.01)

    def test_bandlimited_varispeed_preserves_outside_and_rejects_alias(self) -> None:
        repaired = repair_clipping(self.mic.pcm, self.config).pcm
        source = repaired.astype(np.float64) / 32768.0
        segments = detect_silence(source, self.config.sample_rate, self.config)
        source = apply_silence_gate(source, segments)
        output = resample_narration(source, self.timeline, self.config)
        self.assertEqual(output.shape, (self.timeline.output_frames, 1))
        self.assertTrue(np.all(np.isfinite(output)))
        for segment in self.timeline.segments:
            if segment.kind == "normal":
                self.assertTrue(
                    np.array_equal(
                        output[segment.output_start : segment.output_end],
                        source[segment.source_start : segment.source_end],
                    )
                )

        sample_rate = 48_000
        time_axis = np.arange(sample_rate, dtype=np.float64) / sample_rate
        alias_source = np.sin(
            2.0 * math.pi * 23_000.0 * time_axis
        )[:, None]
        event = EventSpec(
            index=0,
            start_ms=Decimal(100),
            duration_ms=Decimal(1000),
            category="kill",
            rate=Decimal(2),
            start_frame=4_800,
            end_frame=48_000,
        )
        high_rate_timeline = build_timeline(sample_rate, [event], self.config)
        alias_output = resample_narration(
            alias_source, high_rate_timeline, self.config
        )
        event_segment = high_rate_timeline.segments[1]
        event_output = alias_output[
            event_segment.output_start : event_segment.output_end
        ]
        source_rms = float(np.sqrt(np.mean(alias_source[4_800:] ** 2)))
        output_rms = float(np.sqrt(np.mean(event_output**2)))
        self.assertLess(output_rms / source_rms, 0.01)

    def test_export_is_byte_deterministic_and_manifest_is_detailed(self) -> None:
        second = ExportRequest(
            game=self.samples["game"],
            mic=self.samples["mic"],
            events=self.samples["events"],
            output=str(self.root / "second" / "mix.wav"),
            manifest=str(self.root / "second" / "manifest.json"),
            config=self.config,
        )
        manifest = run_export(second)
        self.assertEqual(sha256_file(second.output), self.reference_hash)
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["output"]["sample_count_matches_timeline"])
        self.assertEqual(manifest["output"]["frames"], self.timeline.output_frames)
        self.assertEqual(
            manifest["parameter_fingerprint"],
            self.reference_manifest["parameter_fingerprint"],
        )
        self.assertIn("sha256", manifest["inputs"]["game"])
        self.assertIn("sha256", manifest["inputs"]["microphone"])
        self.assertIn("sha256", manifest["inputs"]["event_table"])
        self.assertEqual(
            manifest["stages"]["clipping"]["samples_repaired"], 108
        )
        self.assertGreaterEqual(
            manifest["stages"]["silence_gate"]["segment_count"], 6
        )
        self.assertIsNotNone(
            manifest["stages"]["final_measurement"]["integrated_lufs"]
        )
        self.assertIsNotNone(manifest["stages"]["final_measurement"]["peak_dbfs"])
        self.assertIn("EVENT_OVERLAP", manifest["business_codes"])
        self.assertEqual(
            len(manifest["chunk_completion_table"]),
            math.ceil(self.timeline.output_frames / self.config.chunk_frames),
        )
        output_bytes = Path(second.output).read_bytes()
        for chunk in manifest["chunk_completion_table"]:
            start = 44 + chunk["start_frame"] * manifest["output"]["channels"] * 2
            end = start + chunk["frames"] * manifest["output"]["channels"] * 2
            self.assertEqual(
                hashlib.sha256(output_bytes[start:end]).hexdigest(),
                chunk["sha256"],
            )

        completed_rerun = run_export(self.reference_request)
        self.assertEqual(
            completed_rerun["diagnostics"]["resume_decision"],
            "resume_fingerprint_match",
        )
        self.assertEqual(sha256_file(self.reference_request.output), self.reference_hash)

    def test_forced_kill_then_resume_is_byte_identical(self) -> None:
        output = self.root / "forced-kill" / "mix.wav"
        manifest_path = self.root / "forced-kill" / "manifest.json"
        state_path = Path(str(output) + ".state.json")
        command = [
            PYTHON,
            "-m",
            "gamepost",
            "export",
            "--game",
            self.samples["game"],
            "--mic",
            self.samples["mic"],
            "--events",
            self.samples["events"],
            "--chunk-frames",
            "1200",
            "--out",
            str(output),
            "--manifest",
            str(manifest_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + 30.0
        killed_at = 0
        while time.time() < deadline:
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.002)
                    continue
                killed_at = len(state.get("chunks", []))
                if killed_at >= 8:
                    process.kill()
                    break
            if process.poll() is not None:
                break
            time.sleep(0.002)
        process.communicate(timeout=10)
        self.assertGreaterEqual(killed_at, 8)
        self.assertNotEqual(process.returncode, 0)
        state_after_kill = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state_after_kill["status"], "in_progress")
        self.assertGreaterEqual(len(state_after_kill["chunks"]), 8)
        self.assertFalse(output.exists())
        self.assertTrue(Path(str(output) + ".part.wav").exists())

        resumed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(sha256_file(output), self.reference_hash)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["diagnostics"]["resume_decision"],
            "resume_fingerprint_match",
        )
        self.assertEqual(len(manifest["chunks"]), math.ceil(566_400 / 1200))
        indices = [chunk["index"] for chunk in manifest["chunks"]]
        self.assertEqual(indices, list(range(len(indices))))

    def test_cancellation_occurs_at_chunk_boundary_and_resumes(self) -> None:
        request = ExportRequest(
            game=self.samples["game"],
            mic=self.samples["mic"],
            events=self.samples["events"],
            output=str(self.root / "cancel" / "mix.wav"),
            manifest=str(self.root / "cancel" / "manifest.json"),
            config=self.config,
        )
        token = CancellationToken()

        def progress(payload: dict) -> None:
            if (
                payload["status"] == "in_progress"
                and payload["completed_chunks"] >= 3
            ):
                token.cancel()

        with self.assertRaises(CancelledError):
            run_export(request, token=token, progress=progress)
        state_path = Path(str(request.output) + ".state.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "in_progress")
        self.assertGreaterEqual(len(state["chunks"]), 3)
        self.assertFalse(Path(request.output).exists())
        for index, chunk in enumerate(state["chunks"]):
            self.assertEqual(chunk["index"], index)

        manifest = run_export(request)
        self.assertEqual(sha256_file(request.output), self.reference_hash)
        self.assertEqual(
            manifest["diagnostics"]["resume_decision"],
            "resume_fingerprint_match",
        )

    def test_fingerprint_mismatch_discards_old_chunks(self) -> None:
        request = ExportRequest(
            game=self.samples["game"],
            mic=self.samples["mic"],
            events=self.samples["events"],
            output=str(self.root / "fingerprint" / "mix.wav"),
            manifest=str(self.root / "fingerprint" / "manifest.json"),
            config=self.config,
        )
        first = run_export(request)
        first_state = json.loads(
            Path(str(request.output) + ".state.json").read_text(encoding="utf-8")
        )
        changed = replace(
            request,
            config=replace(self.config, silence_threshold_dbfs=-40.0),
        )
        second = run_export(changed)
        second_state = json.loads(
            Path(str(request.output) + ".state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            second["diagnostics"]["resume_decision"],
            "reset_fingerprint_mismatch",
        )
        self.assertNotEqual(
            first["parameter_fingerprint"], second["parameter_fingerprint"]
        )
        self.assertNotEqual(
            first_state["parameter_fingerprint"],
            second_state["parameter_fingerprint"],
        )
        self.assertEqual(second_state["chunks"], second["chunks"])
        self.assertEqual(sha256_file(request.output), second["output"]["sha256"])

    def test_batch_queues_under_shared_memory_quota(self) -> None:
        batch_root = self.root / "batch"
        jobs = {
            "jobs": [
                {
                    "id": "a",
                    "game": self.samples["game"],
                    "mic": self.samples["mic"],
                    "events": self.samples["events"],
                    "output": str(batch_root / "a.wav"),
                    "config": {"chunk_frames": 4_800},
                },
                {
                    "id": "b",
                    "game": self.samples["game"],
                    "mic": self.samples["mic"],
                    "events": self.samples["events"],
                    "output": str(batch_root / "b.wav"),
                    "config": {"chunk_frames": 4_800},
                },
            ]
        }
        jobs_path = batch_root / "jobs.json"
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
        report_path = batch_root / "report.json"
        completed = subprocess.run(
            [
                PYTHON,
                "-m",
                "gamepost",
                "batch",
                "--jobs",
                str(jobs_path),
                "--report",
                str(report_path),
                "--workers",
                "2",
                "--memory-mb",
                "110",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["workers"], 2)
        self.assertTrue(
            all(job["status"] == "complete" for job in report["jobs"])
        )
        self.assertEqual(sha256_file(batch_root / "a.wav"), self.reference_hash)
        self.assertEqual(sha256_file(batch_root / "b.wav"), self.reference_hash)
        leases = [job["memory_lease_bytes"] for job in report["jobs"]]
        self.assertTrue(all(value < report["memory_quota_bytes"] for value in leases))


if __name__ == "__main__":
    unittest.main()
