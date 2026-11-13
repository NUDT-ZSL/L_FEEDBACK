from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eventmix.batch import BatchOptions, run_batch
from eventmix.demo import _write_plain_pcm_wav, generate_demo
from eventmix.timeline import Anchor, fit_alignment
from eventmix.util import (
    atomic_write_bytes,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
)
from eventmix.wav import (
    ExtensibleWavSink,
    parse_wav,
    read_frames,
    read_wav_metadata,
)


WORK_ROOT = ROOT / ".work" / "tests"
WORK_ROOT.mkdir(parents=True, exist_ok=True)


class EventMixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="case-", dir=str(WORK_ROOT)))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        jobs: int = 1,
        max_open_wav: int = 8,
        max_memory_mb: int = 128,
        target_rate: int | None = None,
        bit_depth: int | None = None,
        on_track_error: str | None = None,
        resume: bool = True,
    ) -> dict:
        options = BatchOptions(
            output_dir=output_dir,
            checkpoint_path=output_dir / ".eventmix-checkpoint.json",
            resume=resume,
            jobs=jobs,
            max_open_wav=max_open_wav,
            max_memory_mb=max_memory_mb,
            target_sample_rate=target_rate,
            bit_depth=bit_depth,
            on_track_error=on_track_error,
        )
        return run_batch(input_path, options)

    @staticmethod
    def _task(report: dict, task_id: str) -> dict:
        return next(task for task in report["tasks"] if task["task_id"] == task_id)

    @staticmethod
    def _read_output(path: Path) -> tuple[object, np.ndarray]:
        info = parse_wav(path)
        _, audio = read_frames(info, 0, info.frames)
        return info, audio

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        atomic_write_bytes(
            path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )

    @staticmethod
    def _onset_delta(
        audio: np.ndarray, channels: list[int], program_seconds: float, sample_rate: int
    ) -> int:
        center = int(round(program_seconds * sample_rate))
        radius = int(round(0.025 * sample_rate))
        start = max(0, center - radius)
        end = min(audio.shape[0], center + radius)
        envelope = np.max(np.abs(audio[start:end][:, channels]), axis=1)
        hits = np.flatnonzero(envelope > 0.15)
        if not len(hits):
            raise AssertionError("no click onset found near expected anchor")
        return start + int(hits[0]) - center

    def test_alignment_fit_reports_quantified_residuals(self) -> None:
        anchors = [
            Anchor(0.5, 0.4),
            Anchor(2.0, 1.901),
            Anchor(5.0, 4.9005),
            Anchor(8.0, 7.899),
        ]
        fit = fit_alignment(anchors, 48000)
        self.assertGreater(fit.drift_factor, 0.999)
        self.assertLess(fit.drift_factor, 1.001)
        self.assertEqual(len(fit.anchor_errors), 4)
        self.assertAlmostEqual(
            fit.max_abs_error_ms,
            max(abs(item.error_ms) for item in fit.anchor_errors),
        )
        self.assertGreater(fit.max_abs_error_ms, 0.0)

    def test_extensible_writer_bit_depths_and_odd_data_padding(self) -> None:
        cases = [(16, 2, 3), (24, 1, 1), (32, 2, 3)]
        for bits, channels, frames in cases:
            path = self.temp_dir / f"writer-{bits}-{channels}-{frames}.wav"
            audio = np.linspace(-0.5, 0.5, frames * channels).reshape(frames, channels)
            sink = ExtensibleWavSink(
                path,
                sample_rate=48000,
                bits=bits,
                channels=channels,
                channel_mask=0x4,
                total_frames=frames,
                cues=[],
                timeline_bytes=b'{"test":true}',
                info_metadata={"title": "writer test", "software": "EventMix"},
            )
            sink.write_float(audio)
            sink.close()
            self.assertEqual(sink.clamp_count, 0)
            info = parse_wav(path)
            self.assertEqual(info.format_tag, 0xFFFE)
            self.assertEqual(info.bits_per_sample, bits)
            self.assertEqual(info.channels, channels)
            self.assertEqual(info.channel_mask, 0x4)
            with open(path, "rb") as handle:
                raw = handle.read()
            declared_riff = struct.unpack_from("<I", raw, 4)[0]
            self.assertEqual(len(raw), declared_riff + 8)

    def test_demo_build_metadata_alignment_and_resume(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        out = self.temp_dir / "out"
        report = self._run(demo["manifest"], out, jobs=2)
        self.assertEqual(report["exit_code"], 0, report)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"]["completed"], 2)

        expected = json.loads(demo["expected"].read_text("utf-8"))
        sidecar_by_task: dict[str, dict] = {}
        for task_id, filename in (
            ("demo-event-a", "final-a"),
            ("demo-event-b", "final-b"),
        ):
            wav_path = out / f"{filename}.wav"
            timeline_path = out / f"{filename}.timeline.json"
            info, audio = self._read_output(wav_path)
            self.assertEqual(info.format_tag, 0xFFFE)
            self.assertEqual(info.audio_format, "pcm")
            self.assertEqual(info.channels, 6)
            self.assertEqual(info.sample_rate, 48000)
            self.assertEqual(info.bits_per_sample, 24)
            self.assertEqual(info.channel_mask, 0x633)
            self.assertEqual(info.frames, 336000)
            self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
            self.assertGreater(float(np.max(np.abs(audio))), 0.2)
            # Mono co-commentary is expanded to both BL and BR.
            self.assertTrue(np.array_equal(audio[:, 2], audio[:, 3]))

            metadata = read_wav_metadata(wav_path)
            sidecar_bytes = timeline_path.read_bytes()
            self.assertEqual(metadata["timeline_bytes"], sidecar_bytes)
            timeline = metadata["timeline_json"]
            sidecar_by_task[task_id] = timeline
            self.assertEqual(timeline["program"]["total_frames"], info.frames)
            self.assertEqual(timeline["program"]["channel_mask"], info.channel_mask)
            self.assertEqual(timeline["integrity"]["algorithm"], "sha256")
            digest_payload = dict(timeline)
            digest_payload.pop("integrity")
            self.assertEqual(
                sha256_bytes(canonical_json_bytes(digest_payload)),
                timeline["integrity"]["digest"],
            )
            self.assertIn(timeline["integrity"]["digest"], metadata["info"]["ICMT"])
            self.assertEqual(len(metadata["cues"]), len(timeline["cues"]))
            for cue in timeline["cues"]:
                wav_cue = next(
                    item
                    for item in metadata["cues"]
                    if item["wav_cue_id"] == cue["wav_cue_id"]
                )
                self.assertEqual(wav_cue["position"], cue["frame"])
                self.assertEqual(wav_cue["sample_offset"], cue["frame"])
                self.assertEqual(
                    metadata["labels"][cue["wav_cue_id"]], cue["name"]
                )
                self.assertAlmostEqual(
                    cue["program_seconds"], cue["frame"] / info.sample_rate
                )

            task = self._task(report, task_id)
            self.assertEqual(task["cues"], timeline["cues"])
            track_alignment = {
                item["track_id"]: item for item in task["alignment"]
            }
            for track in timeline["tracks"]:
                report_alignment = track_alignment[track["id"]]
                self.assertEqual(
                    report_alignment["offset_seconds"],
                    track["alignment"]["offset_seconds"],
                )
                self.assertEqual(
                    report_alignment["drift_factor"],
                    track["alignment"]["drift_factor"],
                )
                self.assertEqual(
                    report_alignment["drift_ppm"], track["alignment"]["drift_ppm"]
                )
                self.assertLess(
                    track["alignment"]["max_abs_anchor_error_ms"], 0.01
                )

            for track_id, channels in (
                ("main", [0, 1]),
                ("sub", [2, 3]),
                ("ambient", [4, 5]),
            ):
                for _, program_seconds in expected["anchor_specs"][track_id]:
                    delta = self._onset_delta(
                        audio, channels, program_seconds, info.sample_rate
                    )
                    self.assertLessEqual(abs(delta), 3)

        first_hash = (out / "final-a.wav").read_bytes()
        resumed = self._run(demo["manifest"], out, jobs=2)
        self.assertEqual(resumed["exit_code"], 0)
        self.assertEqual(resumed["summary"]["skipped"], 2)
        self.assertEqual(resumed["summary"]["completed"], 0)
        self.assertEqual((out / "final-a.wav").read_bytes(), first_hash)

    def test_arbitrary_rate_bit_depth_and_downmix(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        out = self.temp_dir / "resampled"
        report = self._run(
            demo["timeline_a"],
            out,
            target_rate=32000,
            bit_depth=16,
        )
        self.assertEqual(report["exit_code"], 0, report)
        task = report["tasks"][0]
        wav_path = Path(task["outputs"]["wav"])
        timeline_path = Path(task["outputs"]["timeline"])
        info, audio = self._read_output(wav_path)
        timeline = json.loads(timeline_path.read_text("utf-8"))
        self.assertEqual(info.sample_rate, 32000)
        self.assertEqual(info.bits_per_sample, 16)
        self.assertEqual(info.frames, 224000)
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        self.assertEqual([cue["frame"] for cue in timeline["cues"]], [14400, 108800, 195200])
        for cue in timeline["cues"]:
            self.assertAlmostEqual(cue["program_seconds"], cue["frame"] / 32000)
        expected = json.loads(demo["expected"].read_text("utf-8"))
        for track_id, channels in (
            ("main", [0, 1]),
            ("sub", [2, 3]),
            ("ambient", [4, 5]),
        ):
            for _, program_seconds in expected["anchor_specs"][track_id]:
                delta = self._onset_delta(audio, channels, program_seconds, 32000)
                self.assertLessEqual(abs(delta), 4)

        # Explicit stereo-to-mono downmix using equal source weights.
        rate = 48000
        frames = rate
        constant = np.column_stack(
            [
                np.full(frames, 0.25),
                np.full(frames, 0.45),
            ]
        )
        stereo_path = self.temp_dir / "constant_stereo.wav"
        _write_plain_pcm_wav(stereo_path, constant, rate, 24)
        timeline = {
            "schema_version": "eventmix-timeline/v1",
            "event": {"id": "downmix", "title": "Downmix check"},
            "program": {
                "sample_rate": 48000,
                "duration_seconds": 0.5,
                "bit_depth": 24,
            },
            "channel_layout": [
                {
                    "name": "mix.C",
                    "track": "mix",
                    "speaker": "FC",
                    "sources": [
                        {"channel": 0, "gain": 0.5},
                        {"channel": 1, "gain": 0.5},
                    ],
                }
            ],
            "tracks": [
                {
                    "id": "mix",
                    "name": "Mix",
                    "path": str(stereo_path),
                    "anchors": [
                        {"source_seconds": 0.0, "program_seconds": 0.0},
                        {"source_seconds": 0.5, "program_seconds": 0.5},
                    ],
                }
            ],
            "cues": [{"id": "mid", "name": "Mid", "program_seconds": 0.25}],
        }
        timeline_path = self.temp_dir / "downmix.json"
        self._write_json(timeline_path, timeline)
        downmix_out = self.temp_dir / "downmix-out"
        downmix_report = self._run(timeline_path, downmix_out)
        self.assertEqual(downmix_report["exit_code"], 0, downmix_report)
        mix_info, mix_audio = self._read_output(
            Path(downmix_report["tasks"][0]["outputs"]["wav"])
        )
        self.assertEqual(mix_info.channels, 1)
        self.assertAlmostEqual(float(np.mean(mix_audio[12000:18000, 0])), 0.35, places=3)

    def test_checkpoint_incomplete_changed_input_and_stale_artifact(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        out = self.temp_dir / "out"
        first = self._run(demo["manifest"], out, jobs=2)
        self.assertEqual(first["summary"]["completed"], 2)
        checkpoint_path = out / ".eventmix-checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
        checkpoint["jobs"].pop("demo-event-b")
        atomic_write_bytes(checkpoint_path, pretty_json_bytes(checkpoint))
        (out / "final-b.wav").unlink()
        (out / "final-b.timeline.json").unlink()

        resumed = self._run(demo["manifest"], out, jobs=2)
        self.assertEqual(self._task(resumed, "demo-event-a")["status"], "skipped")
        self.assertEqual(self._task(resumed, "demo-event-b")["status"], "completed")

        old_timeline = json.loads((out / "final-a.timeline.json").read_text("utf-8"))
        old_main = next(
            track for track in old_timeline["tracks"] if track["id"] == "main"
        )
        old_offset = old_main["alignment"]["offset_seconds"]
        timeline_a = json.loads(demo["timeline_a"].read_text("utf-8"))
        for anchor in timeline_a["tracks"][0]["anchors"]:
            anchor["program_seconds"] += 0.002
        self._write_json(demo["timeline_a"], timeline_a)

        changed = self._run(demo["manifest"], out, jobs=2)
        self.assertEqual(self._task(changed, "demo-event-a")["status"], "completed")
        self.assertEqual(self._task(changed, "demo-event-b")["status"], "skipped")
        rebuilt_timeline = json.loads(
            (out / "final-a.timeline.json").read_text("utf-8")
        )
        main_track = next(
            track for track in rebuilt_timeline["tracks"] if track["id"] == "main"
        )
        self.assertAlmostEqual(
            main_track["alignment"]["offset_seconds"], old_offset + 0.002, places=6
        )

        timeline_path = out / "final-a.timeline.json"
        stale_bytes = timeline_path.read_bytes()
        timeline_path.write_bytes(stale_bytes + b" \n")
        stale = self._run(demo["manifest"], out, jobs=2)
        self.assertEqual(self._task(stale, "demo-event-a")["status"], "completed")
        self.assertEqual(self._task(stale, "demo-event-b")["status"], "skipped")
        self.assertEqual(timeline_path.read_bytes(), stale_bytes)

    def test_short_corrupt_and_silent_tracks_degrade_locally(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        base_out = self.temp_dir / "baseline"
        baseline_report = self._run(demo["timeline_a"], base_out)
        self.assertEqual(baseline_report["exit_code"], 0, baseline_report)
        base_info, base_audio = self._read_output(Path(baseline_report["tasks"][0]["outputs"]["wav"]))

        original_main = parse_wav(demo["audio_dir"] / "main_44100_16.wav")
        original_sub = parse_wav(demo["audio_dir"] / "sub_48000_24.wav")
        silent_main = self.temp_dir / "silent_main.wav"
        _write_plain_pcm_wav(
            silent_main,
            np.zeros((original_main.frames, original_main.channels)),
            original_main.sample_rate,
            original_main.bits_per_sample,
        )
        short_rate = original_sub.sample_rate
        short_frames = int(round(0.70 * short_rate))
        short_time = np.arange(short_frames) / short_rate
        short_sub = self.temp_dir / "short_sub.wav"
        _write_plain_pcm_wav(
            short_sub,
            0.20 * np.sin(2.0 * np.pi * 500.0 * short_time),
            short_rate,
            original_sub.bits_per_sample,
        )
        corrupt_ambient = self.temp_dir / "corrupt_ambient.wav"
        corrupt_ambient.write_bytes(b"RIFF\x04\x00\x00\x00WAVEbad")

        degraded_timeline = json.loads(demo["timeline_a"].read_text("utf-8"))
        degraded_timeline["event"]["id"] = "degraded-short-corrupt"
        for track in degraded_timeline["tracks"]:
            track["path"] = str(
                (demo["timeline_a"].parent / track["path"]).resolve()
            )
            if track["id"] == "sub":
                track["path"] = str(short_sub)
            elif track["id"] == "ambient":
                track["path"] = str(corrupt_ambient)
        degraded_timeline["policy"]["on_track_error"] = "silence"
        degraded_path = self.temp_dir / "degraded.json"
        self._write_json(degraded_path, degraded_timeline)
        degraded_out = self.temp_dir / "degraded-out"
        degraded_report = self._run(degraded_path, degraded_out)
        self.assertEqual(degraded_report["exit_code"], 1, degraded_report)
        self.assertEqual(degraded_report["status"], "partial")
        degraded_task = degraded_report["tasks"][0]
        self.assertTrue(degraded_task["degraded"])
        codes = {(item["track_id"], item["code"]) for item in degraded_task["warnings"]}
        self.assertIn(("sub", "track_short_end"), codes)
        self.assertIn(("ambient", "wav_header_corrupt"), codes)
        for warning in degraded_task["warnings"]:
            self.assertTrue(warning["track_id"])
            self.assertLessEqual(
                warning["range"]["start_seconds"], warning["range"]["end_seconds"]
            )
        degraded_info, degraded_audio = self._read_output(
            Path(degraded_task["outputs"]["wav"])
        )
        self.assertEqual(degraded_info.frames, base_info.frames)
        # A damaged route does not perturb the other independently rendered routes.
        self.assertTrue(np.array_equal(degraded_audio[:, 0:2], base_audio[:, 0:2]))
        self.assertLess(
            float(np.sqrt(np.mean(degraded_audio[48000:, 2:4] ** 2))), 1.0e-7
        )
        self.assertLess(
            float(np.sqrt(np.mean(degraded_audio[:, 4:6] ** 2))), 1.0e-7
        )

        silent_timeline = json.loads(demo["timeline_a"].read_text("utf-8"))
        silent_timeline["event"]["id"] = "degraded-silent"
        for track in silent_timeline["tracks"]:
            track["path"] = str(
                (demo["timeline_a"].parent / track["path"]).resolve()
            )
            if track["id"] == "main":
                track["path"] = str(silent_main)
        silent_path = self.temp_dir / "silent.json"
        self._write_json(silent_path, silent_timeline)
        silent_report = self._run(silent_path, self.temp_dir / "silent-out")
        self.assertEqual(silent_report["exit_code"], 1)
        silent_task = silent_report["tasks"][0]
        silent_codes = {
            (item["track_id"], item["code"]) for item in silent_task["warnings"]
        }
        self.assertIn(("main", "track_silent"), silent_codes)
        _, silent_audio = self._read_output(Path(silent_task["outputs"]["wav"]))
        self.assertLess(float(np.max(np.abs(silent_audio[:, 0:2]))), 1.0e-7)
        self.assertTrue(np.array_equal(silent_audio[:, 2:4], base_audio[:, 2:4]))
        self.assertTrue(np.array_equal(silent_audio[:, 4:6], base_audio[:, 4:6]))

        fail_timeline = json.loads(degraded_path.read_text("utf-8"))
        fail_timeline["event"]["id"] = "degraded-fail-policy"
        fail_timeline["policy"]["on_track_error"] = "fail"
        fail_path = self.temp_dir / "fail.json"
        self._write_json(fail_path, fail_timeline)
        fail_out = self.temp_dir / "fail-out"
        fail_report = self._run(fail_path, fail_out, on_track_error="fail")
        self.assertEqual(fail_report["exit_code"], 2)
        self.assertEqual(fail_report["tasks"][0]["status"], "failed")
        self.assertEqual(fail_report["errors"][0]["code"], "track_input_error")
        self.assertFalse((fail_out / "degraded-fail-policy.wav").exists())

    def test_concurrency_limits_are_byte_deterministic_and_collisions_fail(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        out1 = self.temp_dir / "serial"
        out2 = self.temp_dir / "parallel"
        serial = self._run(demo["manifest"], out1, jobs=1, max_memory_mb=128)
        parallel = self._run(
            demo["manifest"],
            out2,
            jobs=4,
            max_open_wav=1,
            max_memory_mb=16,
        )
        self.assertEqual(serial["exit_code"], 0, serial)
        self.assertEqual(parallel["exit_code"], 0, parallel)
        for filename in (
            "final-a.wav",
            "final-a.timeline.json",
            "final-b.wav",
            "final-b.timeline.json",
        ):
            self.assertEqual(
                (out1 / filename).read_bytes(),
                (out2 / filename).read_bytes(),
                filename,
            )

        manifest = json.loads(demo["manifest"].read_text("utf-8"))
        manifest["jobs"][1]["wav"] = "final-a.wav"
        collision_path = self.temp_dir / "collision.json"
        self._write_json(collision_path, manifest)
        with self.assertRaises(Exception) as context:
            self._run(collision_path, self.temp_dir / "collision-out")
        self.assertIn("collision", str(context.exception).lower())

    def test_cli_exit_codes_and_machine_readable_stdout(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        out = self.temp_dir / "cli-out"
        command = [
            sys.executable,
            "-m",
            "eventmix",
            "build",
            str(demo["timeline_a"]),
            "--output-dir",
            str(out),
            "--no-progress",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, env=env, cwd=str(ROOT)
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema_version"], "eventmix-report/v1")
        self.assertEqual(report["exit_code"], 0)

        bad = json.loads(demo["timeline_a"].read_text("utf-8"))
        bad["tracks"][0]["path"] = str(self.temp_dir / "missing.wav")
        bad["event"]["id"] = "missing-track"
        bad_path = self.temp_dir / "missing.json"
        self._write_json(bad_path, bad)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "eventmix",
                "build",
                str(bad_path),
                "--output-dir",
                str(self.temp_dir / "bad-out"),
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "partial")

        collision = json.loads(demo["manifest"].read_text("utf-8"))
        collision["jobs"][1]["wav"] = collision["jobs"][0]["wav"]
        collision_path = self.temp_dir / "cli-collision.json"
        self._write_json(collision_path, collision)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "eventmix",
                "build",
                str(collision_path),
                "--output-dir",
                str(self.temp_dir / "cli-collision-out"),
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        collision_report = json.loads(completed.stdout)
        self.assertEqual(collision_report["status"], "fatal")
        self.assertEqual(collision_report["errors"][0]["code"], "output_collision")

    def test_two_processes_can_share_one_output_directory(self) -> None:
        demo = generate_demo(self.temp_dir / "demo")
        out = self.temp_dir / "shared"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        command = [
            sys.executable,
            "-m",
            "eventmix",
            "build",
            str(demo["manifest"]),
            "--output-dir",
            str(out),
            "--jobs",
            "2",
            "--no-progress",
        ]
        first = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        second = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        stdout1, stderr1 = first.communicate(timeout=120)
        stdout2, stderr2 = second.communicate(timeout=120)
        self.assertEqual(first.returncode, 0, stderr1)
        self.assertEqual(second.returncode, 0, stderr2)
        self.assertEqual(json.loads(stdout1)["exit_code"], 0)
        self.assertEqual(json.loads(stdout2)["exit_code"], 0)
        sequential_out = self.temp_dir / "sequential"
        sequential = self._run(demo["manifest"], sequential_out, jobs=1)
        self.assertEqual(sequential["exit_code"], 0)
        for filename in ("final-a.wav", "final-b.wav"):
            self.assertEqual(
                (out / filename).read_bytes(),
                (sequential_out / filename).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()

