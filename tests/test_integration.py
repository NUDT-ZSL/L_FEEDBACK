from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from skillgen.dsp import align_transient
from tests.fixture_factory import make_base_wav, skill_entry, write_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_FFMPEG = shutil.which("ffmpeg")


def run_cli(
    config_path: Path,
    *arguments: str,
    timeout: float = 120.0,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "skillgen", str(config_path), *arguments],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=check,
        env=environment,
    )


def parse_stdout(completed: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(completed.stdout)


def pcm_bytes(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def read_state(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class SkillgenIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="skillgen_test_"))
        self.assets = self.root / "assets"
        self.marker_a = make_base_wav(
            self.assets / "base_a.wav",
            duration=1.5,
            marker_seconds=0.12,
            seed=197,
        )
        self.marker_b = make_base_wav(
            self.assets / "base_b.wav",
            duration=1.4,
            marker_seconds=0.085,
            seed=991,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _two_skill_config(
        self,
        name: str,
        *,
        output_dir: str,
        concurrency: int,
        variant_count: int = 3,
        ffmpeg: str | None = None,
    ) -> Path:
        skills = [
            skill_entry(
                "fireball",
                "assets/base_a.wav",
                self.marker_a,
                variant_count=variant_count,
                random_seed=1234,
            ),
            skill_entry(
                "frost_nova",
                "assets/base_b.wav",
                self.marker_b,
                variant_count=variant_count,
                random_seed=98765,
            ),
        ]
        return write_config(
            self.root / name,
            skills,
            output_dir=output_dir,
            temp_dir="tmp",
            concurrency=concurrency,
            ffmpeg=ffmpeg,
        )

    def _assert_valid_manifest(self, manifest: dict, variant_count: int) -> None:
        self.assertEqual(manifest["version"], "1.0")
        self.assertEqual(manifest["summary"]["variant_count"], 2 * variant_count)
        self.assertEqual(manifest["summary"]["success_count"], 2 * variant_count)
        self.assertEqual(manifest["summary"]["error_count"], 0)
        for skill in manifest["skills"]:
            self.assertEqual(skill["success_count"], variant_count)
            self.assertEqual(skill["variant_count"], variant_count)
            signatures = set()
            for variant in skill["variants"]:
                self.assertEqual(variant["status"], "success")
                parameters = variant["parameters"]
                self.assertGreaterEqual(parameters["pitch_semitones"], -3.0)
                self.assertLessEqual(parameters["pitch_semitones"], 3.0)
                self.assertGreaterEqual(parameters["time_stretch"], 0.9)
                self.assertLessEqual(parameters["time_stretch"], 1.1)
                self.assertGreaterEqual(parameters["lowpass_hz"], 400.0)
                self.assertLessEqual(
                    parameters["lowpass_hz"], variant["sample_rate"] / 2.0
                )
                self.assertLessEqual(
                    abs(variant["transient"]["deviation_ms"]), 5.0
                )
                self.assertLessEqual(
                    abs(
                        variant["loudness"]["after_mlfs"]
                        - variant["loudness"]["target_mlfs"]
                    ),
                    0.5,
                )
                self.assertLessEqual(variant["peak"], 1.0)
                self.assertEqual(
                    variant["sample_count"],
                    round(1.2 * variant["sample_rate"]),
                )
                signatures.add(
                    (
                        parameters["pitch_semitones"],
                        parameters["time_stretch"],
                        parameters["lowpass_hz"],
                    )
                )
            self.assertEqual(len(signatures), variant_count)

    def test_determinism_transient_loudness_and_wav_format(self) -> None:
        config = self._two_skill_config(
            "config.json", output_dir="out", concurrency=2
        )
        first = run_cli(config)
        self.assertEqual(first.returncode, 0, first.stderr)
        first_manifest = parse_stdout(first)
        self._assert_valid_manifest(first_manifest, 3)

        output_dir = self.root / "out"
        first_pcm = {}
        for skill in first_manifest["skills"]:
            for variant in skill["variants"]:
                path = output_dir / variant["file"]
                self.assertTrue(path.is_file())
                with wave.open(str(path), "rb") as handle:
                    self.assertEqual(handle.getnchannels(), 1)
                    self.assertEqual(handle.getsampwidth(), 2)
                    self.assertEqual(handle.getframerate(), 44100)
                    self.assertEqual(handle.getnframes(), variant["sample_count"])
                first_pcm[variant["file"]] = pcm_bytes(path)

        second = run_cli(config, "--force")
        self.assertEqual(second.returncode, 0, second.stderr)
        second_manifest = parse_stdout(second)
        self.assertEqual(first_manifest, second_manifest)
        for file_name, expected in first_pcm.items():
            self.assertEqual(pcm_bytes(output_dir / file_name), expected)

        on_disk = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(on_disk, second_manifest)

    def test_serial_and_concurrent_manifests_are_identical(self) -> None:
        serial_config = self._two_skill_config(
            "serial.json", output_dir="out_serial", concurrency=1
        )
        parallel_config = self._two_skill_config(
            "parallel.json", output_dir="out_parallel", concurrency=4
        )
        serial = run_cli(serial_config)
        parallel = run_cli(parallel_config)
        self.assertEqual(serial.returncode, 0, serial.stderr)
        self.assertEqual(parallel.returncode, 0, parallel.stderr)
        serial_manifest = parse_stdout(serial)
        parallel_manifest = parse_stdout(parallel)
        self.assertEqual(serial_manifest, parallel_manifest)
        self.assertEqual(
            (self.root / "out_serial" / "manifest.json").read_bytes(),
            (self.root / "out_parallel" / "manifest.json").read_bytes(),
        )
        for skill in serial_manifest["skills"]:
            for variant in skill["variants"]:
                file_name = variant["file"]
                self.assertEqual(
                    pcm_bytes(self.root / "out_serial" / file_name),
                    pcm_bytes(self.root / "out_parallel" / file_name),
                )

    def test_config_errors_are_structured_and_nonzero(self) -> None:
        good_skill = skill_entry(
            "fireball",
            "assets/base_a.wav",
            self.marker_a,
        )
        missing_field = dict(good_skill)
        del missing_field["variant_count"]
        config = write_config(
            self.root / "missing_field.json",
            [missing_field],
            output_dir="out_missing_field",
        )
        completed = run_cli(config)
        self.assertEqual(completed.returncode, 2)
        report = parse_stdout(completed)
        self.assertEqual(report["error_code"], "E_CONFIG")
        self.assertTrue(
            any(issue["code"] == "E_CONFIG" for issue in report["issues"])
        )
        self.assertTrue(
            (self.root / "out_missing_field" / "error_report.json").is_file()
        )

        missing_file_skill = skill_entry(
            "missing_asset",
            "assets/not_here.wav",
            self.marker_a,
        )
        missing_file_config = write_config(
            self.root / "missing_file.json",
            [missing_file_skill],
            output_dir="out_missing_file",
        )
        missing = run_cli(missing_file_config)
        self.assertEqual(missing.returncode, 2)
        missing_report = parse_stdout(missing)
        self.assertEqual(missing_report["error_code"], "E_MISSING_FILE")
        self.assertEqual(missing_report["issues"][0]["code"], "E_MISSING_FILE")

    def test_ffmpeg_failure_is_recorded_and_error_is_retried(self) -> None:
        sentinel = self.root / "allow_ffmpeg"
        fake_ffmpeg = self.root / "fake_ffmpeg.sh"
        fake_ffmpeg.write_text(
            "#!/usr/bin/env bash\n"
            f'if [ -f "{sentinel}" ]; then\n'
            f'  exec "{REAL_FFMPEG}" "$@"\n'
            "fi\n"
            "echo 'injected encoder failure' >&2\n"
            "exit 9\n",
            encoding="utf-8",
        )
        fake_ffmpeg.chmod(0o755)
        config = self._two_skill_config(
            "encode_fail.json",
            output_dir="out_encode",
            concurrency=2,
            variant_count=2,
            ffmpeg=str(fake_ffmpeg),
        )
        failed = run_cli(config)
        self.assertEqual(failed.returncode, 0, failed.stderr)
        failed_manifest = parse_stdout(failed)
        self.assertEqual(failed_manifest["summary"]["error_count"], 4)
        self.assertEqual(failed_manifest["summary"]["error_codes"], {"E_ENCODE": 4})
        for skill in failed_manifest["skills"]:
            for variant in skill["variants"]:
                self.assertEqual(variant["error_code"], "E_ENCODE")
                self.assertIn("injected encoder failure", variant["error_message"])
                self.assertFalse((self.root / "out_encode" / variant["file"]).exists())
        self.assertEqual(
            [
                entry
                for entry in (self.root / "tmp").iterdir()
                if entry.name.startswith("skillgen_tmp_")
            ],
            [],
        )

        sentinel.write_text("ok", encoding="utf-8")
        retried = run_cli(config)
        self.assertEqual(retried.returncode, 0, retried.stderr)
        retried_manifest = parse_stdout(retried)
        self.assertEqual(retried_manifest["summary"]["success_count"], 4)
        self.assertEqual(retried_manifest["summary"]["error_count"], 0)
        state = read_state(self.root / "out_encode" / "state.jsonl")
        self.assertEqual(len(state), 4)
        self.assertEqual(len({(r["skill_id"], r["variant_index"]) for r in state}), 4)
        self.assertTrue(all(record["status"] == "success" for record in state))

    @unittest.skipUnless(REAL_FFMPEG, "ffmpeg is required")
    def test_sigkill_resume_skips_successful_variants(self) -> None:
        slow_ffmpeg = self.root / "slow_ffmpeg.sh"
        slow_ffmpeg.write_text(
            "#!/usr/bin/env bash\n"
            f'"{REAL_FFMPEG}" "$@"\n'
            "status=$?\n"
            "sleep 0.45\n"
            "exit $status\n",
            encoding="utf-8",
        )
        slow_ffmpeg.chmod(0o755)
        skill = skill_entry(
            "chain_lightning",
            "assets/base_a.wav",
            self.marker_a,
            variant_count=6,
            random_seed=777,
        )
        config = write_config(
            self.root / "resume.json",
            [skill],
            output_dir="out_resume",
            temp_dir="tmp_resume",
            concurrency=1,
            ffmpeg=str(slow_ffmpeg),
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        process = subprocess.Popen(
            [sys.executable, "-m", "skillgen", str(config)],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=environment,
        )
        state_path = self.root / "out_resume" / "state.jsonl"
        initial_successes: dict[tuple[str, int], dict] = {}
        deadline = time.monotonic() + 25.0
        try:
            while time.monotonic() < deadline:
                if state_path.is_file():
                    try:
                        records = read_state(state_path)
                    except (json.JSONDecodeError, OSError):
                        records = []
                    for record in records:
                        if record.get("status") == "success":
                            initial_successes[
                                (record["skill_id"], record["variant_index"])
                            ] = record
                    if len(initial_successes) >= 1:
                        break
                if process.poll() is not None:
                    self.fail("generator finished before SIGKILL test could interrupt it")
                time.sleep(0.01)
            self.assertGreaterEqual(len(initial_successes), 1)
            self.assertLess(len(initial_successes), 6)
            old_pid = next(iter(initial_successes.values()))["processed_by_pid"]
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=10.0)
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10.0)

        resumed = run_cli(config, timeout=120.0)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        manifest = parse_stdout(resumed)
        self.assertEqual(manifest["summary"]["success_count"], 6)
        self.assertEqual(manifest["summary"]["error_count"], 0)

        final_state = read_state(state_path)
        self.assertEqual(len(final_state), 6)
        by_key = {
            (record["skill_id"], record["variant_index"]): record
            for record in final_state
        }
        self.assertEqual(len(by_key), 6)
        for key in initial_successes:
            self.assertEqual(by_key[key]["processed_by_pid"], old_pid)
            self.assertEqual(by_key[key]["status"], "success")
        manifest_variants = {
            (skill["skill_id"], variant["index"]): variant
            for skill in manifest["skills"]
            for variant in skill["variants"]
        }
        for key, record in by_key.items():
            self.assertEqual(record["variant"], manifest_variants[key])
        self.assertEqual(
            [
                entry
                for entry in (self.root / "tmp_resume").iterdir()
                if entry.name.startswith("skillgen_tmp_")
            ],
            [],
        )

    def test_undetectable_transient_returns_e_transient(self) -> None:
        alignment = align_transient(
            np.ones(44100, dtype=np.float64),
            sample_rate=44100,
            reference_sample=1000,
            predicted_sample=1000,
        )
        self.assertFalse(alignment.aligned)
        self.assertEqual(alignment.error_code, "E_TRANSIENT")

    def test_top_level_single_skill_and_unicode_id(self) -> None:
        config_path = self.root / "single.json"
        config_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "output_dir": "out_single",
                    "temp_dir": "tmp_single",
                    "skill_id": "火球术",
                    "base_sample": "assets/base_a.wav",
                    "variant_count": 1,
                    "target_duration": 1.2,
                    "transient_markers": [float(self.marker_a)],
                    "target_loudness_mlfs": -20.0,
                    "random_seed": 55,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        completed = run_cli(config_path)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = parse_stdout(completed)
        self.assertEqual(manifest["summary"]["success_count"], 1)
        self.assertEqual(
            manifest["skills"][0]["variants"][0]["file"], "火球术_var000.wav"
        )
        self.assertTrue(
            (self.root / "out_single" / "火球术_var000.wav").is_file()
        )

    def test_e_transient_skips_loudness_and_encoding(self) -> None:
        transient_path = self.assets / "early_click.wav"
        sample_rate = 44100
        samples = np.zeros(int(1.5 * sample_rate), dtype=np.float64)
        click_start = int(0.03 * sample_rate)
        click_length = int(0.025 * sample_rate)
        click_time = np.arange(click_length) / sample_rate
        click = (
            np.random.default_rng(44).normal(0.0, 1.0, click_length)
            * np.exp(-click_time / 0.004)
        )
        samples[click_start : click_start + click_length] = 0.8 * click
        wavfile.write(
            str(transient_path),
            sample_rate,
            samples.astype(np.float32),
        )
        config = write_config(
            self.root / "transient_error.json",
            [
                skill_entry(
                    "no_transient",
                    "assets/early_click.wav",
                    int(0.8 * sample_rate),
                )
            ],
            output_dir="out_transient_error",
            temp_dir="tmp_transient_error",
        )
        completed = run_cli(config)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = parse_stdout(completed)
        self.assertEqual(
            manifest["summary"]["error_codes"], {"E_TRANSIENT": 3}
        )
        variant = manifest["skills"][0]["variants"][0]
        self.assertEqual(variant["error_code"], "E_TRANSIENT")
        self.assertIsNone(variant["loudness"]["before_mlfs"])
        self.assertIsNone(variant["loudness"]["after_mlfs"])
        self.assertIsNone(variant["parameters"]["gain_db"])
        self.assertIsNone(variant["sha256"])
        self.assertFalse(
            (self.root / "out_transient_error" / variant["file"]).exists()
        )


if __name__ == "__main__":
    unittest.main()
