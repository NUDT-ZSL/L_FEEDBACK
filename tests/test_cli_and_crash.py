from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from ruin_bake.snapshot import load_snapshot

from .helpers import (
    ROOT,
    TEST_TMP,
    cli_env,
    config_dict,
    run_cli,
    write_json,
)


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        cls.directory = Path(cls._temp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_full_report_and_summary(self) -> None:
        config = write_json(self.directory / "full.json", config_dict(samples=8, workers=1, chunk_size=5))
        snapshot = self.directory / "full.snapshot"
        result = run_cli("full", "--config", config, "--out", snapshot)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["command"], "full")
        self.assertGreater(report["probe_count"], 0)
        self.assertEqual(report["affected_probe_count"], report["probe_count"])
        self.assertEqual(report["samples"], 8)
        self.assertEqual(report["snapshot_version"], 1)

        summary = run_cli("summary", "--snapshot", snapshot)
        self.assertEqual(summary.returncode, 0, summary.stderr)
        summary_report = json.loads(summary.stdout)
        self.assertEqual(summary_report["probe_count"], report["probe_count"])
        self.assertEqual(summary_report["snapshot_checksum"], report["snapshot_checksum"])

    def test_parallel_workers_and_chunk_sizes_produce_identical_snapshot_bytes(self) -> None:
        serial_config = write_json(
            self.directory / "serial.json",
            config_dict(samples=10, workers=1, chunk_size=100),
        )
        parallel_config = write_json(
            self.directory / "parallel.json",
            config_dict(samples=10, workers=2, chunk_size=3),
        )
        serial_path = self.directory / "serial.snapshot"
        parallel_path = self.directory / "parallel.snapshot"
        serial = run_cli("full", "--config", serial_config, "--out", serial_path)
        parallel = run_cli("full", "--config", parallel_config, "--out", parallel_path)
        self.assertEqual(serial.returncode, 0, serial.stderr)
        self.assertEqual(parallel.returncode, 0, parallel.stderr)
        self.assertEqual(serial_path.read_bytes(), parallel_path.read_bytes())
        serial_report = json.loads(serial.stdout)
        parallel_report = json.loads(parallel.stdout)
        self.assertEqual(serial_report["snapshot_checksum"], parallel_report["snapshot_checksum"])
        self.assertNotEqual(serial_report["chunk_count"], parallel_report["chunk_count"])

    def test_chunked_incremental_matches_full_patch_bit_for_bit(self) -> None:
        patch = {"geometry_updates": {"3": {"albedo": [0.91, 0.13, 0.06]}}}
        base_config = write_json(
            self.directory / "inc-base-config.json",
            config_dict(samples=8, workers=1, chunk_size=5),
        )
        serial_config = write_json(
            self.directory / "inc-serial-config.json",
            config_dict(samples=8, workers=1, chunk_size=1, patch=patch),
        )
        parallel_config = write_json(
            self.directory / "inc-parallel-config.json",
            config_dict(samples=8, workers=2, chunk_size=3, patch=patch),
        )
        full_config = write_json(
            self.directory / "inc-full-config.json",
            config_dict(samples=8, workers=2, chunk_size=2, patch=patch),
        )
        base_path = self.directory / "inc-base.snapshot"
        serial_path = self.directory / "inc-serial.snapshot"
        parallel_path = self.directory / "inc-parallel.snapshot"
        full_path = self.directory / "inc-full.snapshot"
        base = run_cli("full", "--config", base_config, "--out", base_path)
        self.assertEqual(base.returncode, 0, base.stderr)
        serial = run_cli(
            "incremental", "--config", serial_config, "--base", base_path, "--out", serial_path
        )
        parallel = run_cli(
            "incremental", "--config", parallel_config, "--base", base_path, "--out", parallel_path
        )
        full = run_cli("full", "--config", full_config, "--out", full_path)
        self.assertEqual(serial.returncode, 0, serial.stderr)
        self.assertEqual(parallel.returncode, 0, parallel.stderr)
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(serial_path.read_bytes(), parallel_path.read_bytes())
        incremental_snapshot = load_snapshot(serial_path)
        full_snapshot = load_snapshot(full_path)
        self.assertEqual(
            [(record.probe_id, record.position, record.normal, record.irradiance) for record in incremental_snapshot.probes],
            [(record.probe_id, record.position, record.normal, record.irradiance) for record in full_snapshot.probes],
        )
        parallel_report = json.loads(parallel.stdout)
        self.assertGreater(parallel_report["reused_probe_count"], 0)
        self.assertLess(parallel_report["affected_probe_count"], parallel_report["probe_count"])

    def test_corrupt_snapshot_summary_fails_with_json_error(self) -> None:
        bad_path = self.directory / "corrupt.snapshot"
        bad_path.write_bytes(b"not a snapshot")
        result = run_cli("summary", bad_path)
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertFalse(error["ok"])
        self.assertIn("error", error)

    def test_incremental_without_config_inherits_base_scene_bits(self) -> None:
        config = write_json(self.directory / "inherit.json", config_dict(samples=8, workers=1, chunk_size=5))
        base_path = self.directory / "inherit-base.snapshot"
        next_path = self.directory / "inherit-next.snapshot"
        base = run_cli("full", "--config", config, "--out", base_path)
        self.assertEqual(base.returncode, 0, base.stderr)
        base_report = json.loads(base.stdout)
        incremental = run_cli("incremental", "--base", base_path, "--out", next_path)
        self.assertEqual(incremental.returncode, 0, incremental.stderr)
        incremental_report = json.loads(incremental.stdout)
        self.assertEqual(incremental_report["affected_probe_count"], 0)
        self.assertEqual(incremental_report["reused_probe_count"], base_report["probe_count"])
        self.assertEqual(incremental_report["scene_hash"], base_report["scene_hash"])
        self.assertEqual(incremental_report["snapshot_revision"], base_report["snapshot_revision"] + 1)

    def test_invalid_parameter_has_nonzero_exit(self) -> None:
        result = run_cli("full", "--samples", "0", "--out", self.directory / "never.snapshot")
        self.assertNotEqual(result.returncode, 0)
        error = json.loads(result.stderr)
        self.assertFalse(error["ok"])


class CrashRecoveryTests(unittest.TestCase):
    def test_sigkill_during_temp_write_recovers_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temp:
            directory = Path(temp)
            base_config = write_json(
                directory / "base.json",
                config_dict(samples=8, workers=1, chunk_size=4),
            )
            new_config = write_json(
                directory / "new.json",
                config_dict(
                    samples=8,
                    workers=1,
                    chunk_size=4,
                    patch={"geometry_updates": {"3": {"albedo": [0.95, 0.15, 0.05]}}},
                ),
            )
            snapshot_path = directory / "recover.snapshot"
            base_result = run_cli("full", "--config", base_config, "--out", snapshot_path)
            self.assertEqual(base_result.returncode, 0, base_result.stderr)
            base_before = load_snapshot(snapshot_path)
            complete_size = snapshot_path.stat().st_size

            child_env = cli_env(
                RUIN_BAKE_TEST_HOLD_AFTER_TEMP_BYTES="1024",
                RUIN_BAKE_TEST_HOLD_MS="60000",
                RUIN_BAKE_TEST_WRITE_CHUNK_BYTES="1024",
            )
            command = [
                sys.executable,
                "-m",
                "ruin_bake",
                "incremental",
                "--config",
                str(new_config),
                "--base",
                str(snapshot_path),
                "--out",
                str(snapshot_path),
            ]
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            temp_path = snapshot_path.with_name(snapshot_path.name + ".tmp")
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if temp_path.exists() and temp_path.stat().st_size >= 1024:
                    break
                if child.poll() is not None:
                    stdout, stderr = child.communicate()
                    self.fail(f"writer exited before crash point: {stdout!r} {stderr!r}")
                time.sleep(0.02)
            else:
                child.kill()
                child.communicate()
                self.fail("timed out waiting for temporary snapshot")

            self.assertLess(temp_path.stat().st_size, complete_size)
            child.kill()
            child.communicate(timeout=10)
            self.assertIsNotNone(child.returncode)

            recovered = load_snapshot(snapshot_path)
            self.assertEqual(recovered.checksum, base_before.checksum)
            self.assertEqual(recovered.scene_hash, base_before.scene_hash)
            self.assertEqual(recovered.revision, base_before.revision)
            self.assertTrue(temp_path.exists(), "partial temp should remain after SIGKILL")

            retry = run_cli(
                "incremental",
                "--config",
                new_config,
                "--base",
                snapshot_path,
                "--out",
                snapshot_path,
            )
            self.assertEqual(retry.returncode, 0, retry.stderr)
            retry_report = json.loads(retry.stdout)
            self.assertEqual(retry_report["snapshot_revision"], base_before.revision + 1)
            updated = load_snapshot(snapshot_path)
            self.assertNotEqual(updated.scene_hash, base_before.scene_hash)
            self.assertFalse(temp_path.exists())

            full_path = directory / "full-new.snapshot"
            full = run_cli("full", "--config", new_config, "--out", full_path)
            self.assertEqual(full.returncode, 0, full.stderr)
            full_snapshot = load_snapshot(full_path)
            self.assertEqual(
                tuple(record.irradiance for record in updated.probes),
                tuple(record.irradiance for record in full_snapshot.probes),
            )


if __name__ == "__main__":
    unittest.main()

