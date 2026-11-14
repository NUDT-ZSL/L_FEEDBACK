from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fogcity.output import pretty_json_bytes
from fogcity.sample import build_sample_document
from fogcity.util import strip_volatile


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fogcity", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def error_payload(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in process.stderr.splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="fogcity-test-")
        self.root = Path(self.temp.name)
        document = build_sample_document(
            width=96,
            height=54,
            fps=1.0,
            start_time=0.0,
            end_time=1.0,
            fog_steps=10,
            shadow_resolution=192,
        )
        self.scene_path = self.root / "scene.json"
        self.scene_path.write_bytes(pretty_json_bytes(document))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_two_runs_are_byte_identical_and_verifiable(self) -> None:
        out_a = self.root / "out-a"
        out_b = self.root / "out-b"
        out_parallel = self.root / "out-parallel"

        first = run_cli(
            "render",
            "--scene",
            str(self.scene_path),
            "--out",
            str(out_a),
            "--parallel-frames",
            "1",
        )
        second = run_cli(
            "render",
            "--scene",
            str(self.scene_path),
            "--out",
            str(out_b),
            "--parallel-frames",
            "1",
        )
        parallel = run_cli(
            "render",
            "--scene",
            str(self.scene_path),
            "--out",
            str(out_parallel),
            "--parallel-frames",
            "2",
        )
        for process in (first, second, parallel):
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(json.loads(process.stdout.strip().splitlines()[-1])["ok"])

        hashes_a = file_hashes(out_a)
        hashes_b = file_hashes(out_b)
        hashes_parallel = file_hashes(out_parallel)
        for name in hashes_a:
            if name.endswith(".png"):
                self.assertEqual(hashes_a[name], hashes_b[name], name)
                self.assertEqual(hashes_a[name], hashes_parallel[name], name)

        manifest_a = json.loads((out_a / "stats.json").read_text())
        manifest_b = json.loads((out_b / "stats.json").read_text())
        manifest_parallel = json.loads((out_parallel / "stats.json").read_text())
        self.assertEqual(strip_volatile(manifest_a), strip_volatile(manifest_b))
        self.assertEqual(strip_volatile(manifest_a), strip_volatile(manifest_parallel))
        self.assertIn("generated_at", manifest_a)

        for frame_index in range(2):
            stats_path = out_a / f"frame_{frame_index:04d}.json"
            stats = json.loads(stats_path.read_text())
            image = (out_a / stats["image"]).read_bytes()
            self.assertEqual(hashlib.sha256(image).hexdigest(), stats["image_sha256"])
            panels = stats["transparent_sort"]["panels"]
            keys = [(panel["depth_key"], -panel["panel_id"]) for panel in panels]
            self.assertEqual(keys, sorted(keys, reverse=True))
            self.assertFalse(
                any(
                    panel["panel_id"] == 105 and panel["rendered"]
                    for panel in panels
                )
            )

        snapshot_before = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in out_a.iterdir()
            if path.is_file()
        }
        verify = run_cli(
            "render",
            "--scene",
            str(self.scene_path),
            "--out",
            str(out_a),
            "--validate-only",
            "--parallel-frames",
            "2",
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertTrue(json.loads(verify.stdout)["ok"])
        snapshot_after = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in out_a.iterdir()
            if path.is_file()
        }
        self.assertEqual(snapshot_before, snapshot_after)

        corrupt = self.root / "corrupt"
        shutil.copytree(out_a, corrupt)
        corrupt_png = corrupt / "frame_0001.png"
        corrupt_png.write_bytes(corrupt_png.read_bytes() + b"corrupted")
        mismatch = run_cli(
            "render",
            "--scene",
            str(self.scene_path),
            "--out",
            str(corrupt),
            "--validate-only",
        )
        self.assertEqual(mismatch.returncode, 41, mismatch.stderr)
        self.assertEqual(error_payload(mismatch)["error"]["code"], "E_VERIFY_MISMATCH")

    def test_invalid_inputs_have_structured_errors(self) -> None:
        missing = run_cli(
            "render",
            "--scene",
            str(self.root / "missing.json"),
            "--out",
            str(self.root / "out"),
        )
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(error_payload(missing)["error"]["code"], "E_SCENE_NOT_FOUND")

        document = json.loads(self.scene_path.read_text())
        duplicate = copy_document(document)
        duplicate["camera_keyframes"][2]["time"] = duplicate["camera_keyframes"][1]["time"]
        duplicate_path = self.root / "duplicate.json"
        duplicate_path.write_bytes(pretty_json_bytes(duplicate))
        duplicate_process = run_cli(
            "render", "--scene", str(duplicate_path), "--out", str(self.root / "out2")
        )
        self.assertEqual(duplicate_process.returncode, 12)
        self.assertEqual(
            error_payload(duplicate_process)["error"]["code"],
            "E_KEYFRAME_TIME_DUPLICATE",
        )

        non_monotonic = copy_document(document)
        non_monotonic["camera_keyframes"][2]["time"] = 0.5
        non_monotonic_path = self.root / "non-monotonic.json"
        non_monotonic_path.write_bytes(pretty_json_bytes(non_monotonic))
        non_monotonic_process = run_cli(
            "render",
            "--scene",
            str(non_monotonic_path),
            "--out",
            str(self.root / "out3"),
        )
        self.assertEqual(non_monotonic_process.returncode, 11)
        self.assertEqual(
            error_payload(non_monotonic_process)["error"]["code"],
            "E_KEYFRAME_TIME_ORDER",
        )

        invalid = copy_document(document)
        invalid["render"]["height"] = 0
        invalid_path = self.root / "invalid.json"
        invalid_path.write_bytes(pretty_json_bytes(invalid))
        invalid_process = run_cli(
            "render", "--scene", str(invalid_path), "--out", str(self.root / "out4")
        )
        self.assertEqual(invalid_process.returncode, 10)
        self.assertEqual(error_payload(invalid_process)["error"]["code"], "E_SCENE_FIELD")

        missing_output = run_cli(
            "render",
            "--scene",
            str(self.scene_path),
            "--out",
            str(self.root / "missing-output"),
            "--validate-only",
        )
        self.assertEqual(missing_output.returncode, 40)
        self.assertEqual(
            error_payload(missing_output)["error"]["code"],
            "E_VERIFY_OUTPUT_MISSING",
        )

    def test_generate_sample_and_recompute_interpolation(self) -> None:
        generated = self.root / "generated.json"
        process = run_cli(
            "generate-sample",
            "--path",
            str(generated),
            "--width",
            "64",
            "--height",
            "36",
            "--fps",
            "1",
            "--start-time",
            "0",
            "--end-time",
            "1",
            "--steps",
            "8",
            "--shadow-resolution",
            "128",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        document = json.loads(generated.read_text())
        self.assertEqual(document["schema_version"], 1)

        output = self.root / "generated-out"
        render = run_cli(
            "render", "--scene", str(generated), "--out", str(output)
        )
        self.assertEqual(render.returncode, 0, render.stderr)
        stats = json.loads((output / "frame_0001.json").read_text())
        start = document["camera_keyframes"][0]["position"]
        end = document["camera_keyframes"][1]["position"]
        expected = [(a + b) * 0.5 for a, b in zip(start, end)]
        self.assertEqual(stats["camera"]["position"], [round(value, 6) for value in expected])
        self.assertAlmostEqual(stats["camera"]["segment"]["alpha"], 0.5)
        self.assertEqual(stats["camera"]["frame_index"], 1)


def copy_document(document: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(document))


if __name__ == "__main__":
    unittest.main()

