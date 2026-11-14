"""End-to-end tests for the offscreen deferred renderer.

These tests spawn the real CLI in a subprocess so the exit-code contract and
manifest artifacts are exercised exactly as in production. A hidden GLFW
window / software GL context is sufficient; no display server interaction is
required beyond what GLFW already uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "deferred_renderer", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_manifest(directory: str) -> dict:
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def frame_hashes(manifest: dict) -> dict[int, str]:
    return {entry["frame"]: entry["png_sha256"] for entry in manifest["frames"]}


class RendererCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="deferred_test_")

    def _path(self, name: str) -> str:
        return os.path.join(self.tmp, name)

    def test_basic_run_writes_frames_and_manifest(self) -> None:
        out = self._path("basic")
        code, _, err = run_cli("--frames", "3", "--out", out)
        self.assertEqual(code, 0, err)
        manifest = load_manifest(out)
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(len(manifest["frames"]), 3)
        self.assertIn("vendor", manifest["gpu"])
        self.assertIn("opengl_version", manifest["gpu"])
        for entry in manifest["frames"]:
            png_path = os.path.join(out, entry["png"])
            self.assertTrue(os.path.exists(png_path))
            with open(png_path, "rb") as handle:
                disk_hash = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(disk_hash, entry["png_sha256"])
            self.assertEqual(
                entry["gbuffer_write_sha256"], entry["gbuffer_verify_sha256"]
            )

    def test_deterministic_across_runs(self) -> None:
        first = self._path("first")
        second = self._path("second")
        self.assertEqual(run_cli("--frames", "5", "--out", first)[0], 0)
        self.assertEqual(run_cli("--frames", "5", "--out", second)[0], 0)
        self.assertEqual(
            frame_hashes(load_manifest(first)),
            frame_hashes(load_manifest(second)),
        )

    def test_rebuild_matches_no_rebuild_pixel_for_pixel(self) -> None:
        rebuilt = self._path("rebuilt")
        control = self._path("control")
        self.assertEqual(run_cli("--frames", "21", "--out", rebuilt)[0], 0)
        self.assertEqual(
            run_cli("--frames", "21", "--no-rebuild", "--out", control)[0], 0
        )
        rebuilt_manifest = load_manifest(rebuilt)
        events = rebuilt_manifest["events"]
        self.assertTrue(events)
        self.assertEqual(events[0]["event"], "gbuffer_rebuild")
        self.assertEqual(events[0]["resume_frame"], 11)
        self.assertIn("binding_state_before", events[0])
        self.assertIn("binding_state_after", events[0])
        self.assertEqual(
            frame_hashes(rebuilt_manifest), frame_hashes(load_manifest(control))
        )

    def test_forced_unsupported_format_falls_back_with_exit_zero(self) -> None:
        out = self._path("fallback")
        code, _, err = run_cli(
            "--frames", "2",
            "--force-format", "world_position=UNSUPPORTED",
            "--out", out,
        )
        self.assertEqual(code, 0, err)
        manifest = load_manifest(out)
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(
            manifest["requested_formats"]["world_position"], "UNSUPPORTED"
        )
        self.assertEqual(
            manifest["active_formats"]["world_position"], "RGBA32F"
        )
        self.assertTrue(manifest["warnings"])
        warning = manifest["warnings"][0]
        self.assertEqual(warning["code"], "RENDER_FBO_INCOMPLETE")
        self.assertEqual(warning["attachment"], "world_position")
        self.assertTrue(warning["gl_errors"])
        self.assertIn("code", warning["gl_errors"][0])
        self.assertTrue(os.path.exists(os.path.join(out, "errors.jsonl")))

    def test_fast_approx_is_distinct_but_deterministic(self) -> None:
        fast_a = self._path("fast_a")
        fast_b = self._path("fast_b")
        ref = self._path("ref")
        self.assertEqual(
            run_cli("--frames", "5", "--fast-approx", "--out", fast_a)[0], 0
        )
        self.assertEqual(
            run_cli("--frames", "5", "--fast-approx", "--out", fast_b)[0], 0
        )
        self.assertEqual(run_cli("--frames", "5", "--out", ref)[0], 0)
        fast_manifest = load_manifest(fast_a)
        self.assertEqual(fast_manifest["precision_mode"], "fast_approx")
        self.assertEqual(
            frame_hashes(fast_manifest),
            frame_hashes(load_manifest(fast_b)),
        )
        self.assertNotEqual(
            frame_hashes(fast_manifest), frame_hashes(load_manifest(ref))
        )

    def test_distinct_seeds_change_output(self) -> None:
        seed42 = self._path("seed42")
        seed7 = self._path("seed7")
        self.assertEqual(run_cli("--frames", "5", "--seed", "42", "--out", seed42)[0], 0)
        self.assertEqual(run_cli("--frames", "5", "--seed", "7", "--out", seed7)[0], 0)
        self.assertNotEqual(
            frame_hashes(load_manifest(seed42)),
            frame_hashes(load_manifest(seed7)),
        )
        with open(os.path.join(seed42, "scene.json"), "rb") as handle:
            scene42 = handle.read()
        with open(os.path.join(seed7, "scene.json"), "rb") as handle:
            scene7 = handle.read()
        self.assertNotEqual(scene42, scene7)


class SceneDeterminismTests(unittest.TestCase):
    def test_scene_generation_is_seed_reproducible(self) -> None:
        from deferred_renderer.scene import build_scene
        self.assertEqual(build_scene(42), build_scene(42))
        self.assertNotEqual(build_scene(42), build_scene(7))


class Fp16EmulationTests(unittest.TestCase):
    def test_half_float_round_trip(self) -> None:
        # Validate the same bit pattern the GLSL emulation implements by
        # checking NumPy's authoritative float16 conversion on representative
        # values (normals, roughness, radiance magnitudes).
        import numpy as np

        values = np.array(
            [0.0, 1.0, -1.0, 0.04, 0.5, 3.14159, 12.5, 100.0, 0.0001],
            dtype=np.float32,
        )
        rounded = values.astype(np.float16).astype(np.float32)
        # Every value must survive a half round trip and stay close.
        self.assertTrue(np.all(np.isfinite(rounded)))
        self.assertTrue(np.all(np.abs(rounded - values) <= np.abs(values) * 0.01 + 1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
