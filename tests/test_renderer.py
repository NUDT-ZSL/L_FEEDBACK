"""End-to-end acceptance tests for the crystal renderer."""

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

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, "-m", "crystal_renderer.cli"]


def run_cli(args, cwd=ROOT):
    return subprocess.run(
        CLI + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def load_frames(out_dir):
    frames_dir = os.path.join(out_dir, "frames")
    result = {}
    for name in sorted(os.listdir(frames_dir)):
        if name.endswith(".npy"):
            step = int(name.split("_")[1].split(".")[0])
            result[step] = np.load(os.path.join(frames_dir, name))
    return result


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="crystal_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def gen_scene(self, name="scene.json", seed=7, frames=40, anomaly=False):
        out = self.path(name)
        args = ["gen", "--out", out, "--seed", str(seed), "--frames", str(frames)]
        if anomaly:
            args.append("--anomaly")
        result = run_cli(args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return out


class TestVerify(Workspace):
    def test_incremental_matches_full_and_is_cheaper(self):
        scene = self.gen_scene()
        out = self.path("verify")
        result = run_cli(["verify", "--config", scene, "--out", out])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.load(open(os.path.join(out, "verify_report.json")))
        self.assertTrue(report["all_frames_identical"])
        self.assertGreater(report["single_change_frames"], 0)
        self.assertTrue(report["single_change_incr_always_cheaper"])
        self.assertLess(report["incr_total_shaded"], report["full_total_shaded"])

    def test_determinism_across_runs(self):
        scene = self.gen_scene()
        frames_a = self.path("a")
        frames_b = self.path("b")
        for out in (frames_a, frames_b):
            result = run_cli(["run", "--config", scene, "--out", out, "--mode", "incremental"])
            self.assertEqual(result.returncode, 0, result.stderr)
        a = load_frames(frames_a)
        b = load_frames(frames_b)
        self.assertEqual(a.keys(), b.keys())
        for step in a:
            self.assertTrue(np.array_equal(a[step], b[step]), f"frame {step} diverged")

    def test_input_order_independence(self):
        scene = self.gen_scene()
        config = json.load(open(scene))
        config["crystals"] = list(reversed(config["crystals"]))
        shuffled = self.path("shuffled.json")
        with open(shuffled, "w") as handle:
            json.dump(config, handle)
        out_a = self.path("order_a")
        out_b = self.path("order_b")
        run_cli(["run", "--config", scene, "--out", out_a, "--mode", "full"])
        run_cli(["run", "--config", shuffled, "--out", out_b, "--mode", "full"])
        a = load_frames(out_a)
        b = load_frames(out_b)
        for step in a:
            self.assertTrue(np.array_equal(a[step], b[step]), f"frame {step} diverged")
        report_a = json.load(open(os.path.join(out_a, "render_report.json")))
        report_b = json.load(open(os.path.join(out_b, "render_report.json")))
        seq_a = [tuple(f["draw_sequence"]) for f in report_a["per_frame"]]
        seq_b = [tuple(f["draw_sequence"]) for f in report_b["per_frame"]]
        self.assertEqual(seq_a, seq_b)


class TestAnomalies(Workspace):
    def test_error_codes_and_partial_render(self):
        scene = self.gen_scene(name="anomaly.json", seed=11, frames=12, anomaly=True)
        out = self.path("anomaly_out")
        result = run_cli(["run", "--config", scene, "--out", out, "--mode", "incremental"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for code in ("E101", "E102", "E103", "E105"):
            self.assertIn(code, result.stderr)
        report = json.load(open(os.path.join(out, "render_report.json")))
        codes = {entry["code"] for entry in report["errors"]}
        self.assertEqual({"E101", "E102", "E103", "E105"}, codes)
        frames = load_frames(out)
        self.assertEqual(len(frames), 12)
        dropped_ids = {"bad_degenerate", "bad_zero_area", "bad_ior"}
        for record in report["per_frame"]:
            self.assertTrue(dropped_ids.isdisjoint(record["draw_sequence"]))
            self.assertIn("bad_keyframe", record["draw_sequence"])


class TestSnapshots(Workspace):
    def _reference_run(self, scene):
        out = self.path("reference")
        result = run_cli(["run", "--config", scene, "--out", out, "--mode", "incremental"])
        self.assertEqual(result.returncode, 0, result.stderr)
        return load_frames(out)

    def test_checksum_mismatch_recomputes_from_zero(self):
        scene = self.gen_scene(frames=21)
        reference = self._reference_run(scene)
        out = self.path("corrupt")
        run_cli(["run", "--config", scene, "--out", out, "--mode", "incremental"])
        snap_dir = os.path.join(out, "snapshots")
        metas = sorted(n for n in os.listdir(snap_dir) if n.endswith(".meta.json"))
        latest = metas[-1]
        buffer_name = latest.replace(".meta.json", ".npy")
        buffer_path = os.path.join(snap_dir, buffer_name)
        data = bytearray(open(buffer_path, "rb").read())
        data[-1] ^= 0xFF
        open(buffer_path, "wb").write(bytes(data))
        shutil.rmtree(os.path.join(out, "frames"))
        result = run_cli(
            ["run", "--config", scene, "--out", out, "--mode", "incremental", "--resume"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("E109", result.stderr)
        recovery = json.load(open(os.path.join(out, "recovery_report.json")))
        event = recovery["events"][0]
        self.assertEqual(event["source"], "config")
        self.assertEqual(event["frame"], 0)
        frames = load_frames(out)
        self.assertEqual(frames.keys(), reference.keys())
        for step in reference:
            self.assertTrue(np.array_equal(frames[step], reference[step]))

    def test_version_mismatch_recomputes_from_zero(self):
        scene = self.gen_scene(frames=21)
        reference = self._reference_run(scene)
        out = self.path("versioned")
        run_cli(["run", "--config", scene, "--out", out, "--mode", "incremental"])
        snap_dir = os.path.join(out, "snapshots")
        metas = sorted(n for n in os.listdir(snap_dir) if n.endswith(".meta.json"))
        meta_path = os.path.join(snap_dir, metas[-1])
        meta = json.load(open(meta_path))
        meta["version"] = -1
        with open(meta_path, "w") as handle:
            json.dump(meta, handle)
        shutil.rmtree(os.path.join(out, "frames"))
        result = run_cli(
            ["run", "--config", scene, "--out", out, "--mode", "incremental", "--resume"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("E108", result.stderr)
        frames = load_frames(out)
        for step in reference:
            self.assertTrue(np.array_equal(frames[step], reference[step]))

    def test_sigkill_resume_matches_uninterrupted(self):
        scene = self.gen_scene(frames=30)
        reference = self._reference_run(scene)
        out = self.path("killed")
        os.makedirs(out, exist_ok=True)
        proc = subprocess.Popen(
            CLI + [
                "run", "--config", scene, "--out", out,
                "--mode", "incremental", "--delay", "0.05",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.6)
        proc.kill()
        proc.wait()
        self.assertNotEqual(proc.returncode, 0)
        result = run_cli(
            ["run", "--config", scene, "--out", out, "--mode", "incremental", "--resume"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        recovery = json.load(open(os.path.join(out, "recovery_report.json")))
        self.assertIn(recovery["events"][0]["source"], ("snapshot", "config"))
        frames = load_frames(out)
        self.assertEqual(frames.keys(), reference.keys())
        for step in reference:
            self.assertTrue(np.array_equal(frames[step], reference[step]))

    def test_sigterm_graceful_interrupt_then_resume(self):
        scene = self.gen_scene(frames=30)
        reference = self._reference_run(scene)
        out = self.path("sigterm")
        os.makedirs(out, exist_ok=True)
        proc = subprocess.Popen(
            CLI + [
                "run", "--config", scene, "--out", out,
                "--mode", "incremental", "--delay", "0.05",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.6)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        self.assertEqual(proc.returncode, 3)
        recovery = json.load(open(os.path.join(out, "recovery_report.json")))
        events = recovery["events"]
        self.assertEqual(events[-1]["event"], "interrupted")
        self.assertEqual(events[-1]["signal"], "SIGTERM")
        result = run_cli(
            ["run", "--config", scene, "--out", out, "--mode", "incremental", "--resume"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        recovery = json.load(open(os.path.join(out, "recovery_report.json")))
        self.assertEqual(recovery["events"][0]["source"], "snapshot")
        frames = load_frames(out)
        self.assertEqual(frames.keys(), reference.keys())
        for step in reference:
            self.assertTrue(np.array_equal(frames[step], reference[step]))


if __name__ == "__main__":
    unittest.main()
