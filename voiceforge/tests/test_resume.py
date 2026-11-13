import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest

from voiceforge.generator import generate_dataset
from voiceforge.hashing import sha256_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(args, env=None, cwd=None):
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "voiceforge", *args],
        cwd=cwd or REPO_ROOT, env=merged_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vf-resume-")
        self.input_dir = os.path.join(self.root, "in")
        self.full_out = os.path.join(self.root, "full")
        self.resume_out = os.path.join(self.root, "resume")
        generate_dataset(self.input_dir, seed=99)
        full = run_cli(["synthesize", self.input_dir, self.full_out])
        self.assertEqual(full.returncode, 0, full.stderr)
        self.baseline = {
            name: sha256_file(os.path.join(self.full_out, name))
            for name in ("voice_track.wav", "timeline.json", "summary.json",
                         "loudness_report.json", "mix_float.f32")}

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _assert_identical(self, output_dir):
        for name, digest in self.baseline.items():
            self.assertEqual(
                sha256_file(os.path.join(output_dir, name)), digest, name)

    def test_repeat_run_summary_is_identical(self):
        again = os.path.join(self.root, "again")
        result = run_cli(["synthesize", self.input_dir, again])
        self.assertEqual(result.returncode, 0, result.stderr)
        self._assert_identical(again)

    def test_interrupt_then_resume_is_byte_identical(self):
        env = {"VOICEFORGE_INTERRUPT_AFTER": "3"}
        first = run_cli(["synthesize", self.input_dir, self.resume_out], env=env)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("中断", first.stdout)
        state_path = os.path.join(self.resume_out, ".checkpoint", "state.json")
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        self.assertEqual(len(state["completed_ids"]), 3)
        self.assertFalse(os.path.exists(
            os.path.join(self.resume_out, "voice_track.wav")))

        second = run_cli(["synthesize", self.input_dir, self.resume_out])
        self.assertEqual(second.returncode, 0, second.stderr)
        self._assert_identical(self.resume_out)

    def test_hard_kill_recovers_from_durable_checkpoint(self):
        env = {"VOICEFORGE_INTERRUPT_AFTER": "2"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "voiceforge", "synthesize",
             self.input_dir, self.resume_out],
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 30
        state_path = os.path.join(self.resume_out, ".checkpoint", "state.json")
        while time.time() < deadline and not os.path.exists(state_path):
            time.sleep(0.05)
        self.assertTrue(os.path.exists(state_path))
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
        proc.stdout.close()
        proc.stderr.close()
        self.assertNotEqual(proc.returncode, 0)

        second = run_cli(["synthesize", self.input_dir, self.resume_out])
        self.assertEqual(second.returncode, 0, second.stderr)
        self._assert_identical(self.resume_out)

    def test_real_sigint_flushes_checkpoint_and_resumes(self):
        env = {"VOICEFORGE_INTERRUPT_AFTER": "0"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "voiceforge", "synthesize",
             self.input_dir, self.resume_out],
            cwd=REPO_ROOT, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        state_path = os.path.join(self.resume_out, ".checkpoint", "state.json")
        deadline = time.time() + 30
        while time.time() < deadline and not os.path.exists(state_path):
            time.sleep(0.05)
        time.sleep(0.1)
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertIn("中断", stdout)

        second = run_cli(["synthesize", self.input_dir, self.resume_out])
        self.assertEqual(second.returncode, 0, second.stderr)
        self._assert_identical(self.resume_out)

    def test_changed_shard_invalidates_player_and_rebuilds_identical_output(self):
        target = None
        with open(os.path.join(self.input_dir, "events.json"),
                  encoding="utf-8") as handle:
            manifest = json.load(handle)
        for player in manifest["players"]:
            for shard in player["segments"]:
                if shard["resample_required"]:
                    target = shard
                    break
        self.assertIsNotNone(target)
        path = os.path.join(self.input_dir, target["file"])
        with open(path, "r+b") as handle:
            handle.seek(44)
            handle.write(b"\x55\x55")

        changed = os.path.join(self.root, "changed")
        run_cli(["synthesize", self.input_dir, self.full_out])
        result = run_cli(["synthesize", self.input_dir, changed])
        self.assertEqual(result.returncode, 0, result.stderr)
        track_digest = sha256_file(os.path.join(changed, "voice_track.wav"))
        self.assertNotEqual(track_digest, self.baseline["voice_track.wav"])

        resume_changed = os.path.join(self.root, "changed-resume")
        env = {"VOICEFORGE_INTERRUPT_AFTER": "4"}
        first = run_cli(["synthesize", self.input_dir, resume_changed], env=env)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = run_cli(["synthesize", self.input_dir, resume_changed])
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            sha256_file(os.path.join(resume_changed, "voice_track.wav")),
            track_digest)
        self.assertEqual(
            sha256_file(os.path.join(resume_changed, "summary.json")),
            sha256_file(os.path.join(changed, "summary.json")))


if __name__ == "__main__":
    unittest.main()

