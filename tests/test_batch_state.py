import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_mixer.batch import BatchJob, process_batch
from adaptive_mixer.config import load_config
from adaptive_mixer.engine import MixEngine, RenderInterrupted
from adaptive_mixer.errors import MixerError
from adaptive_mixer.state import load_checkpoint
from adaptive_mixer.wavio import read_wav
from tests.test_engine import build_workspace


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config_path = build_workspace(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_interrupt_and_resume_is_continuous(self):
        cfg = load_config(self.config_path)
        output = self.root / "resume" / "mix.wav"
        state = self.root / "resume" / "state.json"
        engine = MixEngine(cfg, self.root, output, state_file=state)

        interrupted_at = []
        def stop_after_first(scene_id: str) -> bool:
            interrupted_at.append(scene_id)
            return len(interrupted_at) == 1

        with self.assertRaises(RenderInterrupted):
            engine.render(interrupt_after=stop_after_first)
        self.assertTrue(state.is_file())
        checkpoint = load_checkpoint(state)
        self.assertEqual(checkpoint.completed_scenes,
                         [cfg.scenes[0].scene_id])
        self.assertEqual(checkpoint.completed_samples,
                         (cfg.scenes[0].end_frame
                          - cfg.scenes[0].start_frame)
                         * cfg.samples_per_frame)

        # Resume: remaining scenes are rendered and the WAV is continuous.
        engine2 = MixEngine(cfg, self.root, output, state_file=state)
        result = engine2.render()
        self.assertTrue(result.resumed)
        samples, sr = read_wav(output)
        self.assertEqual(sr, 48000)
        self.assertEqual(len(samples), cfg.total_samples)
        self.assertEqual(result.report["timeline"]["alignment_error_samples"],
                         0)
        self.assertEqual(
            len(result.report["checkpoint"]["completed_scenes"]),
            len(cfg.scenes))

    def test_stale_version_is_rejected(self):
        cfg = load_config(self.config_path)
        output = self.root / "stale" / "mix.wav"
        state = self.root / "stale" / "state.json"
        engine = MixEngine(cfg, self.root, output, state_file=state)
        with self.assertRaises(RenderInterrupted):
            engine.render(interrupt_after=lambda sid: True)

        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["crossfade_ms"] = 200.0
        self.config_path.write_text(json.dumps(raw, ensure_ascii=False),
                                    encoding="utf-8")
        changed = load_config(self.config_path)
        engine2 = MixEngine(changed, self.root, output, state_file=state)
        with self.assertRaises(MixerError) as ctx:
            engine2.render()
        self.assertEqual(ctx.exception.code, "E_STATE_VERSION")

    def test_corrupt_state_is_rejected(self):
        cfg = load_config(self.config_path)
        state = self.root / "broken.json"
        state.write_text("{not valid", encoding="utf-8")
        engine = MixEngine(cfg, self.root, self.root / "b.wav",
                           state_file=state)
        with self.assertRaises(MixerError) as ctx:
            engine.render()
        self.assertEqual(ctx.exception.code, "E_STATE_CORRUPT")

    def test_missing_part_file_re_renders(self):
        cfg = load_config(self.config_path)
        output = self.root / "lost" / "mix.wav"
        state = self.root / "lost" / "state.json"
        engine = MixEngine(cfg, self.root, output, state_file=state)
        with self.assertRaises(RenderInterrupted):
            engine.render(interrupt_after=lambda sid: sid == "explore")
        # Simulate loss of the persisted part while the state remains.
        for part in engine.parts_dir.glob("*.npy"):
            part.unlink()
        result = MixEngine(cfg, self.root, output,
                           state_file=state).render()
        samples, _ = read_wav(output)
        self.assertEqual(len(samples), cfg.total_samples)
        self.assertEqual(
            result.report["timeline"]["alignment_error_samples"], 0)


class BatchIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_four_jobs_one_bad_three_good(self):
        jobs = []
        for index in range(4):
            workspace = self.root / f"job{index}"
            config_path = build_workspace(workspace)
            if index == 2:
                # Corrupt the config to force a structured failure.
                config_path.write_text("{broken", encoding="utf-8")
            jobs.append(BatchJob(
                job_id=f"job{index}",
                config_path=str(config_path),
                output_path=str(workspace / "out" / "mix.wav"),
                state_file=str(workspace / "out" / "state.json")))

        summary = process_batch(jobs, max_workers=4)
        self.assertEqual(summary["max_workers"], 4)
        self.assertEqual(summary["total_jobs"], 4)
        self.assertEqual(summary["succeeded"], 3)
        self.assertEqual(summary["failed"], 1)
        by_id = {item["job_id"]: item for item in summary["jobs"]}
        self.assertFalse(by_id["job2"]["ok"])
        self.assertTrue(by_id["job2"]["code"].startswith("E_"))
        self.assertEqual(by_id["job2"]["code"], "E_CONFIG_PARSE")
        for index in (0, 1, 3):
            self.assertTrue(by_id[f"job{index}"]["ok"])
            output = Path(by_id[f"job{index}"]["output_path"])
            samples, sr = read_wav(output)
            self.assertEqual(sr, 48000)
            self.assertGreater(len(samples), 0)


if __name__ == "__main__":
    unittest.main()
