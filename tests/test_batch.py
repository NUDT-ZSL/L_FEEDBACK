import json
import shutil
import tempfile
import unittest
from pathlib import Path

from audiomixer.pipeline import run_batch_file
from audiomixer.config import load_batch
from audiomixer.errors import MixerError
from audiomixer.report import verify_report
from tests.fixtures import make_event, make_scene, make_track, write_batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audiomixer_batch_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_batch_mixes_independent_scenes_and_summarises(self):
        good_a = make_scene(
            "scene_a",
            duration=2.0,
            tracks=[make_track("t", [make_event("a", 0.0, 2.0, amplitude=0.4)])],
        )
        good_b = make_scene(
            "scene_b",
            duration=2.0,
            project_sr=44100,
            tracks=[
                make_track(
                    "t",
                    [make_event("b", 0.0, 2.0, amplitude=0.3, sample_rate=32000)],
                )
            ],
        )
        bad = make_scene(
            "scene_c",
            duration=2.0,
            target_lufs=-10.0,
            peak_limit=-40.0,
            tolerance=0.1,
            tracks=[make_track("t", [make_event("c", 0.0, 2.0, amplitude=0.9)])],
        )
        config_path = self.tmp / "batch.json"
        write_batch(config_path, [good_a, good_b, bad])

        summary, results = run_batch_file(config_path, self.tmp / "summary.json")

        self.assertEqual(summary["total_scenes"], 3)
        self.assertEqual(summary["succeeded_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["status"], "partial_failure")
        failed = summary["failed"][0]
        self.assertEqual(failed["scene_id"], "scene_c")
        self.assertEqual(failed["error_code"], "E_PEAK_LIMIT")
        self.assertFalse(failed["output_wav_present"])

        succeeded_ids = {item["scene_id"] for item in summary["succeeded"]}
        self.assertEqual(succeeded_ids, {"scene_a", "scene_b"})

        # Successful scenes keep independent, individually verifiable artefacts.
        for item in summary["succeeded"]:
            self.assertTrue(Path(item["output_path"]).is_file())
            verification = verify_report(item["report_path"])
            self.assertTrue(verification["checks"]["all_passed"])

        # The failed scene's own report is still written for diagnostics.
        failed_report = json.loads(Path(failed["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(failed_report["status"], "error")
        self.assertEqual(failed_report["error_code"], "E_PEAK_LIMIT")

        # Summary on disk matches the returned object.
        on_disk = json.loads((self.tmp / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["succeeded_count"], 2)
        self.assertEqual(on_disk["failed_count"], 1)

    def test_all_succeed_summary_status_ok(self):
        scene = make_scene(
            "only",
            duration=1.0,
            tracks=[make_track("t", [make_event("e", 0.0, 1.0, amplitude=0.4)])],
        )
        config_path = self.tmp / "batch.json"
        write_batch(config_path, [scene])
        summary, _ = run_batch_file(config_path)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["failed_count"], 0)

    def test_malformed_batch_json_is_invalid_config(self):
        config_path = self.tmp / "broken.json"
        config_path.write_text("{ broken", encoding="utf-8")
        with self.assertRaises(MixerError) as context:
            load_batch(config_path)
        self.assertEqual(context.exception.code, "E_INVALID_CONFIG")


if __name__ == "__main__":
    unittest.main()
