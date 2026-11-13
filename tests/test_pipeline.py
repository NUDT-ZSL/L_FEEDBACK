import json
import os
import unittest

import soundfile as sf

from audiopack.assets import AssetStore, FailureInjector
from audiopack.config import Config
from audiopack.pipeline import run_pipeline
from tests.helpers import make_workspace, cleanup, write_manifest


class PipelineEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        (self.root, self.assets, self.manifest_path,
         self.work, self.out, self.config) = make_workspace()
        self.config = Config(max_workers=3)

    def tearDown(self) -> None:
        cleanup(self.root)

    def test_full_run_outputs_reconciled_wav_cue_and_report(self) -> None:
        result = run_pipeline(
            self.manifest_path, self.assets, self.work, self.out, self.config
        )
        self.assertTrue(result.ok, result.issues)
        report = result.report
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["reconciliation_errors"], [])
        self.assertEqual(len(report["manifest_hash"]), 64)

        for lang in ("zh", "en", "ja"):
            wav = os.path.join(self.out, f"track_{lang}.wav")
            info = sf.info(wav)
            self.assertEqual((info.samplerate, info.channels, info.subtype),
                             (48000, 1, "PCM_16"))
            # Duration rounded to whole milliseconds.
            self.assertEqual(info.frames % 48, 0)
            self.assertTrue(os.path.exists(os.path.join(self.out, f"track_{lang}.cue")))
            with open(os.path.join(self.out, f"index_{lang}.json")) as handle:
                index = json.load(handle)
            self.assertEqual(index["total_frames"], info.frames)
            for entry in index["entries"]:
                self.assertLessEqual(entry["end_sample"], info.frames)
                self.assertEqual(entry["lang"], lang)

        # Demo includes at least one shifted retain and one drop.
        self.assertTrue(report["shifts"])
        self.assertTrue(report["dropped_events"])

    def test_interrupted_run_resumes_with_reuse(self) -> None:
        interrupted = run_pipeline(
            self.manifest_path, self.assets, self.work, self.out,
            self.config, stop_after_lang="en",
        )
        self.assertFalse(interrupted.ok)
        self.assertEqual(interrupted.report["status"], "interrupted")
        persisted = os.listdir(os.path.join(self.work, "blocks"))
        self.assertEqual(persisted, ["en"])

        resumed = run_pipeline(
            self.manifest_path, self.assets, self.work, self.out, self.config
        )
        self.assertTrue(resumed.ok, resumed.issues)
        self.assertEqual(resumed.report["reused_langs"], ["en"])
        self.assertEqual(sorted(resumed.report["recomputed_langs"]),
                         ["ja", "zh"])

        # A third identical run reuses every block.
        third = run_pipeline(
            self.manifest_path, self.assets, self.work,
            os.path.join(self.root, "out3"), self.config
        )
        self.assertEqual(sorted(third.report["reused_langs"]),
                         ["en", "ja", "zh"])
        self.assertEqual(third.report["recomputed_langs"], [])

    def test_manifest_change_recomputes_blocks_with_new_manifest_hash(self) -> None:
        run_pipeline(self.manifest_path, self.assets, self.work, self.out,
                     self.config)
        with open(self.manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for event in manifest["events"]:
            if event["id"] == "EVT-005":
                event["tone"] = "changed"
        changed_path = os.path.join(self.root, "manifest_v2.json")
        with open(changed_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        result = run_pipeline(
            changed_path, self.assets, self.work,
            os.path.join(self.root, "out_v2"), self.config
        )
        self.assertTrue(result.ok)
        reasons = {block["lang"]: block["reason"] for block in result.report["blocks"]}
        self.assertTrue(all(reason == "manifest_changed" for reason in reasons.values()))

    def test_missing_asset_fails_with_business_code_and_no_snapshots(self) -> None:
        target = AssetStore(self.config).asset_path(
            self.assets, "victory", "ja"
        )
        result = run_pipeline(
            self.manifest_path, self.assets,
            os.path.join(self.root, "work_fail"),
            os.path.join(self.root, "out_fail"),
            self.config, injector=FailureInjector(missing={target}),
        )
        self.assertFalse(result.ok)
        self.assertIn("E_MISSING_ASSET", result.report["error_codes"])
        blocks_root = os.path.join(self.root, "work_fail", "blocks")
        self.assertFalse(os.path.exists(blocks_root) and os.listdir(blocks_root))

    def test_invalid_manifest_reports_collected_codes(self) -> None:
        bad_path = os.path.join(self.root, "bad.json")
        write_manifest(bad_path, [
            {"id": "X", "type": "t", "time_ms": 10, "priority": 1,
             "langs": ["zh"]},
            {"id": "X", "type": "t", "time_ms": 5, "priority": 1,
             "langs": ["zh"]},
        ])
        result = run_pipeline(
            bad_path, self.assets,
            os.path.join(self.root, "work_bad"),
            os.path.join(self.root, "out_bad"), self.config,
        )
        self.assertFalse(result.ok)
        codes = set(result.report["error_codes"])
        self.assertIn("E_DUP_ID", codes)
        self.assertIn("E_BAD_TIME", codes)


if __name__ == "__main__":
    unittest.main()
