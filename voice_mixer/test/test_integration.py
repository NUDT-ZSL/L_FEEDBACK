import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from voice_mixer.exporter import export_project
from voice_mixer.parser import parse_project
from voice_mixer.sample_project import build_clipping_project, build_sample_project
from voice_mixer.timeline import build_timeline


class ExportIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mixer-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _project(self, factory, name):
        data = factory(
            output_wav=os.path.join(self.tmp, f"{name}.wav"),
            manifest=os.path.join(self.tmp, f"{name}.json"),
            work_dir=os.path.join(self.tmp, f"{name}_work"),
        )
        return parse_project(data)

    def _export(self, project, flag=None):
        timeline = build_timeline(project)
        code, manifest = export_project(project, timeline, interrupt_flag=flag, logger=lambda _: None)
        return code, manifest, timeline

    def test_wav_header_and_manifest_complete(self):
        project = self._project(build_sample_project, "sample")
        code, manifest, _ = self._export(project)
        self.assertEqual(code, 0)
        with wave.open(project.export.output_wav, "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnchannels(), 2)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getnframes(), manifest["total_samples"])
        with open(project.export.manifest_path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertEqual(on_disk["status"], "complete")
        self.assertEqual(on_disk["errors"], [])
        self.assertIn("squad", on_disk["channel_gains_db"])
        self.assertIn("recon", on_disk["muted_channels"])
        self.assertGreater(on_disk["chunk_count"], 1)

    def test_resume_matches_one_shot_byte_for_byte(self):
        one_shot = self._project(build_sample_project, "oneshot")
        code, _, _ = self._export(one_shot)
        self.assertEqual(code, 0)
        with open(one_shot.export.output_wav, "rb") as handle:
            one_shot_bytes = handle.read()

        resumed = self._project(build_sample_project, "resume")
        state = {"remaining": 1}

        def stop_after_two_chunks():
            if state["remaining"] <= 0:
                return True
            state["remaining"] -= 1
            return False

        interrupted_code, incomplete, _ = self._export(resumed, stop_after_two_chunks)
        self.assertEqual(interrupted_code, 10)
        self.assertIn(incomplete["status"], ("interrupted", "incomplete"))
        self.assertFalse(os.path.exists(resumed.export.output_wav))
        with open(resumed.export.manifest_path, encoding="utf-8") as handle:
            disk_manifest = json.load(handle)
        self.assertNotEqual(disk_manifest["status"], "complete")
        self.assertTrue(disk_manifest["errors"])
        snapshot_path = os.path.join(resumed.export.work_dir, "snapshot.json")
        with open(snapshot_path, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        for required in ("samples_done", "project_hash", "engine_version", "chunks"):
            self.assertIn(required, snapshot)

        resumed_code, resumed_manifest, _ = self._export(resumed, None)
        self.assertEqual(resumed_code, 0)
        with open(resumed.export.output_wav, "rb") as handle:
            resumed_bytes = handle.read()
        self.assertEqual(hashlib.sha256(one_shot_bytes).hexdigest(),
                         hashlib.sha256(resumed_bytes).hexdigest())
        self.assertEqual(one_shot_bytes, resumed_bytes)
        self.assertEqual(resumed_manifest["status"], "complete")

    def test_hash_mismatch_rejects_resume_and_rebuilds(self):
        project = self._project(build_sample_project, "mismatch")
        state = {"remaining": 0}

        def stop_after_one():
            if state["remaining"] <= 0:
                return True
            state["remaining"] -= 1
            return False

        self._export(project, stop_after_one)
        changed = copy.deepcopy(project.raw)
        changed["events"][0]["tick"] = 101
        changed_project = parse_project(changed)
        changed_project.export.work_dir  # keeps same work directory
        code, manifest, _ = self._export(changed_project, None)
        self.assertEqual(code, 0)
        self.assertTrue(manifest["resume"]["rejected"])
        self.assertTrue(os.path.isfile(changed_project.export.output_wav))

    def test_clipping_stats_reported(self):
        project = self._project(build_clipping_project, "clip")
        code, manifest, _ = self._export(project)
        self.assertEqual(code, 0)
        self.assertGreater(manifest["peaks"]["pre_soft"], 1.0)
        self.assertLessEqual(manifest["peaks"]["post_soft"], 1.0000001)
        self.assertGreater(manifest["clipped_samples"], 0)

    def test_mute_and_block_produce_silence(self):
        data = build_sample_project(
            output_wav=os.path.join(self.tmp, "mute.wav"),
            manifest=os.path.join(self.tmp, "mute.json"),
            work_dir=os.path.join(self.tmp, "mute_work"),
        )
        for channel in data["channels"]:
            channel["muted"] = True
        project = parse_project(data)
        code, manifest, _ = self._export(project)
        self.assertEqual(code, 0)
        with wave.open(project.export.output_wav, "rb") as handle:
            pcm = handle.readframes(handle.getnframes())
        intro_bytes = int(0.31 * project.sample_rate) * 2 * 2
        outro_bytes = int(0.41 * project.sample_rate) * 2 * 2
        middle = pcm[intro_bytes:len(pcm) - outro_bytes]
        self.assertTrue(middle)
        self.assertEqual(set(middle), {0})

    def test_outro_extension_reported_and_does_not_overwrite_voice(self):
        project = self._project(build_sample_project, "extension")
        code, manifest, timeline = self._export(project)
        self.assertEqual(code, 0)
        outro_length = int(round(400.0 * project.sample_rate / 1000.0))
        self.assertEqual(manifest["outro_extension"]["samples"], outro_length)
        self.assertEqual(
            manifest["total_samples"],
            timeline.last_voice_end + outro_length,
        )

    def test_forced_kill_simulation_recovers_from_persisted_prefix(self):
        project = self._project(build_sample_project, "kill")
        timeline = build_timeline(project)
        state = {"remaining": 0}

        def hard_stop_like_flag():
            if state["remaining"] <= 0:
                return True
            state["remaining"] -= 1
            return False

        code, _ = export_project(project, timeline, interrupt_flag=hard_stop_like_flag, logger=lambda _: None)
        self.assertEqual(code, 10)
        snapshot_path = os.path.join(project.export.work_dir, "snapshot.json")
        with open(snapshot_path, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        self.assertEqual(snapshot["project_hash"], project.canonical_hash)
        self.assertEqual(snapshot["engine_version"], manifest_engine())
        code, manifest, _ = self._export(project, None)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["chunk_count"], len(snapshot["chunks"]) +
                         _remaining_chunk_count(project, snapshot))


def manifest_engine():
    from voice_mixer import ENGINE_VERSION
    return ENGINE_VERSION


def _remaining_chunk_count(project, snapshot):
    total = snapshot["total_samples"]
    done = snapshot["samples_done"]
    chunk = project.export.chunk_samples
    return (total - done + chunk - 1) // chunk


if __name__ == "__main__":
    unittest.main()
