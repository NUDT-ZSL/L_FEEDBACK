import os
import unittest
from dataclasses import replace

import numpy as np

from audiopack.assets import AssetStore
from audiopack.manifest import load_manifest
from audiopack.renderer import build_language_track, total_frames
from audiopack.scheduler import schedule_all
from audiopack.snapshots import (
    block_asset_hash,
    evaluate_reuse,
    load_snapshot,
    load_track,
    revalidate_offsets,
    save_snapshot,
)
from tests.helpers import make_workspace, cleanup


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        (self.root, self.assets, self.manifest_path,
         self.work, self.out, self.config) = make_workspace()
        self.manifest = load_manifest(self.manifest_path, self.config)
        store = AssetStore(self.config)
        pairs = {
            (event.voice_key, lang)
            for event in self.manifest.events if event.voice_key
            for lang in event.langs
        }
        paths = store.required_paths(self.assets, pairs)
        loaded, failures = store.load_many(list(paths))
        self.assertEqual(failures, [])
        clips_by_lang = {}
        for path, (voice_key, lang) in paths.items():
            clips_by_lang.setdefault(lang, {})[voice_key] = loaded[path]
        self.schedules = schedule_all(self.manifest, clips_by_lang, self.config)
        self.clips_by_lang = clips_by_lang

    def tearDown(self) -> None:
        cleanup(self.root)

    def _build_block(self, lang: str):
        schedule = self.schedules[lang]
        clips = {
            placement.event_id: self.clips_by_lang[lang][placement.voice_key]
            for placement in schedule.placements
        }
        track, norm = build_language_track(schedule, clips, self.config)
        block_dir = os.path.join(self.work, "blocks", lang)
        clip_hashes = {
            clip.source_name: clip.content_hash
            for clip in self.clips_by_lang[lang].values()
        }
        asset_hash = block_asset_hash(schedule, clip_hashes)
        from audiopack.cue_index import build_index_entries
        entries = build_index_entries(schedule, self.config)
        wav_info = {"frames": track.shape[0], "sample_rate": 48000,
                    "channels": 1, "subtype": "PCM_16", "duration_ms": 0.0,
                    "path": f"track_{lang}.wav"}
        snapshot = save_snapshot(
            block_dir, self.config, self.manifest.manifest_hash,
            asset_hash, schedule, track, norm, wav_info, entries,
        )
        return block_dir, schedule, track, snapshot

    def test_snapshot_round_trip_and_reuse(self) -> None:
        block_dir, schedule, track, snapshot = self._build_block("zh")
        decision = evaluate_reuse(
            block_dir, self.config, self.manifest.manifest_hash,
            snapshot.asset_hash, schedule,
        )
        self.assertTrue(decision.reusable, decision.reason)
        reloaded = load_track(block_dir, load_snapshot(block_dir))
        np.testing.assert_array_equal(reloaded, track)

    def test_asset_change_invalidates_only_affected_block(self) -> None:
        zh_dir, zh_schedule, _, zh_snapshot = self._build_block("zh")
        en_dir, en_schedule, _, en_snapshot = self._build_block("en")
        # Pretend the zh asset hash changed; en inputs stay identical.
        zh_decision = evaluate_reuse(
            zh_dir, self.config, self.manifest.manifest_hash,
            "different-asset-hash", zh_schedule,
        )
        en_decision = evaluate_reuse(
            en_dir, self.config, self.manifest.manifest_hash,
            en_snapshot.asset_hash, en_schedule,
        )
        self.assertFalse(zh_decision.reusable)
        self.assertEqual(zh_decision.reason, "asset_changed")
        self.assertTrue(en_decision.reusable)

    def test_version_mismatch_forces_recompute(self) -> None:
        block_dir, schedule, _, snapshot = self._build_block("ja")
        new_config = replace(self.config, snapshot_version=self.config.snapshot_version + 1)
        decision = evaluate_reuse(
            block_dir, new_config, self.manifest.manifest_hash,
            snapshot.asset_hash, schedule,
        )
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, "version_mismatch")

    def test_offset_revalidation_detects_stale_offset(self) -> None:
        block_dir, schedule, _, snapshot = self._build_block("zh")
        good = revalidate_offsets(
            snapshot, schedule, total_frames(schedule)
        )
        self.assertEqual(good, [])

        # Tamper a stored offset; revalidation against the live schedule fails.
        tampered = load_snapshot(block_dir)
        tampered.index_entries[0]["sample_offset"] += 999
        bad = revalidate_offsets(
            tampered, schedule, total_frames(schedule)
        )
        self.assertTrue(any(v["code"] == "E_OFFSET_REVALIDATION" for v in bad))

    def test_track_tampering_breaks_content_hash(self) -> None:
        block_dir, schedule, _, snapshot = self._build_block("zh")
        track_path = os.path.join(block_dir, snapshot.track_file)
        with open(track_path, "r+b") as handle:
            handle.seek(100)
            handle.write(b"\x01\x02\x03\x04")
        decision = evaluate_reuse(
            block_dir, self.config, self.manifest.manifest_hash,
            snapshot.asset_hash, schedule,
        )
        self.assertFalse(decision.reusable)
        self.assertEqual(decision.reason, "track_hash_mismatch")


if __name__ == "__main__":
    unittest.main()
