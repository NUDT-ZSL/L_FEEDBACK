from __future__ import annotations

import hashlib
import json
import os

from replay_exporter import export
from replay_exporter.engine import MASTER_FILE
from replay_exporter.snapshot import (
    SNAPSHOT_VERSION,
    load_snapshot,
    write_snapshot_atomic,
)

from .base import ExporterTestCase


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class ResumeTests(ExporterTestCase):
    def test_interrupted_export_resumes_and_matches_full_render(self):
        interrupted_dir = os.path.join(self.temp_root, "interrupted")
        full_dir = os.path.join(self.temp_root, "full")

        first = export(
            self.timeline_path,
            interrupted_dir,
            checkpoint_every=2,
            stop_after_blocks=6,
        )
        self.assertTrue(first["interrupted"])
        checkpoint = self.read_json(
            os.path.join(interrupted_dir, "render_checkpoint.json")
        )
        self.assertEqual(len(checkpoint["completed_blocks"]), 6)

        resumed = export(self.timeline_path, interrupted_dir, checkpoint_every=2)
        self.assertFalse(resumed["interrupted"])
        self.assertEqual(resumed["resumed_blocks"], 6)
        self.assertEqual(resumed["rebuilt_blocks"], [])

        export(self.timeline_path, full_dir)
        self.assertEqual(
            file_digest(os.path.join(interrupted_dir, MASTER_FILE)),
            file_digest(os.path.join(full_dir, MASTER_FILE)),
        )

    def test_half_written_block_is_detected_and_recomputed(self):
        output_dir = os.path.join(self.temp_root, "partial")
        export(
            self.timeline_path,
            output_dir,
            checkpoint_every=1,
            stop_after_blocks=4,
        )
        master = os.path.join(output_dir, MASTER_FILE)
        checkpoint_path = os.path.join(output_dir, "render_checkpoint.json")
        checkpoint = self.read_json(checkpoint_path)

        # Corrupt block 3 inside the master but leave the checkpoint untouched,
        # simulating a crash after the checkpoint was written but during a later
        # overwrite (or a torn write). Target the middle of the block so the
        # torn bytes cannot coincide with a sine-wave zero crossing.
        snapshot = load_snapshot(output_dir)
        block_frames = snapshot.plan["block_frames"]
        corrupt_offset = 44 + (3 * block_frames + block_frames // 2) * 4
        with open(master, "r+b") as handle:
            handle.seek(corrupt_offset)
            handle.write(b"\xff" * 128)

        resumed = export(self.timeline_path, output_dir)
        rebuilt_indexes = {item["block_index"] for item in resumed["rebuilt_blocks"]}
        self.assertIn(3, rebuilt_indexes)

        full_dir = os.path.join(self.temp_root, "full_ref")
        export(self.timeline_path, full_dir)
        self.assertEqual(
            file_digest(master),
            file_digest(os.path.join(full_dir, MASTER_FILE)),
        )

    def test_truncated_master_is_rebuilt_from_checkpoint(self):
        output_dir = os.path.join(self.temp_root, "truncated")
        export(
            self.timeline_path,
            output_dir,
            checkpoint_every=1,
            stop_after_blocks=5,
        )
        master = os.path.join(output_dir, MASTER_FILE)
        snapshot = load_snapshot(output_dir)
        block_frames = snapshot.plan["block_frames"]
        # Truncate in the middle of block 4 (not at its tail, which may be a
        # silent gap and would hash identically after zero-fill on read).
        target_size = 44 + (4 * block_frames + block_frames // 2) * 4
        self.assertLess(target_size, os.path.getsize(master))
        with open(master, "r+b") as handle:
            handle.truncate(target_size)
        resumed = export(self.timeline_path, output_dir)
        self.assertTrue(any(
            item["reason"] == "half_written_or_hash_mismatch"
            for item in resumed["rebuilt_blocks"]
        ))

    def test_old_snapshot_version_deprecates_every_block(self):
        output_dir = os.path.join(self.temp_root, "versioned")
        export(
            self.timeline_path,
            output_dir,
            checkpoint_every=1,
            stop_after_blocks=3,
        )
        snapshot = load_snapshot(output_dir)
        old_block_count = len(snapshot.blocks)
        snapshot.version = "0.9.0-legacy"
        write_snapshot_atomic(output_dir, snapshot)

        # Remove the checkpoint so the legacy snapshot is reconciled purely at
        # planning time: every legacy block must land in the deprecation ledger
        # and none of the old offsets may be marked reusable.
        os.remove(os.path.join(output_dir, "render_checkpoint.json"))
        result = export(self.timeline_path, output_dir)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["deprecated_blocks"]), old_block_count)
        self.assertTrue(
            all(item["reason"] == "version_mismatch" for item in result["deprecated_blocks"])
        )
        refreshed = load_snapshot(output_dir)
        self.assertEqual(refreshed.version, SNAPSHOT_VERSION)

    def test_checkpoint_snapshot_version_conflict_is_rejected(self):
        from replay_exporter import ExporterError
        from replay_exporter.errors import ErrorCode

        output_dir = os.path.join(self.temp_root, "conflict")
        export(
            self.timeline_path,
            output_dir,
            checkpoint_every=1,
            stop_after_blocks=3,
        )
        checkpoint_path = os.path.join(output_dir, "render_checkpoint.json")
        checkpoint = self.read_json(checkpoint_path)
        checkpoint["version"] = "0.9.0-legacy"
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle)

        with self.assertRaises(ExporterError) as caught:
            export(self.timeline_path, output_dir)
        self.assertEqual(caught.exception.code, ErrorCode.SNAPSHOT_VERSION_MISMATCH)
        # A structured error report must be persisted without touching the
        # previously valid checkpoint contents.
        error_report = self.read_json(os.path.join(output_dir, "error_report.json"))
        self.assertEqual(error_report["code"], ErrorCode.SNAPSHOT_VERSION_MISMATCH)
        persisted = self.read_json(checkpoint_path)
        self.assertEqual(persisted["completed_blocks"], checkpoint["completed_blocks"])

    def test_atomic_checkpoint_never_leaves_temp_file_on_success(self):
        output_dir = os.path.join(self.temp_root, "atomic")
        export(self.timeline_path, output_dir, checkpoint_every=1)
        leftovers = [
            name for name in os.listdir(output_dir) if name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])
