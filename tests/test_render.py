from __future__ import annotations

import hashlib
import json
import os
import struct

from replay_exporter import export, validate
from replay_exporter.dsp import db_to_linear
from replay_exporter.engine import (
    calculate_gains,
    parse_and_plan,
    render_master,
)
from replay_exporter.snapshot import SNAPSHOT_VERSION, load_snapshot

from .base import ExporterTestCase


class RenderTests(ExporterTestCase):
    def test_full_export_is_deterministic(self):
        first = os.path.join(self.temp_root, "out_a")
        second = os.path.join(self.temp_root, "out_b")
        result_a = export(self.timeline_path, first)
        result_b = export(self.timeline_path, second)
        self.assertTrue(result_a["ok"] and result_b["ok"])

        def digest(path: str) -> str:
            with open(path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()

        self.assertEqual(
            digest(os.path.join(first, "master.wav")),
            digest(os.path.join(second, "master.wav")),
        )

    def test_block_render_matches_single_render_byte_for_byte(self):
        plan, sources = parse_and_plan(self.timeline_path)
        gains, previous_gains, _ = calculate_gains(plan, sources)
        from replay_exporter.dsp import render_block

        concatenated = bytearray()
        for index in range(plan.block_count()):
            concatenated += render_block(plan, index, sources, gains, previous_gains)

        output_dir = os.path.join(self.temp_root, "out_once")
        export(self.timeline_path, output_dir)
        pcm, data_size = self.read_master_data(output_dir)
        self.assertEqual(len(concatenated), data_size)
        self.assertEqual(bytes(concatenated), pcm)

    def test_master_is_48k_stereo_16bit_and_limited(self):
        output_dir = os.path.join(self.temp_root, "out_fmt")
        export(self.timeline_path, output_dir)
        with open(os.path.join(output_dir, "master.wav"), "rb") as handle:
            blob = handle.read()
        self.assertEqual(blob[0:4], b"RIFF")
        (channels, sample_rate) = struct.unpack_from("<HI", blob, 22)
        (bits,) = struct.unpack_from("<H", blob, 34)
        self.assertEqual(channels, 2)
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(bits, 16)
        (data_size,) = struct.unpack_from("<I", blob, 40)
        samples = struct.unpack_from(f"<{data_size // 2}h", blob, 44)
        ceiling = int(db_to_linear(-1.0) * 32767)
        self.assertLessEqual(max(abs(value) for value in samples), ceiling)

    def test_gain_report_is_traceable_per_segment(self):
        output_dir = os.path.join(self.temp_root, "out_gain")
        export(self.timeline_path, output_dir)
        report = self.read_json(os.path.join(output_dir, "gain_report.json"))
        self.assertEqual(len(report["segments"]), 4)
        for row in report["segments"]:
            for key in (
                "measured_peak_dbfs",
                "measured_rms_dbfs",
                "loudness_gain_db",
                "peak_gain_db",
                "applied_gain_db",
                "limiter_ceiling_dbfs",
            ):
                self.assertIn(key, row)
            self.assertEqual(
                row["applied_gain_db"],
                min(row["loudness_gain_db"], row["peak_gain_db"]),
            )

    def test_snapshot_contains_version_and_block_hashes(self):
        output_dir = os.path.join(self.temp_root, "out_snap")
        export(self.timeline_path, output_dir)
        snapshot = load_snapshot(output_dir)
        self.assertEqual(snapshot.version, SNAPSHOT_VERSION)
        self.assertGreater(len(snapshot.blocks), 0)
        for block in snapshot.blocks:
            self.assertEqual(len(block["input_hash"]), 64)
        checkpoint = self.read_json(os.path.join(output_dir, "render_checkpoint.json"))
        self.assertEqual(len(checkpoint["block_output_hashes"]), len(snapshot.blocks))

    def test_validation_report_passes(self):
        output_dir = os.path.join(self.temp_root, "out_validate")
        export(self.timeline_path, output_dir)
        report = validate(output_dir)
        self.assertTrue(report["passed"], msg=json.dumps(report, indent=2))
        self.assertEqual(len(report["segments"]), 4)
        self.assertEqual(len(report["crossfades"]), 3)
        for segment in report["segments"]:
            self.assertTrue(segment["passed"])
        for crossfade in report["crossfades"]:
            self.assertTrue(crossfade["passed"])
