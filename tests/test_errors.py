from __future__ import annotations

import json
import os

from replay_exporter import ErrorCode, ExporterError, export
from replay_exporter.planner import build_plan, load_segment_inputs, load_sources

from .base import ExporterTestCase


class ErrorChainTests(ExporterTestCase):
    def test_error_report_json_is_written_and_checkpoint_survives(self):
        good_dir = os.path.join(self.temp_root, "good")
        export(self.timeline_path, good_dir, checkpoint_every=1, stop_after_blocks=3)
        checkpoint_before = self.read_json(
            os.path.join(good_dir, "render_checkpoint.json")
        )

        metadata = self.segment_metadata(2)
        del metadata["dialogue_end_ms"]
        self.write_segment_metadata(2, metadata)

        with self.assertRaises(ExporterError) as caught:
            export(self.timeline_path, good_dir)
        self.assertEqual(caught.exception.code, ErrorCode.MISSING_DIALOGUE_RANGE)
        self.assertEqual(caught.exception.segment_index, 2)

        error_report = self.read_json(os.path.join(good_dir, "error_report.json"))
        self.assertEqual(error_report["ok"], False)
        self.assertEqual(error_report["code"], ErrorCode.MISSING_DIALOGUE_RANGE)
        checkpoint_after = self.read_json(
            os.path.join(good_dir, "render_checkpoint.json")
        )
        self.assertEqual(
            checkpoint_before["completed_blocks"],
            checkpoint_after["completed_blocks"],
        )

    def test_distinct_codes_for_each_failure_class(self):
        # Missing dialogue field.
        metadata = self.segment_metadata(0)
        del metadata["dialogue_start_ms"]
        self.write_segment_metadata(0, metadata)
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.MISSING_DIALOGUE_RANGE)

    def test_missing_metadata_field_code(self):
        metadata = self.segment_metadata(1)
        del metadata["target_peak_dbfs"]
        self.write_segment_metadata(1, metadata)
        with self.assertRaises(ExporterError) as caught:
            load_segment_inputs(self.timeline_path)
        self.assertEqual(caught.exception.code, ErrorCode.MISSING_METADATA_FIELD)

    def test_unwritable_output_directory(self):
        output_dir = os.path.join(self.temp_root, "readonly")
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(output_dir, 0o500)
        try:
            with self.assertRaises((ExporterError, PermissionError)):
                export(self.timeline_path, output_dir)
        finally:
            os.chmod(output_dir, 0o755)

    def test_segment_order_conflict_code(self):
        # Exercise the real ordering guard inside build_plan by reusing the
        # validated inputs but mutating one segment's gap to force an overlap.
        inputs, timeline = load_segment_inputs(self.timeline_path)
        # A negative gap is normally rejected while parsing; bypass the parser
        # by editing the already-validated in-memory inputs directly.
        inputs[0].gap_after_ms = -200.0
        with self.assertRaises(ExporterError) as caught:
            build_plan(inputs, timeline)
        self.assertEqual(caught.exception.code, ErrorCode.SEGMENT_ORDER_CONFLICT)
        self.assertEqual(caught.exception.segment_index, 1)
