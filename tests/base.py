"""Shared helpers for the unittest suite."""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import unittest

from replay_exporter.samplegen import generate_project


class ExporterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = tempfile.mkdtemp(prefix="replay_exporter_")
        self.project_dir = os.path.join(self.temp_root, "project")
        self.output_dir = os.path.join(self.temp_root, "output")
        generated = generate_project(
            self.project_dir,
            segment_count=4,
            segment_ms=1000,
            fade_ms=80,
            gaps_ms=[100, 0, 150],
        )
        self.timeline_path = generated["timeline"]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def read_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def read_master_data(self, output_dir: str):
        with open(os.path.join(output_dir, "master.wav"), "rb") as handle:
            blob = handle.read()
        (data_size,) = struct.unpack_from("<I", blob, 40)
        return blob[44 : 44 + data_size], data_size

    def segment_metadata_path(self, index: int) -> str:
        return os.path.join(self.project_dir, f"segment_{index}", "metadata.json")

    def write_segment_metadata(self, index: int, payload: dict) -> None:
        with open(self.segment_metadata_path(index), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    def segment_metadata(self, index: int) -> dict:
        return self.read_json(self.segment_metadata_path(index))
