from __future__ import annotations

import json
import math
import os
import struct

from replay_exporter.audioio import write_mono_wav
from replay_exporter.dsp import db_to_linear
from replay_exporter.exporter import export

from .base import ExporterTestCase


class LimiterTests(ExporterTestCase):
    def test_hard_limiter_ceiling_is_enforced_without_overflow(self):
        # Replace segment 0 sources with full-scale tones so the mix would
        # otherwise clip; the -1 dBFS hard limiter must hold every sample.
        from replay_exporter.dsp import ms_to_frames
        from replay_exporter.audioio import SAMPLE_RATE

        segment_dir = os.path.join(self.project_dir, "segment_0")
        frames = ms_to_frames(1000, SAMPLE_RATE)
        loud = [32767 if index % 2 == 0 else -32768 for index in range(frames)]
        write_mono_wav(os.path.join(segment_dir, "voice.wav"), loud)
        write_mono_wav(os.path.join(segment_dir, "ambience.wav"), loud)

        # Relax the peak target so normalization does not simply attenuate.
        metadata = self.segment_metadata(0)
        metadata["target_peak_dbfs"] = 0.0
        self.write_segment_metadata(0, metadata)
        with open(self.timeline_path, "r", encoding="utf-8") as handle:
            timeline = json.load(handle)
        timeline["loudness_target_dbfs"] = 0.0
        with open(self.timeline_path, "w", encoding="utf-8") as handle:
            json.dump(timeline, handle)

        output_dir = os.path.join(self.temp_root, "limited")
        export(self.timeline_path, output_dir)
        with open(os.path.join(output_dir, "master.wav"), "rb") as handle:
            blob = handle.read()
        (data_size,) = struct.unpack_from("<I", blob, 40)
        samples = struct.unpack_from(f"<{data_size // 2}h", blob, 44)
        ceiling = math.floor(db_to_linear(-1.0) * 32767)
        self.assertLessEqual(max(abs(value) for value in samples), ceiling)
        self.assertGreater(max(abs(value) for value in samples), 0)

        gain_report = self.read_json(os.path.join(output_dir, "gain_report.json"))
        self.assertEqual(gain_report["limiter_ceiling_dbfs"], -1.0)

    def test_limiter_unit_helper(self):
        from replay_exporter.dsp import limit_int16

        ceiling = db_to_linear(-1.0)
        self.assertEqual(limit_int16(2.0, ceiling), math.floor(ceiling * 32767))
        self.assertEqual(limit_int16(-2.0, ceiling), -math.floor(ceiling * 32767))
        self.assertEqual(limit_int16(0.0, ceiling), 0)
        self.assertEqual(limit_int16(0.5, 1.0), 16384)
