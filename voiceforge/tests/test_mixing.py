import array
import json
import os
import shutil
import tempfile
import unittest

from voiceforge import OUTPUT_SAMPLE_RATE
from voiceforge.audio import (FLOAT_SCALE, floats_to_int16, read_wav,
                              write_wav)
from voiceforge.engine import canonical_manifest_text, synthesize
from voiceforge.outputs import write_outputs
from voiceforge.resampling import needs_resampling, render_segment


def _wav(path, rate, normalized):
    samples = array.array("h", (int(round(v * FLOAT_SCALE)) for v in normalized))
    write_wav(path, rate, samples)


def _manifest(segments, intervals, missing=None):
    players = {}
    for shard in segments:
        players.setdefault(shard["player_id"], []).append(shard)
    return {
        "schema_version": 1,
        "seed": 1,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "drift_tolerance_ms": 20.0,
        "gate_threshold": 0.1,
        "players": [{"player_id": pid, "segments": items}
                    for pid, items in sorted(players.items())],
        "intervals": intervals,
        "missing_players": missing or [],
    }


class ResamplingTests(unittest.TestCase):
    def test_drift_threshold_and_linear_resample(self):
        self.assertFalse(needs_resampling(100.0, 115.0))
        self.assertTrue(needs_resampling(100.0, 125.0))
        source = [0.0, 1.0]
        start, rendered = render_segment(source, 1000, 0.0, 100.0, 1000)
        self.assertEqual(start, 0)
        self.assertEqual(len(rendered), 100)
        self.assertAlmostEqual(rendered[0], 0.0)
        self.assertAlmostEqual(rendered[-1], 1.0)


class MixingTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vf-mix-")
        self.input_dir = os.path.join(self.root, "in")
        self.output_dir = os.path.join(self.root, "out")
        os.makedirs(self.input_dir)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _synthesize(self, manifest):
        text = canonical_manifest_text(manifest)
        return synthesize(manifest, text, self.input_dir, self.output_dir)

    def test_gate_gain_and_saturation(self):
        rate = OUTPUT_SAMPLE_RATE
        count = rate // 10
        loud = [0.5] * count
        quiet = [0.05] * count
        _wav(os.path.join(self.input_dir, "a.wav"), rate, loud)
        _wav(os.path.join(self.input_dir, "b.wav"), rate, quiet)
        segments = [
            {"segment_id": "A", "player_id": "P0", "file": "a.wav",
             "sample_rate": rate, "record_offset_ms": 0.0,
             "declared_duration_ms": 100.0, "actual_duration_ms": 100.0,
             "drift_ms": 0.0, "dropped_first_packets_ms": 0,
             "speech_start_ms": 0.0, "speech_end_ms": 100.0,
             "threshold": 0.1, "gain": 2.0, "resample_required": False},
            {"segment_id": "B", "player_id": "P1", "file": "b.wav",
             "sample_rate": rate, "record_offset_ms": 0.0,
             "declared_duration_ms": 100.0, "actual_duration_ms": 100.0,
             "drift_ms": 0.0, "dropped_first_packets_ms": 0,
             "speech_start_ms": 0.0, "speech_end_ms": 100.0,
             "threshold": 0.1, "gain": 1.0, "resample_required": False},
        ]
        intervals = [
            {"interval_id": "A", "player_id": "P0", "segment_id": "A",
             "start_ms": 0.0, "end_ms": 100.0,
             "threshold": 0.1, "gain": 2.0},
            {"interval_id": "B", "player_id": "P1", "segment_id": "B",
             "start_ms": 0.0, "end_ms": 100.0,
             "threshold": 0.1, "gain": 1.0},
        ]
        manifest = _manifest(segments, intervals)
        result = self._synthesize(manifest)
        self.assertFalse(result["interrupted"])
        write_outputs(result, manifest, self.input_dir, self.output_dir)
        # loud 0.5*2.0 survives gating; quiet 0.05 is gated to zero.
        self.assertEqual(result["loudness_map"]["B"]["peak"], 0.0)
        self.assertGreater(result["loudness_map"]["A"]["new_clipped_samples"], 0)
        # 0.5 * 2.0 = 1.0 saturates to the 16-bit maximum.
        wav = read_wav(os.path.join(self.output_dir, "voice_track.wav"))
        self.assertTrue(all(sample == 32767 for sample in wav.samples[: count]))
        self.assertEqual(sum(result["clip_mask"]),
                         result["loudness_map"]["A"]["new_clipped_samples"])

    def test_missing_source_does_not_crash(self):
        intervals = [
            {"interval_id": "G", "player_id": "PX", "segment_id": "G",
             "start_ms": 0.0, "end_ms": 50.0,
             "threshold": 0.1, "gain": 1.0},
        ]
        manifest = _manifest([], intervals, missing=["PX"])
        result = self._synthesize(manifest)
        self.assertFalse(result["interrupted"])
        self.assertEqual(result["loudness_map"]["G"]["peak"], 0.0)
        self.assertIn("PX", result["missing_players"])


if __name__ == "__main__":
    unittest.main()

