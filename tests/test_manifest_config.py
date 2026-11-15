import json
import os
import tempfile
import unittest

from audiopipeline.config import Config
from audiopipeline.errors import ERROR_CATALOG, PipelineError
from audiopipeline.manifest import load_manifest, ms_to_samples, samples_to_ms


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.audio = os.path.join(self.root, "a.wav")
        open(self.audio, "wb").write(b"RIFFxxxx")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, payload):
        path = os.path.join(self.root, "manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def test_valid_manifest(self):
        path = self._write(
            {
                "total_duration_ms": 1000,
                "events": [
                    {"id": "e1", "path": "a.wav", "start_ms": 0}
                ],
            }
        )
        events, total, _ = load_manifest(path)
        self.assertEqual(total, 1000.0)
        self.assertEqual(events[0].event_id, "e1")
        self.assertEqual(events[0].errors, [])

    def test_bad_json_is_fatal(self):
        path = os.path.join(self.root, "broken.json")
        with open(path, "w") as fh:
            fh.write("{not json")
        with self.assertRaises(PipelineError) as ctx:
            load_manifest(path)
        self.assertEqual(ctx.exception.code, "E_MANIFEST_PARSE")

    def test_invalid_total_duration_is_fatal(self):
        path = self._write({"total_duration_ms": -1, "events": []})
        with self.assertRaises(PipelineError) as ctx:
            load_manifest(path)
        self.assertEqual(ctx.exception.code, "E_DURATION_INVALID")

    def test_missing_events_field_fatal(self):
        path = self._write({"total_duration_ms": 1000})
        with self.assertRaises(PipelineError) as ctx:
            load_manifest(path)
        self.assertEqual(ctx.exception.code, "E_MANIFEST_FIELD")

    def test_per_event_errors_do_not_abort(self):
        path = self._write(
            {
                "total_duration_ms": 1000,
                "events": [
                    {"start_ms": 0, "path": "a.wav"},          # missing id
                    {"id": "e2", "path": "missing.wav",
                     "start_ms": 0},                            # missing file
                    {"id": "e3", "path": "a.wav",
                     "start_ms": -5},                           # bad time
                    {"id": "e4", "path": "a.mp3",
                     "start_ms": 0},                            # bad format
                ],
            }
        )
        events, _, _ = load_manifest(path)
        self.assertIn("E_FIELD_MISSING", events[0].errors)
        self.assertIn("E_FILE_NOT_FOUND", events[1].errors)
        self.assertIn("E_TIME_INVALID", events[2].errors)
        self.assertIn("E_FORMAT_UNSUPPORTED", events[3].errors)

    def test_sample_roundtrip_sub_millisecond(self):
        for ms in (0.0, 1.0, 123.456, 1000.0):
            samples = ms_to_samples(ms)
            back = samples_to_ms(samples)
            self.assertLessEqual(abs(back - ms), 1.0 / 48.0 + 1e-9)


class ConfigTests(unittest.TestCase):
    def test_defaults_match_spec(self):
        cfg = Config()
        self.assertEqual(cfg.target_integrated_lufs, -16.0)
        self.assertEqual(cfg.true_peak_limit_dbtp, -1.0)
        self.assertEqual(cfg.loudness_block_ms, 400.0)
        self.assertEqual(cfg.loudness_overlap, 0.75)
        self.assertEqual(cfg.crossfade_ms, 10.0)

    def test_out_of_range_rejected(self):
        cfg = Config(crossfade_ms=9999.0)
        with self.assertRaises(PipelineError) as ctx:
            cfg.validate()
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

    def test_unknown_config_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w") as fh:
                json.dump({"crossfade_ms": 5, "bogus": 1}, fh)
            with self.assertRaises(PipelineError) as ctx:
                Config.from_file(path)
            self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

    def test_all_error_codes_use_prefix(self):
        for code in ERROR_CATALOG:
            self.assertTrue(code.startswith("E_"))


if __name__ == "__main__":
    unittest.main()
