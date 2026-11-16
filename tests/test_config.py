import json
import tempfile
import unittest
from pathlib import Path

from lipsync_tool.config import DEFAULT_CONFIG, load_config


class ConfigTests(unittest.TestCase):
    def test_missing_file_uses_defaults(self):
        config, notes = load_config(Path("/nonexistent/lipsync-config.json"))
        self.assertEqual(config.align_tolerance_ms, DEFAULT_CONFIG["align_tolerance_ms"])
        self.assertTrue(any("not found" in note for note in notes))

    def test_invalid_fields_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "stretch_min": 9.0,
                        "align_tolerance_ms": "banana",
                        "overlap_penalty_weight": -3.0,
                        "simulate_mux_failure": True,
                    }
                ),
                encoding="utf-8",
            )
            config, notes = load_config(path)
            self.assertEqual(config.stretch_min, DEFAULT_CONFIG["stretch_min"])
            self.assertEqual(config.align_tolerance_ms, DEFAULT_CONFIG["align_tolerance_ms"])
            self.assertEqual(config.overlap_penalty_weight, DEFAULT_CONFIG["overlap_penalty_weight"])
            self.assertTrue(config.simulate_mux_failure)
            self.assertGreaterEqual(len(notes), 3)

    def test_inverted_bounds_reset_both(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"stretch_min": 1.2, "stretch_max": 0.8}), encoding="utf-8")
            config, notes = load_config(path)
            self.assertEqual(config.stretch_min, DEFAULT_CONFIG["stretch_min"])
            self.assertEqual(config.stretch_max, DEFAULT_CONFIG["stretch_max"])
            self.assertTrue(any("stretch_min" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
