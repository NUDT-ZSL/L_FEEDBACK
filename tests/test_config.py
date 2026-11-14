import json
import tempfile
import unittest
from pathlib import Path

from adaptive_mixer.config import load_config, validate_config
from adaptive_mixer.errors import MixerError


def base_raw():
    return {
        "sample_rate": 48000,
        "frame_rate": 100,
        "total_frames": 300,
        "target_lufs": -23.0,
        "crossfade_ms": 80.0,
        "crossfade_type": "equal_power",
        "tracks": [
            {"name": "strings", "path": "strings.wav"},
            {"name": "percussion", "path": "percussion.wav"},
        ],
        "scenes": [
            {"scene_id": "explore", "start_frame": 0, "end_frame": 120,
             "active_tracks": ["strings"]},
            {"scene_id": "battle", "start_frame": 120, "end_frame": 300,
             "active_tracks": ["strings", "percussion"]},
        ],
    }


class ConfigTests(unittest.TestCase):
    def test_valid_config(self):
        cfg = validate_config(base_raw())
        self.assertEqual(cfg.samples_per_frame, 480)
        self.assertEqual(cfg.total_samples, 300 * 480)
        self.assertEqual(len(cfg.version), 16)

    def test_missing_field(self):
        raw = base_raw()
        del raw["scenes"]
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_CONFIG_MISSING_FIELD")

    def test_frame_rate_mismatch(self):
        raw = base_raw()
        raw["frame_rate"] = 36000
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_SAMPLE_RATE_MISMATCH")

    def test_scene_gap(self):
        raw = base_raw()
        raw["scenes"][1]["start_frame"] = 130
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_SCENE_OVERLAP")

    def test_total_frame_mismatch(self):
        raw = base_raw()
        raw["total_frames"] = 250
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_SCENE_OVERLAP")

    def test_bad_crossfade_type(self):
        raw = base_raw()
        raw["crossfade_type"] = "exponential"
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_CONFIG_VALUE")

    def test_bad_lufs(self):
        raw = base_raw()
        raw["target_lufs"] = 5.0
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_LUFS_TARGET")

    def test_unknown_track_reference(self):
        raw = base_raw()
        raw["scenes"][0]["active_tracks"] = ["brass"]
        with self.assertRaises(MixerError) as ctx:
            validate_config(raw)
        self.assertEqual(ctx.exception.code, "E_CONFIG_MISSING_FIELD")

    def test_parse_json_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(MixerError) as ctx:
                load_config(path)
            self.assertEqual(ctx.exception.code, "E_CONFIG_PARSE")

    def test_fingerprint_stable_and_sensitive(self):
        cfg1 = validate_config(base_raw())
        raw2 = base_raw()
        raw2["crossfade_ms"] = 100.0
        cfg2 = validate_config(raw2)
        self.assertNotEqual(cfg1.version, cfg2.version)
        self.assertEqual(validate_config(base_raw()).version, cfg1.version)


if __name__ == "__main__":
    unittest.main()
