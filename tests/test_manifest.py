import unittest

from audiopack.config import Config
from audiopack.errors import ErrorCode
from audiopack.manifest import parse_manifest


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_collects_multiple_errors_without_short_circuiting(self) -> None:
        raw = {
            "events": [
                {"id": "A", "type": "kill", "time_ms": 100, "priority": 1,
                 "voice_key": "kill", "langs": ["zh"]},
                {"id": "A", "type": "kill", "time_ms": 50, "priority": 1,
                 "voice_key": "kill", "langs": ["zh"]},
                {"id": "B", "type": "kill", "time_ms": "nope", "priority": 1,
                 "voice_key": "kill", "langs": ["xx"]},
                {"id": "C", "type": "kill", "time_ms": 200, "priority": "high",
                 "voice_key": "kill", "langs": []},
            ]
        }
        manifest = parse_manifest(raw, self.config)
        codes = manifest.issues.codes
        self.assertIn(ErrorCode.E_DUP_ID, codes)
        self.assertIn(ErrorCode.E_BAD_TIME, codes)
        self.assertIn(ErrorCode.E_BAD_FIELD, codes)
        self.assertIn(ErrorCode.E_BAD_LANG, codes)
        # All rows were examined even though earlier rows were invalid.
        self.assertGreaterEqual(len(codes), 5)

    def test_monotonic_violation_is_bad_time(self) -> None:
        raw = {
            "events": [
                {"id": "A", "type": "x", "time_ms": 200, "priority": 1},
                {"id": "B", "type": "x", "time_ms": 199, "priority": 1},
            ]
        }
        manifest = parse_manifest(raw, self.config)
        self.assertIn(ErrorCode.E_BAD_TIME, manifest.issues.codes)

    def test_out_of_range_timestamp(self) -> None:
        raw = {
            "events": [
                {"id": "A", "type": "x", "time_ms": -1, "priority": 1},
                {"id": "B", "type": "x",
                 "time_ms": self.config.max_time_ms + 1, "priority": 1},
            ]
        }
        manifest = parse_manifest(raw, self.config)
        bad_time = [
            issue for issue in manifest.issues.issues
            if issue.code == ErrorCode.E_BAD_TIME
        ]
        self.assertEqual(len(bad_time), 2)

    def test_unknown_language_code(self) -> None:
        raw = {
            "events": [
                {"id": "A", "type": "x", "time_ms": 0, "priority": 1,
                 "langs": ["zh", "klingon"]},
            ]
        }
        manifest = parse_manifest(raw, self.config)
        self.assertIn(ErrorCode.E_BAD_LANG, manifest.issues.codes)

    def test_valid_manifest_defaults_langs_and_hashes(self) -> None:
        raw = {
            "events": [
                {"id": "A", "type": "kill", "time_ms": 0, "priority": 1,
                 "voice_key": "kill"},
                {"id": "B", "type": "mark", "time_ms": 5, "priority": 2},
            ]
        }
        manifest = parse_manifest(raw, self.config)
        self.assertTrue(manifest.valid)
        self.assertEqual(manifest.events[0].langs, self.config.allowed_langs)
        self.assertIsNone(manifest.events[1].voice_key)
        self.assertEqual(len(manifest.manifest_hash), 64)

    def test_integer_sample_position_no_float_drift(self) -> None:
        raw = {
            "events": [
                {"id": "A", "type": "x", "time_ms": 1001, "priority": 1},
            ]
        }
        manifest = parse_manifest(raw, self.config)
        event = manifest.events[0]
        self.assertEqual(
            manifest.start_sample(event, self.config), 1001 * 48
        )
        self.assertIsInstance(
            manifest.start_sample(event, self.config), int
        )


if __name__ == "__main__":
    unittest.main()
