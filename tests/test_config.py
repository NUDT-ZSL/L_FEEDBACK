from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cinemix.config import load_config
from cinemix.errors import CinemixError, Diagnostics


def _project() -> dict:
    return {
        "project": {
            "sample_rate": 48000,
            "output": "out.wav",
            "target_lufs": -23.0,
            "true_peak_dbtp": -1.0,
        },
        "tracks": [
            {
                "id": "a",
                "path": "a.wav",
                "gain_db": 0.0,
                "pan": 0.0,
            }
        ],
    }


class ConfigTests(unittest.TestCase):
    def _write(self, directory: Path, payload: object) -> Path:
        path = directory / "project.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(CinemixError) as raised:
                load_config(path, Diagnostics())
            self.assertEqual(raised.exception.code, "E_CONFIG_JSON")

    def test_invalid_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = _project()
            payload["tracks"][0]["gain_db"] = "loud"
            path = self._write(Path(temporary), payload)
            with self.assertRaises(CinemixError) as raised:
                load_config(path, Diagnostics())
            self.assertEqual(raised.exception.code, "E_CONFIG_FIELD")

    def test_gain_clamped_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = _project()
            payload["tracks"][0]["gain_db"] = 99.0
            path = self._write(Path(temporary), payload)
            diagnostics = Diagnostics()
            config = load_config(path, diagnostics)
            self.assertEqual(config.tracks[0].gain_db, 24.0)
            self.assertIn(
                "W_GAIN_CLAMPED",
                [issue.code for issue in diagnostics.issues],
            )

    def test_crossfade_overlap_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = _project()
            payload["tracks"].append(
                {"id": "b", "path": "b.wav", "gain_db": 0.0}
            )
            payload["tracks"].append(
                {"id": "c", "path": "c.wav", "gain_db": 0.0}
            )
            payload["crossfades"] = [
                {"from": "a", "to": "b", "start": 1.0, "end": 3.0},
                {"from": "a", "to": "c", "start": 2.0, "end": 4.0},
            ]
            path = self._write(Path(temporary), payload)
            with self.assertRaises(CinemixError) as raised:
                load_config(path, Diagnostics())
            self.assertEqual(
                raised.exception.code, "E_CROSSFADE_CONFLICT"
            )

    def test_invalid_curve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = _project()
            payload["tracks"].append(
                {"id": "b", "path": "b.wav", "gain_db": 0.0}
            )
            payload["crossfades"] = [
                {
                    "from": "a",
                    "to": "b",
                    "start": 1.0,
                    "end": 2.0,
                    "curve": "mystery",
                }
            ]
            path = self._write(Path(temporary), payload)
            with self.assertRaises(CinemixError) as raised:
                load_config(path, Diagnostics())
            self.assertEqual(raised.exception.code, "E_CONFIG_FIELD")


if __name__ == "__main__":
    unittest.main()

