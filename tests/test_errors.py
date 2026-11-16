from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from terrain_preview.cli import main
from tests.helpers import small_config, write_config


class ConfigurationErrorTests(unittest.TestCase):
    def _assert_config_error(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            mutate(raw)
            path = write_config(root, raw)
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = main(["--config", str(path)])
            message = stderr.getvalue()
            self.assertEqual(code, 3)
            self.assertIn("ERROR[E_CONFIG]", message)
            self.assertNotIn("Traceback", message)
            self.assertFalse((root / "out" / "preview.png").exists())
            self.assertFalse((root / "out" / "tiles.json").exists())

    def test_zero_camera_distance(self) -> None:
        self._assert_config_error(
            lambda config: config["camera"].__setitem__("distance", 0.0)
        )

    def test_inverted_material_thresholds(self) -> None:
        self._assert_config_error(
            lambda config: config["materials"].__setitem__(
                "elevation_thresholds", [0.5, 0.4, 0.7]
            )
        )

    def test_non_power_of_two_shadow_map(self) -> None:
        self._assert_config_error(
            lambda config: config["light"]["shadow"].__setitem__("resolution", 300)
        )

    def test_out_of_range_heightfield_dimensions(self) -> None:
        self._assert_config_error(
            lambda config: config["terrain"].__setitem__("rows", 1)
        )

    def test_missing_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.json"
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = main(["--config", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("ERROR[E_CONFIG_PARSE]", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_sample_budget_is_a_stable_resource_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            raw["limits"]["max_total_samples"] = 1
            path = write_config(root, raw)
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = main(["--config", str(path)])
            self.assertEqual(code, 7)
            self.assertIn("ERROR[E_RESOURCE]", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertFalse((root / "out" / "preview.png").exists())

    def test_output_write_failure_has_stable_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            raw["output"]["image"] = "out"
            path = write_config(root, raw)
            (root / "out").mkdir()
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = main(["--config", str(path)])
            self.assertEqual(code, 8)
            self.assertIn("ERROR[E_OUTPUT]", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

