import json
import shutil
import tempfile
import unittest
from pathlib import Path

from audiomixer.cli import EXIT_BUSINESS_ERROR, EXIT_OK, main
from tests.fixtures import make_event, make_scene, make_track


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audiomixer_cli_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_config(self, raw):
        path = self.tmp / "scene.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return str(path)

    def test_run_then_no_generate(self):
        raw = make_scene(
            "cli_scene",
            duration=1.0,
            tracks=[make_track("t", [make_event("e", 0.0, 1.0, amplitude=0.4)])],
        )
        config = self._write_config(raw)
        self.assertEqual(main(["run", config]), EXIT_OK)

        # Second run reusing existing assets must also succeed.
        self.assertEqual(main(["run", "--no-generate", config]), EXIT_OK)

        # Removing assets and forcing --no-generate must fail cleanly.
        shutil.rmtree(self.tmp / "assets", ignore_errors=True)
        exit_code = main(["run", "--no-generate", config])
        self.assertEqual(exit_code, EXIT_BUSINESS_ERROR)
        report = json.loads((self.tmp / "out" / "cli_scene.report.json").read_text())
        self.assertEqual(report["error_code"], "E_MISSING_ASSET")
        self.assertFalse((self.tmp / "out" / "cli_scene.wav").exists())

    def test_verify_command_exit_code(self):
        raw = make_scene(
            "verify_scene",
            duration=1.0,
            tracks=[make_track("t", [make_event("e", 0.0, 1.0, amplitude=0.4)])],
        )
        config = self._write_config(raw)
        main(["run", config])
        report_path = str(self.tmp / "out" / "verify_scene.report.json")
        self.assertEqual(main(["verify", report_path]), EXIT_OK)


if __name__ == "__main__":
    unittest.main()
