import json
import os
import subprocess
import sys
import unittest

from tests._helpers import REPO_ROOT, make_fixture, run_cli


class CliHappyPathTests(unittest.TestCase):
    def test_cli_writes_named_outputs(self):
        input_dir, output_dir = make_fixture()
        result = run_cli(input_dir, output_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.strip())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["wav"], "mix_output.wav")
        self.assertEqual(payload["report"], "beat_alignment_report.json")
        self.assertTrue(
            os.path.isfile(os.path.join(output_dir, "mix_output.wav"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(output_dir, "beat_alignment_report.json")
            )
        )

    def test_missing_argument_is_argparse_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "rhythm_mixer", "--input-dir", "/tmp/x"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output-dir", result.stderr)


if __name__ == "__main__":
    unittest.main()
