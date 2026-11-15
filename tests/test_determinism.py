from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matgraph.samples import generate_samples, sample_graph

ROOT = Path(__file__).resolve().parents[1]


def _file_hashes(directory: Path):
    import hashlib

    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class DeterminismTests(unittest.TestCase):
    def test_sample_generation_and_render_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_samples = root / "samples_a"
            second_samples = root / "samples_b"
            generate_samples(first_samples)
            generate_samples(second_samples)
            self.assertEqual(_file_hashes(first_samples), _file_hashes(second_samples))
            self.assertEqual(
                _file_hashes(first_samples / "textures"),
                _file_hashes(second_samples / "textures"),
            )

            outputs = []
            for name in ("run_a", "run_b"):
                output = root / name
                process = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "run.py"),
                        "render",
                        "--graph",
                        str(first_samples / "material_graph.json"),
                        "--scene",
                        str(first_samples / "scene.json"),
                        "--out",
                        str(output),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                outputs.append(output)
            self.assertEqual(_file_hashes(outputs[0]), _file_hashes(outputs[1]))

    def test_invalid_graph_emits_structured_error_without_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_samples(root)
            graph = sample_graph()
            del graph["nodes"][1]["inputs"]
            invalid_path = root / "invalid.json"
            invalid_path.write_text(
                json.dumps(graph, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output = root / "must_not_exist"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "render",
                    "--graph",
                    str(invalid_path),
                    "--scene",
                    str(root / "scene.json"),
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 2)
            self.assertFalse(output.exists())
            error = json.loads(process.stderr)
            self.assertFalse(error["ok"])
            self.assertEqual(error["error"]["code"], "E_MISSING_INPUT")
            self.assertEqual(error["error"]["node_index"], 1)
            self.assertEqual(error["error"]["node_id"], "base_tex")


if __name__ == "__main__":
    unittest.main()
