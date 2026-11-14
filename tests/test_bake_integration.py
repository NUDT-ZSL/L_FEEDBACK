import json
import os
import tempfile
import unittest

import numpy as np

from terrain_baker.baker import bake
from terrain_baker.cli import main
from terrain_baker.errors import MemoryBudgetExceeded, ParameterError
from terrain_baker.heightfield import HeightfieldSpec
from terrain_baker.lod import INDEX_DTYPE, VERTEX_DTYPE
from terrain_baker.pngutil import read_png_chunks
from terrain_baker.verifier import verify_output


SPEC = HeightfieldSpec(seed=11, resolution=257, height_scale=8.0, frequency=2.0)


class BakeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "bake")

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_bake_verifies_and_consistency_passes(self):
        result = bake(SPEC, self.out, levels=[0, 1, 2], tile=128,
                      preview_width=64, preview_height=64, skip_preview=True)
        self.assertTrue(result["verification"]["ok"])
        manifest = result["manifest"]
        for report in manifest["consistency_checks"]:
            self.assertLessEqual(report["max_angle_deg"], 5.0)
            self.assertTrue(report["passed"])
        # Normal PNG carries sRGB and files are referenced relatively.
        normal_path = os.path.join(self.out, manifest["normal_map"]["path"])
        self.assertIn(b"sRGB", read_png_chunks(normal_path))
        for relative in manifest["files"]:
            self.assertTrue(os.path.exists(os.path.join(self.out, relative)))
            self.assertFalse(os.path.isabs(relative))

    def test_metadata_offsets_match_binary_files(self):
        bake(SPEC, self.out, levels=[0, 1], tile=128, skip_preview=True)
        report = verify_output(self.out)
        self.assertTrue(report["ok"], report["errors"])
        manifest = json.load(open(os.path.join(self.out, "metadata.json")))
        for level in manifest["lod_levels"]:
            binary = os.path.join(self.out, level["binary_path"])
            expected_size = (
                level["vertex_offset_bytes"]
                + level["vertex_count"] * VERTEX_DTYPE.itemsize
                + level["index_count"] * INDEX_DTYPE.itemsize
            )
            self.assertEqual(os.path.getsize(binary), expected_size)
            indices = np.fromfile(
                binary, dtype=INDEX_DTYPE, offset=level["index_offset_bytes"],
                count=level["index_count"],
            )
            self.assertEqual(len(indices), level["triangle_count"] * 3)

    def test_sha256_manifest_self_consistent(self):
        bake(SPEC, self.out, levels=[0], tile=128, skip_preview=True)
        manifest_path = os.path.join(self.out, "metadata.json")
        manifest = json.load(open(manifest_path))
        fresh = verify_output(self.out, manifest_path)
        self.assertTrue(fresh["ok"])
        self.assertEqual(len(manifest["sha256"]["per_file"]), len(manifest["files"]))

    def test_memory_budget_enforced_and_recorded(self):
        with self.assertRaises(MemoryBudgetExceeded):
            bake(
                SPEC, self.out, levels=[0], tile=128,
                memory_limit_bytes=4096, skip_preview=True,
            )
        self.assertFalse(os.path.exists(os.path.join(self.out, "metadata.json")))

    def test_invalid_parameters_exit_nonzero(self):
        bad = HeightfieldSpec(seed=1, resolution=256, height_scale=1.0)
        with self.assertRaises(ParameterError):
            bake(bad, self.out, levels=[0], skip_preview=True)

    def test_write_failure_leaves_no_partial_output(self):
        def inject(relative):
            # Fail at commit start so no file reaches the target directory.
            if relative == "__commit__":
                raise OSError("disk full")

        with self.assertRaises(Exception):
            bake(SPEC, self.out, levels=[0], tile=128,
                 inject_failure=inject, skip_preview=True)
        # No committed target files; staging directory removed.
        if os.path.exists(self.out):
            self.assertEqual(
                [entry for entry in os.listdir(self.out) if not entry.endswith(".staging")],
                [],
            )
        self.assertFalse(os.path.exists(self.out + ".staging"))

    def test_tile_stats_recorded(self):
        result = bake(SPEC, self.out, levels=[0, 1], tile=64, skip_preview=True)
        names = [stat["name"] for stat in result["manifest"]["tile_stats"]]
        self.assertIn("normal", names)
        self.assertTrue(any(name.startswith("lod") for name in names))
        self.assertTrue(
            all(stat["elapsed_ms"] >= 0 for stat in result["manifest"]["tile_stats"])
        )


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_bad_args_returns_code_2(self):
        code = main(["--seed", "1", "--resolution", "256",
                     "--height-scale", "1", "--output",
                     os.path.join(self.tmp.name, "x")])
        self.assertEqual(code, 2)

    def test_cli_success_returns_zero_without_preview(self):
        out = os.path.join(self.tmp.name, "ok")
        code = main([
            "--seed", "11", "--resolution", "257", "--height-scale", "8",
            "--frequency", "2", "--levels", "0", "1",
            "--output", out, "--tile", "128", "--skip-preview",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(out, "metadata.json")))


if __name__ == "__main__":
    unittest.main()
