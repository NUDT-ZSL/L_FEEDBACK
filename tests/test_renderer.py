from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "renderer.py"
GENERATOR = ROOT / "tools" / "generate_scene.py"


def run_process(arguments: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {arguments}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def generate_scene(path: Path, *, frames: int, width: int = 320, height: int = 180) -> dict:
    result = run_process(
        [
            str(GENERATOR),
            "--output",
            str(path),
            "--frames",
            str(frames),
            "--width",
            str(width),
            "--height",
            str(height),
            "--shadow-resolution",
            "256",
        ]
    )
    return json.loads(result.stdout)


def render_scene(
    scene_path: Path,
    output_dir: Path,
    *,
    incremental: bool = False,
    threads: int = 1,
) -> dict:
    arguments = [
        str(RENDERER),
        "--scene",
        str(scene_path),
        "--output",
        str(output_dir),
        "--threads",
        str(threads),
    ]
    if incremental:
        arguments.append("--incremental")
    result = run_process(arguments)
    return json.loads(result.stdout)


def output_files(output_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="bone_render_test_")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_threads_and_repeated_runs_are_byte_identical(self) -> None:
        scene_path = self.root / "scene.json"
        generate_scene(scene_path, frames=3, width=320, height=180)
        output_single = self.root / "single"
        output_multi = self.root / "multi"
        output_repeat = self.root / "repeat"
        render_scene(scene_path, output_single, threads=1)
        render_scene(scene_path, output_multi, threads=4)
        render_scene(scene_path, output_repeat, threads=1)
        single = output_files(output_single)
        multi = output_files(output_multi)
        repeated = output_files(output_repeat)
        self.assertEqual(sorted(single), sorted(multi))
        self.assertEqual(single, repeated)
        for name in single:
            self.assertEqual(single[name], multi[name], name)

        frame0 = output_single / "frame_0000.png"
        frame1 = output_single / "frame_0001.png"
        self.assertNotEqual(frame0.read_bytes(), frame1.read_bytes())
        with Image.open(frame0) as image:
            self.assertEqual(image.size, (320, 180))
            self.assertEqual(image.mode, "RGB")
            pixels0 = np.asarray(image)
        with Image.open(frame1) as image:
            pixels1 = np.asarray(image)
        self.assertGreater(int(np.count_nonzero(pixels0 != pixels1)), 0)

    def test_statistics_fixed_precision_cascades_and_materials(self) -> None:
        scene_path = self.root / "scene.json"
        output_dir = self.root / "out"
        generate_scene(scene_path, frames=2)
        render_scene(scene_path, output_dir, threads=2)
        stats_text = (output_dir / "frame_0000.json").read_text(encoding="utf-8")
        stats = json.loads(stats_text)
        for field in (
            "depth_min",
            "depth_max",
            "shadow_bias",
            "shadow_normal_offset",
        ):
            self.assertRegex(stats_text, rf'"{field}": -?\d+\.\d{{6}}')
        ratios = stats["shadow_cascade_valid_ratio"]
        self.assertEqual(len(ratios), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in ratios))
        self.assertTrue(all(value > 0.0 for value in ratios))
        self.assertEqual(len(stats["shadow_cascades"]), 3)
        self.assertEqual(sum(stats["shadow_cascade_pixel_counts"]), stats["rendered_pixel_count"])
        self.assertGreater(stats["rendered_pixel_count"], 0)
        self.assertLess(stats["depth_min"], stats["depth_max"])
        self.assertEqual(len(stats["material_pixel_counts"]), 4)
        self.assertTrue(all(value > 0 for value in stats["material_pixel_counts"]))
        self.assertEqual(stats["invalid_pixel_count"], 0)
        self.assertEqual(stats["nan_pixel_count"], 0)

    def test_incremental_append_matches_full_render(self) -> None:
        scene_path = self.root / "scene.json"
        output_dir = self.root / "incremental"
        generate_scene(scene_path, frames=2)
        render_scene(scene_path, output_dir, incremental=True, threads=1)
        old_mtimes = {
            name: (output_dir / name).stat().st_mtime_ns
            for name in ("frame_0000.png", "frame_0000.json", "frame_0001.png", "frame_0001.json")
        }
        (output_dir / "frame_0099.png").write_bytes(b"stale")
        (output_dir / "frame_0099.json").write_bytes(b"stale")
        time.sleep(0.08)
        generate_scene(scene_path, frames=4)
        summary = render_scene(scene_path, output_dir, incremental=True, threads=2)
        self.assertEqual(summary["skipped_frames"], [0, 1])
        self.assertEqual(summary["rendered_frames"], [2, 3])
        for name, old_mtime in old_mtimes.items():
            self.assertEqual((output_dir / name).stat().st_mtime_ns, old_mtime, name)
        self.assertFalse((output_dir / "frame_0099.png").exists())
        self.assertFalse((output_dir / "frame_0099.json").exists())

        full_dir = self.root / "full"
        render_scene(scene_path, full_dir, threads=4)
        self.assertEqual(output_files(output_dir), output_files(full_dir))

        all_mtimes = {
            path.name: path.stat().st_mtime_ns
            for path in output_dir.iterdir()
            if path.is_file()
        }
        time.sleep(0.08)
        summary = render_scene(scene_path, output_dir, incremental=True, threads=2)
        self.assertEqual(summary["rendered_frames"], [])
        self.assertEqual(summary["skipped_frames"], [0, 1, 2, 3])
        for name, old_mtime in all_mtimes.items():
            if name != "manifest.json":
                self.assertEqual((output_dir / name).stat().st_mtime_ns, old_mtime, name)

    def test_structural_hash_change_rebuilds_output(self) -> None:
        scene_path = self.root / "scene.json"
        output_dir = self.root / "out"
        generate_scene(scene_path, frames=2)
        render_scene(scene_path, output_dir, incremental=True, threads=1)
        old_png = (output_dir / "frame_0000.png").read_bytes()
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        scene["materials"][0]["diffuse"] = [0.12, 0.75, 0.2]
        scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = render_scene(scene_path, output_dir, incremental=True, threads=2)
        self.assertEqual(summary["skipped_frames"], [])
        self.assertEqual(summary["rendered_frames"], [0, 1])
        self.assertNotEqual(old_png, (output_dir / "frame_0000.png").read_bytes())
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["frame_numbers"], [0, 1])
        self.assertEqual(manifest["scene_hash"], summary["scene_hash"])

    def test_tampered_incremental_output_rebuilds(self) -> None:
        scene_path = self.root / "scene.json"
        output_dir = self.root / "out"
        generate_scene(scene_path, frames=2)
        render_scene(scene_path, output_dir, incremental=True, threads=1)
        (output_dir / "frame_0001.png").write_bytes(b"tampered")
        summary = render_scene(scene_path, output_dir, incremental=True, threads=2)
        self.assertEqual(summary["rendered_frames"], [0, 1])
        self.assertEqual(summary["skipped_frames"], [])
        with Image.open(output_dir / "frame_0001.png") as image:
            self.assertEqual(image.size, (320, 180))
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        payload = (output_dir / "frame_0001.png").read_bytes()
        self.assertEqual(
            manifest["frames"][1]["png_sha256"],
            hashlib.sha256(payload).hexdigest(),
        )

    def test_business_error_codes_and_paths(self) -> None:
        scene_path = self.root / "base.json"
        generate_scene(scene_path, frames=1)
        base = json.loads(scene_path.read_text(encoding="utf-8"))

        mutations: list[tuple[str, dict, str, str]] = []

        missing = copy.deepcopy(base)
        del missing["camera"]
        mutations.append(("missing", missing, "E_SCHEMA", "camera"))

        cycle = copy.deepcopy(base)
        cycle["skeleton"]["bones"][2]["parent"] = 3
        cycle["skeleton"]["bones"][3]["parent"] = 2
        mutations.append(("cycle", cycle, "E_BONE_CYCLE", "skeleton.bones[2].parent"))

        bad_matrix = copy.deepcopy(base)
        bad_matrix["skeleton"]["bones"][0]["inverse_bind_matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mutations.append(("matrix", bad_matrix, "E_MATRIX_DIM", "skeleton.bones[0].inverse_bind_matrix"))

        too_many = copy.deepcopy(base)
        too_many["mesh"]["skin_indices"][0].append(0)
        too_many["mesh"]["skin_weights"][0].append(0.0)
        mutations.append(("weights", too_many, "E_WEIGHT_COUNT", "mesh.skin_indices[0]"))

        bad_sum = copy.deepcopy(base)
        bad_sum["mesh"]["skin_weights"][0] = [
            value * 0.5 for value in bad_sum["mesh"]["skin_weights"][0]
        ]
        mutations.append(("sum", bad_sum, "E_WEIGHT_SUM", "mesh.skin_weights[0]"))

        bad_bone = copy.deepcopy(base)
        bad_bone["mesh"]["skin_indices"][0][0] = 99
        mutations.append(("bone", bad_bone, "E_WEIGHT_BONE_RANGE", "mesh.skin_indices[0][0]"))

        duplicate_names = copy.deepcopy(base)
        duplicate_names["output"]["manifest_name"] = "frame_0000.json"
        mutations.append(("names", duplicate_names, "E_SCHEMA", "output"))

        for name, scene, expected_code, expected_path in mutations:
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = run_process(
                [
                    str(RENDERER),
                    "--scene",
                    str(path),
                    "--output",
                    str(self.root / f"out_{name}"),
                ],
                expect_success=False,
            )
            self.assertNotEqual(result.returncode, 0)
            error = json.loads(result.stderr)
            self.assertEqual(error["error"]["code"], expected_code)
            self.assertEqual(error["error"]["path"], expected_path)
            self.assertIn("message", error["error"])

        result = run_process(
            [
                str(RENDERER),
                "--scene",
                str(scene_path),
                "--output",
                str(self.root / "threads"),
                "--threads",
                str((os.cpu_count() or 1) + 1),
            ],
            expect_success=False,
        )
        error = json.loads(result.stderr)["error"]
        self.assertEqual(error["code"], "E_THREADS")
        self.assertEqual(error["path"], "threads")

    def test_runtime_invalid_geometry_is_reported_without_aborting_frames(self) -> None:
        scene_path = self.root / "invalid_runtime.json"
        output_dir = self.root / "out"
        generate_scene(scene_path, frames=2)
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        huge = np.eye(4, dtype=np.float64).reshape(-1).tolist()
        huge[0] = 1e308
        huge[5] = 1e308
        huge[10] = 1e308
        scene["skeleton"]["bones"][1]["local_frames"][0]["matrix"] = huge
        scene["skeleton"]["bones"][2]["local_frames"][0]["matrix"] = huge
        scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = render_scene(scene_path, output_dir, threads=1)
        self.assertEqual(summary["rendered_frames"], [0, 1])
        self.assertTrue((output_dir / "frame_0000.png").is_file())
        self.assertTrue((output_dir / "frame_0001.png").is_file())
        stats = json.loads((output_dir / "frame_0000.json").read_text(encoding="utf-8"))
        issues = stats["issues"]
        self.assertGreater(
            issues["invalid_vertex_count"] + issues["nan_vertex_count"], 0
        )
        self.assertGreater(stats["rendered_pixel_count"], 0)

    def test_invalid_json_is_schema_error(self) -> None:
        path = self.root / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        result = run_process(
            [
                str(RENDERER),
                "--scene",
                str(path),
                "--output",
                str(self.root / "bad_out"),
            ],
            expect_success=False,
        )
        self.assertEqual(result.returncode, 10)
        error = json.loads(result.stderr)["error"]
        self.assertEqual(error["code"], "E_SCHEMA")
        self.assertEqual(error["path"], "$")


if __name__ == "__main__":
    unittest.main()
