from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


class RenderValidatorAcceptanceTests(unittest.TestCase):
    maxDiff = None

    def _run(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + current if current else "")
        return subprocess.run(
            [sys.executable, "-m", "render_check", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _base_material(self) -> dict:
        material = json.loads((ASSETS / "materials.json").read_text(encoding="utf-8"))[0]
        material = copy.deepcopy(material)
        material["vertex_shader"] = str((ASSETS / material["vertex_shader"]).resolve())
        material["fragment_shader"] = str((ASSETS / material["fragment_shader"]).resolve())
        material["texture"]["path"] = str(
            (ASSETS / material["texture"]["path"]).resolve()
        )
        return material

    def _write_materials(self, directory: Path, material: dict) -> Path:
        path = directory / "materials.json"
        path.write_text(json.dumps([material], indent=2), encoding="utf-8")
        return path

    def _assert_error(
        self,
        result: subprocess.CompletedProcess[str],
        exit_code: int,
        business_code: str,
        *,
        out_dir: Path | None = None,
    ) -> None:
        self.assertEqual(result.returncode, exit_code, result.stderr)
        self.assertIn(business_code, result.stderr)
        if out_dir is not None:
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "error")
            self.assertEqual(summary["error"]["code"], business_code)
            self.assertEqual(summary["error"]["exit_code"], exit_code)

    def test_success_artifacts_mapping_and_cross_run_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first_dir = base / "first"
            second_dir = base / "second"
            common = (
                "--instances",
                "4",
                "--seed",
                "42",
                "--width",
                "48",
                "--height",
                "48",
            )
            first = self._run("--out", str(first_dir), *common)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self._run("--out", str(second_dir), *common)
            self.assertEqual(second.returncode, 0, second.stderr)

            summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "ok")
            self.assertIsNone(summary["error"])
            self.assertEqual(summary["successful_instances"], 4)
            self.assertEqual(summary["failed_instances"], 0)
            self.assertGreater(summary["peak_memory_bytes"], 0)
            self.assertLessEqual(summary["peak_memory_bytes"], 512 * 1024 * 1024)
            self.assertLess(summary["elapsed_seconds"], 30.0)
            self.assertGreaterEqual(summary["context"]["version_code"], 410)
            self.assertEqual(summary["context"]["profile"], "core")
            self.assertEqual(summary["channel_format"], "GL_RGBA32F")
            self.assertLessEqual(summary["determinism"]["max_abs_diff"], 1e-6)

            with (first_dir / "render_log.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["instance_index"]) for row in rows], [0, 1, 2, 3])
            self.assertEqual(
                [row["material_name"] for row in rows],
                ["pbr_metal", "rough_plastic", "emissive", "pbr_metal"],
            )
            required_columns = {
                "instance_index",
                "material_name",
                "color_file",
                "normal_file",
                "depth_file",
                "object_id_file",
                "render_time_ms",
                "state_hash",
            }
            self.assertTrue(required_columns.issubset(rows[0].keys()))

            channel_map = json.loads(
                (first_dir / "channel_map.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(channel_map), 4)
            snapshots = [
                json.loads(line)
                for line in (first_dir / "state_snapshots.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual(len(snapshots), 4)
            for index, row in enumerate(rows):
                self.assertEqual(channel_map[index]["state_hash"], row["state_hash"])
                self.assertEqual(snapshots[index]["state_hash"], row["state_hash"])
                self.assertTrue(snapshots[index]["matches_expected"])
                object_id = index + 1
                channels: dict[str, np.ndarray] = {}
                for channel in ("color", "normal", "depth", "object_id"):
                    filename = row[f"{channel}_file"]
                    path = first_dir / filename
                    self.assertTrue(path.is_file(), path)
                    array = np.load(path, allow_pickle=False)
                    self.assertEqual(array.shape, (48, 48, 4))
                    self.assertEqual(array.dtype, np.float32)
                    self.assertTrue(np.isfinite(array).all())
                    channels[channel] = array
                ids = set(np.unique(channels["object_id"][..., 0]).tolist())
                self.assertEqual(ids, {0.0, float(object_id)})
                self.assertGreater(float(channels["depth"][..., 2].max()), 0.0)

            for path in sorted(first_dir.glob("instance_*.npy")):
                first_array = np.load(path, allow_pickle=False)
                second_array = np.load(second_dir / path.name, allow_pickle=False)
                difference = np.max(np.abs(first_array - second_array), initial=0.0)
                self.assertLessEqual(float(difference), 1e-6, path.name)
            with (second_dir / "render_log.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                second_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["state_hash"] for row in rows],
                [row["state_hash"] for row in second_rows],
            )

    def test_material_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            material = self._base_material()
            del material["texture"]
            config = self._write_materials(base, material)
            out_dir = base / "out"
            result = self._run(
                "--materials",
                str(config),
                "--out",
                str(out_dir),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
            )
            self._assert_error(result, 2, "E_MAT_001", out_dir=out_dir)

    def test_shader_compile_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            material = self._base_material()
            broken = base / "broken.frag"
            broken.write_text(
                "#version 410 core\nout vec4 out_color;\n"
                "void main(){ this_does_not_exist(); }\n",
                encoding="utf-8",
            )
            material["fragment_shader"] = str(broken)
            config = self._write_materials(base, material)
            out_dir = base / "out"
            result = self._run(
                "--materials",
                str(config),
                "--out",
                str(out_dir),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
            )
            self._assert_error(result, 3, "E_GL_001", out_dir=out_dir)
            self.assertFalse(list(out_dir.glob("instance_*.npy")))

    def test_shader_link_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            material = self._base_material()
            vertex = base / "link.vert"
            fragment = base / "link.frag"
            vertex.write_text(
                "#version 410 core\nout vec3 v_color;\n"
                "void main(){ gl_Position=vec4(0.0); v_color=vec3(1.0); }\n",
                encoding="utf-8",
            )
            fragment.write_text(
                "#version 410 core\nin vec4 v_color; out vec4 out_color;\n"
                "void main(){ out_color=v_color; }\n",
                encoding="utf-8",
            )
            material["vertex_shader"] = str(vertex)
            material["fragment_shader"] = str(fragment)
            config = self._write_materials(base, material)
            out_dir = base / "out"
            result = self._run(
                "--materials",
                str(config),
                "--out",
                str(out_dir),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
            )
            self._assert_error(result, 3, "E_GL_002", out_dir=out_dir)
            self.assertFalse(list(out_dir.glob("instance_*.npy")))

    def test_texture_format_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            material = self._base_material()
            material["texture"]["format"] = "RGBA32F"
            config = self._write_materials(base, material)
            out_dir = base / "out"
            result = self._run(
                "--materials",
                str(config),
                "--out",
                str(out_dir),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
            )
            self._assert_error(result, 6, "E_TEX_001", out_dir=out_dir)
            self.assertIn("pbr_metal", result.stderr)
            self.assertIn("RGBA32F", result.stderr)
            self.assertIn("RGBA8", result.stderr)

    def test_uniform_state_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "out"
            result = self._run(
                "--out",
                str(out_dir),
                "--instances",
                "3",
                "--width",
                "32",
                "--height",
                "32",
                "--fault-inject",
                "state_residue",
            )
            self._assert_error(result, 4, "E_STATE_001", out_dir=out_dir)
            self.assertIn("u_metallic", result.stderr)
            snapshots = [
                json.loads(line)
                for line in (out_dir / "state_snapshots.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertTrue(snapshots)
            self.assertFalse(snapshots[-1]["matches_expected"])

    def test_determinism_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "out"
            result = self._run(
                "--out",
                str(out_dir),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
                "--fault-inject",
                "diff",
            )
            self._assert_error(result, 5, "E_DIFF_001", out_dir=out_dir)
            self.assertIn("exceeds", result.stderr)

    def test_performance_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            memory_out = base / "memory"
            memory_result = self._run(
                "--out",
                str(memory_out),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
                "--memory-limit-mb",
                "0.001",
            )
            self._assert_error(memory_result, 7, "E_PERF_001", out_dir=memory_out)
            self.assertIn("memory", memory_result.stderr.lower())

            time_out = base / "time"
            time_result = self._run(
                "--out",
                str(time_out),
                "--instances",
                "1",
                "--width",
                "32",
                "--height",
                "32",
                "--time-limit-sec",
                "0.000000001",
            )
            self._assert_error(time_result, 7, "E_PERF_001", out_dir=time_out)
            self.assertIn("elapsed", time_result.stderr.lower())

    def test_instance_count_bounds(self) -> None:
        for instances in ("0", "65"):
            with self.subTest(instances=instances):
                result = self._run("--instances", instances, "--out", "unused-arg-test")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("E_ARG_001", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

