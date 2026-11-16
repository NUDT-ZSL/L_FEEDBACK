from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

from softdeferred.gbuffer import GBUFFER_LAYOUT, GBuffer


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SCENE = ROOT / "sample" / "sample_scene.json"


def run_cli(output: Path, *extra: str) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(ROOT / "render.py"),
        "--scene",
        str(SAMPLE_SCENE),
        "--output",
        str(output),
    ]
    command.extend(extra)
    return subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )


def load_report(output: Path) -> Dict[str, Any]:
    with (output / "report.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_with_absolute_assets() -> Dict[str, Any]:
    with SAMPLE_SCENE.open("r", encoding="utf-8") as handle:
        scene = json.load(handle)
    scene["materials"] = str((SAMPLE_SCENE.parent / scene["materials"]).resolve())
    for mesh in scene["meshes"]:
        mesh["path"] = str((SAMPLE_SCENE.parent / mesh["path"]).resolve())
    return scene


class PipelineAcceptanceTest(unittest.TestCase):
    def test_render_contract_and_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "render"
            completed = run_cli(
                output,
                "--width",
                "128",
                "--height",
                "96",
                "--shadow-resolution",
                "64",
                "--cascade-count",
                "2",
                "--depth-tolerance",
                "0.0055",
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg="stdout=%s\nstderr=%s" % (completed.stdout, completed.stderr),
            )
            self.assertTrue(output.is_dir())
            report = load_report(output)
            self.assertEqual(report["resolution"], {"width": 128, "height": 96})
            self.assertEqual(report["shadow_maps"]["resolution"], 64)
            self.assertEqual(
                report["shadow_maps"]["directional"]["sun"]["cascade_count"], 2
            )
            self.assertEqual(report["shadow_maps"]["map_count"], 9)
            self.assertAlmostEqual(
                report["depth_parameters"]["compare_tolerance"], 0.0055
            )
            self.assertAlmostEqual(
                report["shadow_maps"]["compare_tolerance"], 0.0055
            )
            self.assertTrue(report["depth_parameters"]["used_for_render_and_query"])

            gbuffer_directory = output / "gbuffer"
            with (gbuffer_directory / "layout.json").open("r", encoding="utf-8") as handle:
                layout = json.load(handle)
            self.assertEqual(set(layout["channels"]), set(GBUFFER_LAYOUT))
            for channel_name, channel_layout in GBUFFER_LAYOUT.items():
                channel_path = gbuffer_directory / channel_layout["file"]
                self.assertTrue(channel_path.is_file(), channel_name)
                array = np.load(channel_path, allow_pickle=False)
                self.assertEqual(
                    str(array.dtype), channel_layout["dtype"], channel_name
                )

            gbuffer = GBuffer.load(gbuffer_directory)
            self.assertGreater(gbuffer.covered_pixels, 1000)
            self.assertEqual(gbuffer.coverage.shape, (96, 128))
            self.assertEqual(gbuffer.position.shape, (96, 128, 3))
            coverage = gbuffer.coverage.astype(bool)
            self.assertTrue(np.all(np.isfinite(gbuffer.position[coverage])))
            normal_length = np.linalg.norm(gbuffer.normal[coverage], axis=1)
            tangent_length = np.linalg.norm(gbuffer.tangent[coverage], axis=1)
            bitangent_length = np.linalg.norm(gbuffer.bitangent[coverage], axis=1)
            self.assertLess(np.max(np.abs(normal_length - 1.0)), 2.0e-4)
            self.assertLess(np.max(np.abs(tangent_length - 1.0)), 2.0e-4)
            self.assertLess(np.max(np.abs(bitangent_length - 1.0)), 2.0e-4)
            orthogonality = np.abs(np.sum(gbuffer.normal[coverage] * gbuffer.tangent[coverage], axis=1))
            self.assertLess(float(np.max(orthogonality)), 2.0e-4)
            self.assertTrue(np.all(gbuffer.material_index[coverage] >= 0))
            self.assertTrue(np.all(gbuffer.material_index[coverage] < 3))
            self.assertAlmostEqual(float(gbuffer.depth[~coverage].min()), 1.0)
            self.assertAlmostEqual(float(gbuffer.depth[~coverage].max()), 1.0)

            hdr = np.load(output / "hdr_color.npy", allow_pickle=False)
            self.assertEqual(hdr.shape, (96, 128, 3))
            self.assertEqual(hdr.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(hdr)))
            self.assertTrue(np.all(hdr >= 0.0))
            self.assertGreater(float(np.max(hdr)), 0.0)
            self.assertEqual(report["hdr"]["tone_mapping"], "none")

            self.assertTrue((output / "depth.png").is_file())
            shadow_files = list((output / "shadow_maps").glob("*.png"))
            self.assertEqual(len(shadow_files), 9)
            self.assertEqual(len({path.name for path in shadow_files}), 9)
            self.assertEqual(
                set(report["outputs"]["shadow_map_files"]),
                set("shadow_maps/" + path.name for path in shadow_files),
            )
            for path in shadow_files:
                self.assertGreater(path.stat().st_size, 0)

            culling = report["culling"]
            self.assertEqual(
                culling["triangles_before_culling"],
                culling["triangles_after_frustum_culling"]
                + culling["mesh_frustum_culled_triangles"],
            )
            self.assertEqual(
                culling["triangles_after_frustum_culling"],
                culling["triangles_after_culling"],
            )
            self.assertGreater(culling["backface_culled_triangles"], 0)
            self.assertGreater(culling["rasterized_triangles"], 0)
            self.assertGreater(culling["fragments_written"], 0)

            interactions = report["light_pixel_interactions"]
            self.assertEqual(set(interactions), {"sun", "point_blue", "spot_amber"})
            self.assertEqual(interactions["sun"], gbuffer.covered_pixels)
            self.assertGreater(interactions["point_blue"], 0)
            self.assertGreater(interactions["spot_amber"], 0)
            self.assertGreater(report["shadow_queries"], 0)

            for stage in (
                "scene_load",
                "geometry_prepare",
                "gbuffer_rasterization",
                "shadow_map_generation",
                "deferred_shading",
                "output_write",
            ):
                self.assertIn(stage, report["timings_ms"])
                self.assertGreaterEqual(report["timings_ms"][stage], 0.0)
            self.assertGreater(report["peak_memory"]["estimate_mb"], 0.0)
            materials = report["scene"]["materials"]
            self.assertNotEqual(
                materials[0]["exponent_tangent"],
                materials[0]["exponent_bitangent"],
            )

    def test_render_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            args = (
                "--width",
                "80",
                "--height",
                "64",
                "--shadow-resolution",
                "32",
                "--cascade-count",
                "2",
            )
            one = run_cli(first, *args)
            two = run_cli(second, *args)
            self.assertEqual(one.returncode, 0, msg=one.stderr)
            self.assertEqual(two.returncode, 0, msg=two.stderr)
            for relative in (
                "hdr_color.npy",
                "depth.png",
                "gbuffer/position.npy",
                "gbuffer/normal.npy",
                "gbuffer/material_index.npy",
                "shadow_maps/directional_sun_cascade_0.png",
                "shadow_maps/point_point_blue_posX.png",
            ):
                self.assertEqual(
                    file_sha256(first / relative),
                    file_sha256(second / relative),
                    relative,
                )

    def test_mesh_frustum_culling_and_empty_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            scene = scene_with_absolute_assets()
            scene["camera"]["position"] = [0.0, 0.0, 100.0]
            scene["camera"]["target"] = [0.0, 0.0, 101.0]
            invalid_scene = temporary_path / "culled_scene.json"
            invalid_scene.write_text(json.dumps(scene), encoding="utf-8")
            output = temporary_path / "culled"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "render.py"),
                    "--scene",
                    str(invalid_scene),
                    "--output",
                    str(output),
                    "--width",
                    "64",
                    "--height",
                    "48",
                    "--shadow-resolution",
                    "32",
                    "--cascade-count",
                    "2",
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            report = load_report(output)
            culling = report["culling"]
            self.assertGreater(culling["mesh_frustum_culled_triangles"], 0)
            self.assertEqual(culling["triangles_after_frustum_culling"], 0)
            self.assertEqual(report["gbuffer"]["covered_pixels"], 0)
            self.assertEqual(report["shadow_queries"], 0)
            self.assertEqual(set(report["light_pixel_interactions"].values()), {0})

    def test_invalid_config_fails_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            scene = scene_with_absolute_assets()
            scene["lights"][0]["type"] = "area"
            invalid_scene = temporary_path / "invalid_scene.json"
            invalid_scene.write_text(json.dumps(scene), encoding="utf-8")
            output = temporary_path / "must_not_exist"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "render.py"),
                    "--scene",
                    str(invalid_scene),
                    "--output",
                    str(output),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("lights[0].type", completed.stderr)
            self.assertFalse(output.exists())
            staging = list(temporary_path.glob(".must_not_exist.tmp-*"))
            self.assertEqual(staging, [])

    def test_invalid_json_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            invalid_scene = temporary_path / "broken.json"
            invalid_scene.write_text('{"camera": ', encoding="utf-8")
            output = temporary_path / "must_not_exist"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "render.py"),
                    "--scene",
                    str(invalid_scene),
                    "--output",
                    str(output),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid JSON", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
