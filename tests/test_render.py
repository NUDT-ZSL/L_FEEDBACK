from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from matgraph.compiler import compile_graph, load_and_compile
from matgraph.evaluate import compute_tangent_frames, evaluate_material
from matgraph.lighting import shade_scene
from matgraph.pngio import png_bit_depth
from matgraph.raster import GBuffer, render_gbuffer
from matgraph.renderer import CHANNEL_FILES, render_scene
from matgraph.samples import generate_samples
from matgraph.scene import load_scene


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.sample_dir = Path(cls.temporary.name)
        generate_samples(cls.sample_dir)
        cls.graph_path = cls.sample_dir / "material_graph.json"
        cls.scene_path = cls.sample_dir / "scene.json"
        cls.compiled = load_and_compile(cls.graph_path)
        cls.scene = load_scene(cls.scene_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_analytic_depth_for_every_hit_including_edges(self):
        gbuffer = render_gbuffer(self.scene)
        valid = gbuffer.valid
        self.assertGreater(int(np.count_nonzero(gbuffer.edge)), 0)
        self.assertTrue(np.all(valid[gbuffer.edge]))

        expected_depth = np.linalg.norm(
            gbuffer.world - self.scene.camera.position[None, None, :],
            axis=-1,
        )
        np.testing.assert_allclose(gbuffer.depth[valid], expected_depth[valid], atol=1e-10)

        for index, obj in enumerate(self.scene.objects):
            mask = gbuffer.object_id == index
            if obj.type == "sphere":
                radii = np.linalg.norm(
                    gbuffer.world[mask] - obj.center[None, :],
                    axis=1,
                )
                np.testing.assert_allclose(radii, obj.radius, atol=1e-10)
            else:
                np.testing.assert_allclose(gbuffer.world[mask, 1], 0.0, atol=1e-10)
                np.testing.assert_allclose(
                    gbuffer.geometric_normal[mask],
                    np.broadcast_to(obj.normal, (int(np.count_nonzero(mask)), 3)),
                    atol=1e-12,
                )

    def test_render_outputs_and_statistics(self):
        result = render_scene(
            self.compiled,
            self.scene,
            graph_sha256="graph",
            scene_sha256="scene",
        )
        self.assertEqual(set(CHANNEL_FILES.values()), set(result.files))
        report = result.report
        self.assertEqual(report["counts"]["total_pixels"], 320 * 240)
        self.assertEqual(
            report["counts"]["valid_pixels"],
            report["counts"]["shaded_pixels"],
        )
        self.assertGreater(report["counts"]["valid_pixels"], 0)
        for channel in ("base_color", "normal", "roughness", "metallic", "emissive"):
            self.assertIn("mean", report["channels"][channel])
            self.assertIn("min", report["channels"][channel])
            self.assertIn("max", report["channels"][channel])

        normals = result.material.normal[result.gbuffer.valid]
        np.testing.assert_allclose(
            np.linalg.norm(normals, axis=1),
            1.0,
            atol=1e-12,
        )
        self.assertTrue(np.all(result.lighting.radiance >= 0.0))
        self.assertTrue(np.all(np.isfinite(result.lighting.radiance)))
        self.assertGreaterEqual(float(result.lighting.ldr_u8.min()), 0)
        self.assertLessEqual(float(result.lighting.ldr_u8.max()), 255)
        self.assertGreaterEqual(float(result.lighting.ldr_srgb.min()), 0.0)
        self.assertLessEqual(float(result.lighting.ldr_srgb.max()), 1.0)

        self.assertEqual(png_bit_depth(result.files["base_color.png"]), 8)
        self.assertEqual(png_bit_depth(result.files["normal.png"]), 16)
        self.assertEqual(png_bit_depth(result.files["roughness.png"]), 16)
        self.assertEqual(png_bit_depth(result.files["metallic.png"]), 16)
        self.assertEqual(png_bit_depth(result.files["emissive.png"]), 16)
        self.assertEqual(png_bit_depth(result.files["final.png"]), 8)
        parsed_report = json.loads(result.files["report.json"])
        self.assertEqual(parsed_report["counts"], report["counts"])

    def test_degenerate_uv_uses_stable_fallback_and_is_recorded(self):
        graph = {
            "schema": "material-graph/v1",
            "name": "degenerate-uv",
            "nodes": [
                {"id": "base", "type": "constant", "value": [0.4, 0.5, 0.6]},
                {"id": "rough", "type": "constant", "value": 0.5},
                {"id": "metal", "type": "constant", "value": 0.0},
                {"id": "emissive", "type": "constant", "value": [0.0, 0.0, 0.0]},
                {"id": "tangent", "type": "constant", "value": [0.5, 0.5, 1.0]},
                {"id": "strength", "type": "constant", "value": 1.0},
                {
                    "id": "normal",
                    "type": "normal_map",
                    "inputs": {"normal": "tangent", "strength": "strength"},
                },
            ],
            "outputs": {
                "base_color": "base",
                "normal": "normal",
                "roughness": "rough",
                "metallic": "metal",
                "emissive": "emissive",
            },
        }
        compiled = compile_graph(graph, Path("."))
        height = 4
        width = 4
        world = np.zeros((height, width, 3), dtype=np.float64)
        world[..., 0] = np.arange(width)[None, :] * 0.1
        world[..., 2] = np.arange(height)[:, None] * -0.1
        gbuffer = GBuffer(
            width=width,
            height=height,
            world=world,
            geometric_normal=np.broadcast_to(
                np.asarray([0.0, 1.0, 0.0]),
                (height, width, 3),
            ).copy(),
            uv=np.full((height, width, 2), 0.5, dtype=np.float64),
            depth=np.full((height, width), 1.0, dtype=np.float64),
            object_id=np.zeros((height, width), dtype=np.int32),
            valid=np.ones((height, width), dtype=bool),
            edge=np.zeros((height, width), dtype=bool),
        )
        frame = compute_tangent_frames(gbuffer)
        self.assertTrue(np.all(frame.degenerate))
        np.testing.assert_allclose(
            np.linalg.norm(frame.tangent, axis=-1),
            1.0,
            atol=1e-12,
        )
        material = evaluate_material(
            compiled,
            gbuffer,
            np.asarray([0.0, 1.0, 3.0]),
        )
        self.assertEqual(int(np.count_nonzero(material.degenerate_tangent)), height * width)
        normals = material.normal.reshape(-1, 3)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)

    def test_lighting_remains_energy_bounded(self):
        height = 8
        width = 8
        normal = np.zeros((height, width, 3), dtype=np.float64)
        normal[..., 1] = 1.0
        gbuffer = GBuffer(
            width=width,
            height=height,
            world=np.zeros((height, width, 3), dtype=np.float64),
            geometric_normal=normal.copy(),
            uv=np.zeros((height, width, 2), dtype=np.float64),
            depth=np.ones((height, width), dtype=np.float64),
            object_id=np.zeros((height, width), dtype=np.int32),
            valid=np.ones((height, width), dtype=bool),
            edge=np.zeros((height, width), dtype=bool),
        )
        from matgraph.evaluate import MaterialBuffers

        base = np.ones((height, width, 3), dtype=np.float64)
        material = MaterialBuffers(
            base_color=base,
            normal=normal,
            roughness=np.zeros((height, width), dtype=np.float64),
            metallic=np.ones((height, width), dtype=np.float64),
            emissive=np.zeros((height, width, 3), dtype=np.float64),
            degenerate_tangent=np.zeros((height, width), dtype=bool),
            normal_fallback=np.zeros((height, width), dtype=bool),
        )
        local_scene = replace(self.scene, width=width, height=height)
        result = shade_scene(local_scene, gbuffer, material)
        self.assertTrue(np.all(result.radiance >= 0.0))
        self.assertTrue(np.all(np.isfinite(result.radiance)))
        self.assertTrue(np.all(result.ldr_srgb >= 0.0))
        self.assertTrue(np.all(result.ldr_srgb <= 1.0))


if __name__ == "__main__":
    unittest.main()
