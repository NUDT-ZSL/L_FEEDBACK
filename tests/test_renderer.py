from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from softraster.math3d import PerspectiveCamera
from softraster.postprocess import apply_srgb_gamma, apply_tonemap
from softraster.raster import iter_mesh_fragments, rasterize_gbuffer
from softraster.renderer import render_scene
from softraster.scene import Layer, Mesh, load_scene
from softraster.shading import perturbed_layer_normal
from softraster.shadows import build_shadow_maps, compute_visibility


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "studio_scene.json"
RENDERER = ROOT / "renderer.py"


def run_renderer(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RENDERER), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_payload = json.loads(SAMPLE.read_text(encoding="utf-8"))

    def test_hdr_output_and_two_render_exact_self_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "hdr.npy"
            completed = run_renderer(
                str(SAMPLE), "-o", str(output), "--self-check"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["deterministic"])
            self.assertEqual(report["different_pixels"], 0)
            self.assertEqual(report["max_abs_diff"], 0.0)
            image = np.load(output)
            self.assertEqual(image.shape, (240, 320, 3))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreater(float(image.max()), 1.0)

            second = Path(temporary) / "hdr_second.npy"
            completed_second = run_renderer(
                str(SAMPLE), "-o", str(second)
            )
            self.assertEqual(completed_second.returncode, 0)
            self.assertTrue(np.array_equal(image, np.load(second)))

    def test_tonemap_and_gamma_switches_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            small_payload = copy.deepcopy(self.sample_payload)
            small_payload["render"]["width"] = 80
            small_payload["render"]["height"] = 60
            for light in small_payload["scene"]["lights"]:
                light["shadow"]["map_size"] = 48
            scene_path = temporary_path / "small.json"
            scene_path.write_text(json.dumps(small_payload), encoding="utf-8")

            raw_path = temporary_path / "raw.npy"
            aces_path = temporary_path / "aces.npy"
            gamma_path = temporary_path / "gamma.npy"
            both_path = temporary_path / "both.npy"
            commands = [
                (raw_path, []),
                (aces_path, ["--tone-map"]),
                (gamma_path, ["--gamma"]),
                (both_path, ["--tone-map", "--gamma"]),
            ]
            for output, flags in commands:
                completed = run_renderer(
                    str(scene_path), "-o", str(output), *flags
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            raw = np.load(raw_path)
            aces = np.load(aces_path)
            gamma = np.load(gamma_path)
            both = np.load(both_path)
            np.testing.assert_allclose(
                aces, apply_tonemap(raw, "aces"), rtol=0.0, atol=1.0e-7
            )
            np.testing.assert_allclose(
                gamma, apply_srgb_gamma(raw), rtol=0.0, atol=2.0e-6
            )
            np.testing.assert_allclose(
                both,
                apply_srgb_gamma(apply_tonemap(raw, "aces")),
                rtol=0.0,
                atol=2.0e-6,
            )
            self.assertFalse(np.array_equal(aces, gamma))
            self.assertFalse(np.array_equal(aces, both))
            self.assertFalse(np.array_equal(gamma, both))

    def test_business_errors_and_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cases = []

            schema_payload = copy.deepcopy(self.sample_payload)
            del schema_payload["scene"]["camera"]
            cases.append(("missing_camera", "RENDER_ERR_SCHEMA", 2, schema_payload))

            index_payload = copy.deepcopy(self.sample_payload)
            index_payload["scene"]["meshes"][0]["material"] = 999
            cases.append(
                ("bad_material_index", "RENDER_ERR_INDEX", 3, index_payload)
            )

            vertex_payload = copy.deepcopy(self.sample_payload)
            vertex_payload["scene"]["meshes"][0]["triangles"][0][0] = 999
            cases.append(
                ("bad_vertex_index", "RENDER_ERR_INDEX", 3, vertex_payload)
            )

            layer_payload = copy.deepcopy(self.sample_payload)
            layers = layer_payload["scene"]["materials"][0]["layers"]
            layer_payload["scene"]["materials"][0]["layers"] = [
                copy.deepcopy(layers[0]) for _ in range(9)
            ]
            cases.append(
                ("too_many_layers", "RENDER_ERR_LAYER_LIMIT", 4, layer_payload)
            )

            for name, expected_code, expected_exit, payload in cases:
                scene_path = temporary_path / f"{name}.json"
                output_path = temporary_path / f"{name}.npy"
                scene_path.write_text(json.dumps(payload), encoding="utf-8")
                completed = run_renderer(
                    str(scene_path), "-o", str(output_path)
                )
                self.assertEqual(completed.returncode, expected_exit)
                error = json.loads(completed.stderr)
                self.assertEqual(error["code"], expected_code)
                self.assertFalse(output_path.exists())

    def test_near_plane_clip_interpolates_only_clipped_vertices(self) -> None:
        camera = PerspectiveCamera(
            eye=np.asarray([0.0, 0.0, 0.0]),
            target=np.asarray([0.0, 0.0, -1.0]),
            up_hint=np.asarray([0.0, 1.0, 0.0]),
            fov_y_radians=np.deg2rad(80.0),
            near=1.0,
            far=20.0,
            width=96,
            height=72,
        )
        vertices = np.asarray(
            [[0.0, 0.0, 0.8], [1.5, -1.5, -5.0], [-1.5, -1.5, -5.0]],
            dtype=np.float64,
        )
        normals = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        mesh = Mesh(
            name="crosses_near",
            vertices=vertices,
            normals=normals,
            indices=np.asarray([[0, 1, 2]], dtype=np.int64),
            material_index=0,
            double_sided=True,
            center=np.asarray([0.0, -1.0, -3.0]),
            radius=3.0,
        )
        fragments = list(iter_mesh_fragments(camera, mesh, False))
        self.assertTrue(fragments)
        all_positions = np.concatenate([item[3] for item in fragments], axis=0)
        all_depths = np.concatenate([item[2] for item in fragments], axis=0)
        self.assertTrue(np.all(np.isfinite(all_positions)))
        self.assertTrue(np.all(all_depths >= camera.near - 1.0e-9))
        self.assertTrue(np.all(all_positions[:, 2] <= -camera.near + 1.0e-9))

        behind = Mesh(
            name="behind_near",
            vertices=np.asarray(
                [[0.0, 0.0, 0.5], [1.0, 0.0, 0.4], [0.0, 1.0, 0.3]]
            ),
            normals=normals.copy(),
            indices=np.asarray([[0, 1, 2]], dtype=np.int64),
            material_index=0,
            double_sided=True,
            center=np.asarray([0.3, 0.3, 0.4]),
            radius=0.8,
        )
        self.assertEqual(list(iter_mesh_fragments(camera, behind, False)), [])

        degenerate = dataclasses.replace(
            mesh,
            vertices=np.asarray(
                [[-1.0, 0.0, -4.0], [0.0, 0.0, -4.0], [1.0, 0.0, -4.0]]
            ),
        )
        self.assertEqual(
            list(iter_mesh_fragments(camera, degenerate, False)), []
        )

    def test_frustum_culling_and_offscreen_shadow_caster(self) -> None:
        scene = load_scene(SAMPLE)
        result = render_scene(scene)
        self.assertFalse(bool(result.mesh_visibility[-1]))
        self.assertFalse(
            np.any(result.gbuffer.mesh_index == len(scene.meshes) - 1)
        )

        hit_pixels = result.gbuffer.hit_mask
        points = result.gbuffer.world_position[hit_pixels]
        normals = result.gbuffer.normal[hit_pixels]
        with_caster = result.shadow_maps
        without_caster = build_shadow_maps(
            dataclasses.replace(scene, meshes=scene.meshes[:-1])
        )
        changed = False
        for light, with_map, without_map in zip(
            scene.lights, with_caster, without_caster
        ):
            visible_with = compute_visibility(light, with_map, points, normals)
            visible_without = compute_visibility(
                light, without_map, points, normals
            )
            if not np.array_equal(visible_with, visible_without):
                changed = True
        self.assertTrue(changed, "offscreen caster must still affect shadows")

    def test_layer_normals_do_not_feed_into_upper_layers(self) -> None:
        geometric = np.asarray(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
        bottom = Layer(
            base_color=np.asarray([0.2, 0.2, 0.2]),
            metallic=0.0,
            roughness=0.5,
            clearcoat=0.0,
            normal_strength=2.0,
            normal=np.asarray([1.0, 0.0, 1.0]),
            opacity=1.0,
        )
        clearcoat = Layer(
            base_color=np.asarray([0.8, 0.8, 0.8]),
            metallic=0.0,
            roughness=0.08,
            clearcoat=1.0,
            normal_strength=1.0,
            normal=np.asarray([0.0, 0.0, 1.0]),
            opacity=0.25,
        )
        perturbed = perturbed_layer_normal(geometric, bottom)
        upper = perturbed_layer_normal(geometric, clearcoat)
        np.testing.assert_allclose(upper, geometric, atol=1.0e-12)
        self.assertFalse(np.allclose(perturbed, geometric))

    def test_pcf_radius_changes_soft_visibility(self) -> None:
        scene = load_scene(SAMPLE)
        light = scene.lights[1]
        wide = dataclasses.replace(
            scene,
            lights=tuple(
                dataclasses.replace(
                    scene.lights[1],
                    shadow=dataclasses.replace(scene.lights[1].shadow, radius=4.0),
                )
                if index == 1
                else item
                for index, item in enumerate(scene.lights)
            ),
        )
        narrow = dataclasses.replace(
            scene,
            lights=tuple(
                dataclasses.replace(
                    scene.lights[1],
                    shadow=dataclasses.replace(scene.lights[1].shadow, radius=0.0),
                )
                if index == 1
                else item
                for index, item in enumerate(scene.lights)
            ),
        )
        gbuffer = rasterize_gbuffer(
            scene.camera,
            scene.meshes,
            np.asarray([True, True, True, True, True, False]),
        )
        floor = gbuffer.mesh_index == 0
        points = gbuffer.world_position[floor][::2]
        normals = gbuffer.normal[floor][::2]
        wide_map = build_shadow_maps(wide)[1]
        narrow_map = build_shadow_maps(narrow)[1]
        wide_visibility = compute_visibility(
            wide.lights[1], wide_map, points, normals
        )
        narrow_visibility = compute_visibility(
            narrow.lights[1], narrow_map, points, normals
        )
        self.assertFalse(np.array_equal(wide_visibility, narrow_visibility))
        self.assertTrue(np.any((wide_visibility > 0.0) & (wide_visibility < 1.0)))
        self.assertTrue(np.all((narrow_visibility >= 0.0) & (narrow_visibility <= 1.0)))
        self.assertEqual(light.kind, "point")


if __name__ == "__main__":
    unittest.main()
