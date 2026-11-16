from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from soft_renderer.imageio import downsample
from soft_renderer.renderer import LayerBuffer, compose_layers, render_scene
from soft_renderer.sample_scene import write_indoor_scene
from soft_renderer.scene import load_scene
from soft_renderer.shadows import build_shadow_maps, light_visibility


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.scene_path = str(Path(cls._temporary.name) / "indoor.json")
        write_indoor_scene(cls.scene_path, 64, 48)
        cls.scene = load_scene(cls.scene_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_block_sizes_are_bitwise_identical(self) -> None:
        frames = []
        for block_size in (1, 2, 3, 4, 8):
            result = render_scene(
                self.scene, block_size=block_size, preset_name="low"
            )
            frames.append(result.frame.tobytes())
        untiled = render_scene(
            self.scene, block_size=8, preset_name="low", no_tiling=True
        )
        frames.append(untiled.frame.tobytes())
        for frame in frames[1:]:
            self.assertEqual(frames[0], frame)

    def test_layers_and_supersampling_are_bitwise_identical(self) -> None:
        result = render_scene(self.scene, block_size=4, preset_name="medium")
        self.assertEqual(result.frame.shape, (48, 64, 3))
        rebuilt = downsample(
            compose_layers(result.layers, self.scene.background),
            result.preset.supersample,
        )
        self.assertEqual(result.frame.tobytes(), rebuilt.tobytes())
        self.assertGreater(float(result.frame.max()), 1.0)

    def test_shadow_stats_include_lit_penumbra_and_occluded(self) -> None:
        result = render_scene(self.scene, block_size=4, preset_name="medium")
        classifications = list(result.shadow_statistics.values())
        self.assertTrue(
            any(
                values["fully_lit_fragment_samples"] > 0
                and values["penumbra_fragment_samples"] > 0
                and values["fully_shadowed_fragment_samples"] > 0
                for values in classifications
            )
        )

    def test_flat_casters_do_not_self_shadow(self) -> None:
        floor_scene = replace(
            self.scene,
            triangles=self.scene.triangles[:2],
            triangle_materials=self.scene.triangle_materials[:2],
        )
        maps, _ = build_shadow_maps(floor_scene, 128)
        x, z = np.meshgrid(np.linspace(-4.0, 4.0, 9), np.linspace(-2.0, 4.0, 9))
        positions = np.stack(
            [x.reshape(-1), np.zeros(x.size), z.reshape(-1)], axis=-1
        )
        normals = np.tile(np.array([0.0, 1.0, 0.0]), (len(positions), 1))
        for light in floor_scene.lights:
            delta = light.position[None, :] - positions
            distance = np.sqrt(np.sum(delta * delta, axis=-1))
            directions = delta / distance[:, None]
            active = np.ones(len(positions), dtype=bool)
            if light.kind == "spot":
                cosine = np.sum(-directions * light.direction[None, :], axis=-1)
                active = cosine >= light.inner_cos
            visibility = light_visibility(
                light,
                maps[light.id],
                positions,
                normals,
                directions,
                distance,
                4,
            )
            self.assertGreater(float(np.min(visibility[active])), 0.999)

    def test_material_filter_returns_only_requested_material(self) -> None:
        result = render_scene(
            self.scene, block_size=4, preset_name="low", material_id=2
        )
        self.assertEqual(list(result.layers), [2])
        self.assertGreater(float(result.frame.max()), 0.0)
        self.assertTrue(np.any(result.frame == 0.0))

    def test_composite_selects_nearest_depth_regardless_of_material_order(self) -> None:
        far_color = np.full((2, 2, 3), 0.25, dtype=np.float32)
        near_color = np.full((2, 2, 3), 0.75, dtype=np.float32)
        layers = {
            0: LayerBuffer(0, far_color, np.full((2, 2), 5.0, dtype=np.float32)),
            1: LayerBuffer(1, near_color, np.full((2, 2), 2.0, dtype=np.float32)),
        }
        composed = compose_layers(layers, np.zeros(3, dtype=np.float64))
        np.testing.assert_array_equal(composed, near_color)


if __name__ == "__main__":
    unittest.main()
