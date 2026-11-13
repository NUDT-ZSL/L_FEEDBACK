from __future__ import annotations

import math
import unittest

from ruin_bake.canonical import canonical_json
from ruin_bake.layout import LayoutParams, build_probe_layout
from ruin_bake.math3d import box_signed_distance
from ruin_bake.scene import apply_scene_patch, generate_scene

from .helpers import SMALL_GENERATOR, make_layout, make_scene


class SceneTests(unittest.TestCase):
    def test_same_seed_is_bit_for_bit_identical(self) -> None:
        first = generate_scene(SMALL_GENERATOR)
        second = generate_scene(SMALL_GENERATOR)
        self.assertEqual(canonical_json(first.to_dict()), canonical_json(second.to_dict()))
        self.assertEqual(first.scene_hash, second.scene_hash)
        self.assertEqual(first.geometry_hashes(), second.geometry_hashes())
        self.assertEqual(first.light_hashes(), second.light_hashes())

    def test_generator_contains_required_features(self) -> None:
        scene = make_scene()
        self.assertGreaterEqual(len(scene.geometry), 2)
        self.assertTrue(any(box.rotation_deg == (0.0, 0.0, 0.0) for box in scene.geometry))
        self.assertTrue(any(box.rotation_deg != (0.0, 0.0, 0.0) for box in scene.geometry))
        self.assertTrue(any(any(component > 0.0 for component in box.emission) for box in scene.geometry))
        self.assertGreaterEqual(len(scene.point_lights), 1)
        self.assertLessEqual(len(scene.point_lights), 4)
        self.assertAlmostEqual(
            math.sqrt(sum(component * component for component in scene.directional.direction)),
            1.0,
            places=14,
        )

    def test_patch_changes_only_component_hashes_it_touches(self) -> None:
        scene = make_scene()
        patched = apply_scene_patch(scene, {"geometry_updates": {"3": {"albedo": [0.9, 0.1, 0.2]}}})
        self.assertNotEqual(scene.scene_hash, patched.scene_hash)
        old_hashes = scene.geometry_hashes()
        new_hashes = patched.geometry_hashes()
        changed = {key for key in old_hashes if old_hashes[key] != new_hashes[key]}
        self.assertEqual(changed, {"3"})
        self.assertEqual(scene.light_hashes(), patched.light_hashes())


class LayoutTests(unittest.TestCase):
    def test_layout_is_stable_and_ids_are_contiguous(self) -> None:
        scene = make_scene()
        first = build_probe_layout(scene, LayoutParams(spacing=2.0, cull_distance=0.4))
        second = build_probe_layout(scene, LayoutParams(spacing=2.0, cull_distance=0.4))
        self.assertEqual(first, second)
        self.assertEqual([probe.probe_id for probe in first.probes], list(range(len(first.probes))))
        self.assertEqual(first.generated_count, first.culled_count + len(first.probes))
        self.assertGreater(first.culled_count, 0)

    def test_all_kept_probes_respect_cull_distance(self) -> None:
        scene = make_scene()
        layout = make_layout(scene)
        threshold = layout.params.cull_distance
        for probe in layout.probes:
            distances = [
                box_signed_distance(probe.position, box.center, box.half_size, box.rotation_deg)
                for box in scene.geometry
            ]
            self.assertGreater(min(distances), threshold)
            self.assertEqual(probe.normal, (0.0, 0.0, 1.0))

    def test_layout_parameter_change_changes_layout_hash(self) -> None:
        scene = make_scene()
        first = build_probe_layout(scene, LayoutParams(spacing=2.0))
        second = build_probe_layout(scene, LayoutParams(spacing=2.5))
        self.assertNotEqual(first.layout_hash, second.layout_hash)


if __name__ == "__main__":
    unittest.main()

