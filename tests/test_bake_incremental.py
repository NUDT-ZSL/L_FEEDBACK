from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path

from ruin_bake.bake import BakeParams, bake_full, bake_incremental
from ruin_bake.layout import LayoutParams
from ruin_bake.scene import apply_scene_patch
from ruin_bake.snapshot import load_snapshot, write_snapshot_atomic

from .helpers import TEST_TMP, make_layout, make_scene


class BakeDeterminismTests(unittest.TestCase):
    def test_serial_chunking_does_not_change_float_bits(self) -> None:
        scene = make_scene()
        layout = make_layout(scene)
        one = bake_full(scene, layout, BakeParams(samples=16, workers=1, chunk_size=1))
        many = bake_full(scene, layout, BakeParams(samples=16, workers=1, chunk_size=100))
        self.assertEqual(one.irradiances, many.irradiances)
        self.assertEqual(one.affected_probe_ids, many.affected_probe_ids)

    def test_irradiance_is_non_negative_and_has_indirect_energy(self) -> None:
        scene = make_scene()
        layout = make_layout(scene)
        output = bake_full(scene, layout, BakeParams(samples=24, workers=1, chunk_size=8))
        self.assertTrue(all(component >= 0.0 for value in output.irradiances for component in value))
        self.assertTrue(any(value[0] > 0.0 for value in output.irradiances))


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        cls.directory = Path(cls._temp.name)
        cls.base_scene = make_scene()
        cls.base_layout = make_layout(cls.base_scene)
        cls.params = BakeParams(samples=16, workers=1, chunk_size=5)
        cls.base_output = bake_full(cls.base_scene, cls.base_layout, cls.params)
        cls.base_path = cls.directory / "base.snapshot"
        cls.base_snapshot = write_snapshot_atomic(
            cls.base_path,
            scene=cls.base_scene,
            layout=cls.base_layout,
            irradiances=cls.base_output.irradiances,
            evaluation=cls.base_output.evaluation,
            revision=1,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def _full_and_incremental(self, patch: dict):
        scene = apply_scene_patch(self.base_scene, patch)
        layout = make_layout(scene)
        full = bake_full(scene, layout, self.params)
        incremental = bake_incremental(scene, layout, self.params, self.base_snapshot)
        self.assertEqual(full.irradiances, incremental.irradiances)
        return scene, layout, full, incremental

    def test_geometry_change_recomputes_subset_and_matches_full(self) -> None:
        _scene, _layout, _full, incremental = self._full_and_incremental(
            {"geometry_updates": {"3": {"albedo": [0.92, 0.12, 0.07]}}}
        )
        self.assertGreater(incremental.affected_probe_count, 0)
        self.assertLess(incremental.affected_probe_count, len(incremental.layout.probes))
        self.assertGreater(incremental.reused_probe_count, 0)
        self.assertTrue(incremental.reuse_allowed)

    def test_point_light_change_recomputes_subset_and_matches_full(self) -> None:
        _scene, _layout, _full, incremental = self._full_and_incremental(
            {"point_light_updates": {"point-0": {"radiance": [55.0, 21.0, 6.0]}}}
        )
        self.assertGreater(incremental.affected_probe_count, 0)
        self.assertLess(incremental.affected_probe_count, len(incremental.layout.probes))

    def test_moved_geometry_matches_full(self) -> None:
        scene = apply_scene_patch(self.base_scene, {})
        box = scene.geometry[4]
        moved = (
            box.center[0] + 0.35,
            box.center[1] - 0.2,
            box.center[2] + 0.15,
        )
        _scene, _layout, _full, incremental = self._full_and_incremental(
            {"geometry_updates": {str(box.object_id): {"center": list(moved)}}}
        )
        self.assertGreater(incremental.affected_probe_count, 0)

    def test_emissive_change_matches_full(self) -> None:
        scene = self.base_scene
        emissive = next(box for box in scene.geometry if any(component > 0.0 for component in box.emission))
        _scene, _layout, _full, incremental = self._full_and_incremental(
            {
                "geometry_updates": {
                    str(emissive.object_id): {"emission": [2.4, 1.1, 0.3]}
                }
            }
        )
        self.assertGreater(incremental.affected_probe_count, 0)

    def test_no_change_reuses_every_probe(self) -> None:
        output = bake_incremental(self.base_scene, self.base_layout, self.params, self.base_snapshot)
        self.assertEqual(output.affected_probe_count, 0)
        self.assertEqual(output.reused_probe_count, len(output.layout.probes))
        self.assertEqual(output.irradiances, self.base_output.irradiances)

    def test_evaluation_parameter_change_invalidates_reuse(self) -> None:
        changed_params = BakeParams(samples=12, workers=1, chunk_size=5)
        full = bake_full(self.base_scene, self.base_layout, changed_params)
        incremental = bake_incremental(
            self.base_scene,
            self.base_layout,
            changed_params,
            self.base_snapshot,
        )
        self.assertFalse(incremental.reuse_allowed)
        self.assertEqual(incremental.affected_probe_count, len(self.base_layout.probes))
        self.assertEqual(full.irradiances, incremental.irradiances)

    def test_layout_parameter_change_invalidates_reuse(self) -> None:
        changed_layout = make_layout(self.base_scene, LayoutParams(spacing=1.5, cull_distance=0.4))
        full = bake_full(self.base_scene, changed_layout, self.params)
        incremental = bake_incremental(
            self.base_scene,
            changed_layout,
            self.params,
            self.base_snapshot,
        )
        self.assertFalse(incremental.reuse_allowed)
        self.assertEqual(incremental.affected_probe_count, len(changed_layout.probes))
        self.assertEqual(full.irradiances, incremental.irradiances)

    def test_combined_changes_match_full(self) -> None:
        _scene, _layout, _full, incremental = self._full_and_incremental(
            {
                "geometry_updates": {"2": {"albedo": [0.05, 0.55, 0.15]}},
                "point_light_updates": {"point-1": {"position": [1.0, -1.0, 3.0]}},
            }
        )
        self.assertTrue(incremental.reuse_allowed)

    def test_randomized_single_component_changes_match_full(self) -> None:
        rng = random.Random(4242)
        for trial in range(12):
            kind = rng.choice(("albedo", "emission", "move", "light", "point_move"))
            if kind in {"albedo", "emission", "move"}:
                box = rng.choice(self.base_scene.geometry[1:])
                if kind == "albedo":
                    fields = {"albedo": [rng.uniform(0.0, 0.9) for _ in range(3)]}
                elif kind == "emission":
                    fields = {"emission": [rng.uniform(0.1, 2.2) for _ in range(3)]}
                else:
                    fields = {
                        "center": [
                            box.center[0] + rng.uniform(-0.7, 0.7),
                            box.center[1] + rng.uniform(-0.7, 0.7),
                            max(box.half_size[2] + 0.1, box.center[2] + rng.uniform(-0.4, 0.4)),
                        ]
                    }
                patch = {"geometry_updates": {str(box.object_id): fields}}
            else:
                light = rng.choice(self.base_scene.point_lights)
                if kind == "light":
                    fields = {"radiance": [rng.uniform(4.0, 45.0) for _ in range(3)]}
                else:
                    fields = {
                        "position": [
                            rng.uniform(-4.0, 4.0),
                            rng.uniform(-3.0, 3.0),
                            rng.uniform(1.5, 3.5),
                        ]
                    }
                patch = {"point_light_updates": {light.light_id: fields}}
            scene = apply_scene_patch(self.base_scene, patch)
            layout = make_layout(scene)
            full = bake_full(scene, layout, self.params)
            incremental = bake_incremental(scene, layout, self.params, self.base_snapshot)
            self.assertEqual(full.irradiances, incremental.irradiances, f"trial {trial}: {kind}")

    def test_incremental_snapshot_round_trip_preserves_float_bits(self) -> None:
        scene, layout, full, incremental = self._full_and_incremental(
            {"geometry_updates": {"3": {"albedo": [0.11, 0.22, 0.33]}}}
        )
        output_path = self.directory / "incremental.snapshot"
        snapshot = write_snapshot_atomic(
            output_path,
            scene=scene,
            layout=layout,
            irradiances=incremental.irradiances,
            evaluation=incremental.evaluation,
            revision=self.base_snapshot.revision + 1,
        )
        loaded = load_snapshot(output_path)
        self.assertEqual(tuple(record.irradiance for record in loaded.probes), full.irradiances)
        self.assertEqual(loaded.revision, self.base_snapshot.revision + 1)


if __name__ == "__main__":
    unittest.main()

