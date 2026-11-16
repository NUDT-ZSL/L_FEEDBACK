from __future__ import annotations

import unittest

import numpy as np

from proc_gi.clustering import (
    cluster_lights,
    rank_vpl_importance,
    truncation_warning,
)
from proc_gi.mathutil import look_at, perspective
from proc_gi.scene_generator import generate_scene


class ClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.mesh = generate_scene()
        self.vpls = cluster_lights(self.config.lights, self.config.clustering)

    def test_clusters_cover_every_light_and_respect_limit(self) -> None:
        self.assertLessEqual(len(self.vpls), self.config.clustering.max_vpls)
        members = [member for vpl in self.vpls for member in vpl.member_ids]
        self.assertEqual(sorted(members), sorted(light.id for light in self.config.lights))
        self.assertEqual(sum(vpl.source_count for vpl in self.vpls), len(self.config.lights))
        self.assertTrue(all(vpl.signature for vpl in self.vpls))

    def test_incremental_clustering_preserves_unaffected_signatures(self) -> None:
        changed = self.config.lights[0]
        payload = changed.to_dict()
        payload["intensity"] += 7.0
        from proc_gi.config import Light

        self.config.lights[0] = Light.from_dict(payload, 0)
        updated = cluster_lights(
            self.config.lights,
            self.config.clustering,
            previous=self.vpls,
            changed_ids={changed.id},
        )
        old_signatures = {vpl.signature for vpl in self.vpls}
        new_signatures = {vpl.signature for vpl in updated}
        self.assertGreaterEqual(len(old_signatures & new_signatures), len(self.vpls) - 2)
        self.assertTrue(new_signatures - old_signatures)
        self.assertLessEqual(len(updated), self.config.clustering.max_vpls)

    def test_importance_truncation_and_warning_payload(self) -> None:
        camera = self.config.camera
        view_proj = perspective(
            camera["fov_y_deg"],
            16.0 / 9.0,
            camera["near"],
            camera["far"],
        ) @ look_at(
            np.asarray(camera["position"]),
            np.asarray(camera["target"]),
            np.asarray(camera["up"]),
        )
        result = rank_vpl_importance(
            self.vpls,
            capacity=8,
            view_proj=view_proj,
            camera_position=np.asarray(camera["position"]),
            scene_triangles=self.mesh.triangles(),
        )
        self.assertEqual(len(result.accepted), 8)
        self.assertEqual(len(result.rejected), len(self.vpls) - 8)
        self.assertTrue(all("occlusion_rate" in score for score in result.scores.values()))
        warning = truncation_warning(len(self.vpls), result, 8)
        self.assertEqual(warning["code"], "VPL_CAPACITY_TRUNCATED")
        self.assertTrue(warning["context"]["truncated_source_light_ids"])
        self.assertTrue(warning["context"]["degraded_state"]["render_continues"])


if __name__ == "__main__":
    unittest.main()

