from __future__ import annotations

import unittest
from dataclasses import replace

from proc_gi.clustering import replace_vpl_signature
from proc_gi.cubemap_cache import VPLCubeScheduler
from proc_gi.scene_generator import generate_scene


class CubemapCacheTests(unittest.TestCase):
    def test_reuse_change_and_release(self) -> None:
        config, _ = generate_scene()
        from proc_gi.clustering import cluster_lights

        vpls = cluster_lights(config.lights, config.clustering)
        scheduler: VPLCubeScheduler[dict[str, str]] = VPLCubeScheduler()
        created: list[str] = []
        released: list[str] = []

        def create(vpl):
            resource = {"id": vpl.id, "signature": vpl.signature}
            created.append(vpl.id)
            return resource

        initial = scheduler.sync(vpls, create)
        self.assertEqual(len(initial.added), len(vpls))
        self.assertEqual(len(created), len(vpls))

        unchanged = scheduler.sync(vpls, create)
        self.assertEqual(unchanged.added, [])
        self.assertEqual(unchanged.changed, [])
        self.assertEqual(len(unchanged.reused), len(vpls))

        changed_vpl = replace_vpl_signature(vpls[0], "changed-version")
        modified = list(vpls)
        modified[0] = changed_vpl
        result = scheduler.sync(modified, create)
        self.assertEqual(result.changed, [vpls[0].id])
        self.assertEqual(len(result.stale_resources), 1)
        result.release_stale(lambda resource: released.append(resource["id"]))
        self.assertEqual(released, [vpls[0].id])

        removed = list(modified[1:])
        result = scheduler.sync(removed, create)
        self.assertEqual(result.removed, [vpls[0].id])
        result.release_stale(lambda resource: released.append(resource["id"]))
        self.assertIn(vpls[0].id, released)

    def test_unchanged_resource_is_not_released(self) -> None:
        config, _ = generate_scene()
        from proc_gi.clustering import cluster_lights

        vpls = cluster_lights(config.lights, config.clustering)
        scheduler: VPLCubeScheduler[object] = VPLCubeScheduler()
        first = scheduler.sync(vpls, lambda _vpl: object())
        released: list[object] = []
        first.release_stale(released.append)
        self.assertEqual(released, [])


if __name__ == "__main__":
    unittest.main()

