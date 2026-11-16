from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from terrain_preview.config import load_config
from terrain_preview.renderer import render
from tests.helpers import small_config, write_config


class IncrementalRenderTests(unittest.TestCase):
    def test_full_then_reuse_then_partial_material_redraw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            path = write_config(root, raw)
            config = load_config(path)
            first_output = io.StringIO()
            with redirect_stdout(first_output):
                first = render(config)
            self.assertEqual(first.redrawn_tiles, first.screen_tile_count)
            self.assertEqual(first.reused_tiles, 0)
            self.assertTrue((root / "out" / "preview.png").is_file())
            self.assertTrue((root / "out" / "normal.png").is_file())
            self.assertTrue((root / "out" / "shadow.png").is_file())
            self.assertTrue((root / "out" / "terrain.bin").is_file())
            self.assertTrue((root / "out" / "tiles.json").is_file())

            with Image.open(root / "out" / "preview.png") as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (192, 144))
                preview = np.array(image)
            with Image.open(root / "out" / "normal.png") as image:
                normal = np.array(image)
            with Image.open(root / "out" / "shadow.png") as image:
                self.assertEqual(image.mode, "L")
                shadow = np.array(image)
            self.assertGreater(len(np.unique(preview.reshape(-1, 3), axis=0)), 100)
            self.assertGreater(len(np.unique(normal.reshape(-1, 3), axis=0)), 100)
            self.assertGreater(int(shadow.max()), 240)
            self.assertLess(int(shadow.min()), 80)
            self.assertGreaterEqual(len(np.unique(shadow)), 5)

            terrain_bytes = (root / "out" / "terrain.bin").read_bytes()
            report = json.loads((root / "out" / "tiles.json").read_text())
            self.assertGreater(len(report["visible_world_tiles"]), 0)
            self.assertEqual(report["screen_tiles"]["redrawn_tiles"], first.redrawn_tiles)

            second_output = io.StringIO()
            with redirect_stdout(second_output):
                second = render(load_config(path))
            self.assertEqual(second.redrawn_tiles, 0)
            self.assertEqual(second.reused_tiles, second.screen_tile_count)
            self.assertTrue(second.shadow_reused)
            self.assertEqual((root / "out" / "terrain.bin").read_bytes(), terrain_bytes)
            self.assertIn("redrawn_tiles=0", second_output.getvalue())

            changed = json.loads(json.dumps(raw))
            changed["materials"]["layers"][3]["color"] = [0.70, 0.82, 0.98]
            path = write_config(root, changed)
            third_output = io.StringIO()
            with redirect_stdout(third_output):
                third = render(load_config(path))
            self.assertGreater(third.redrawn_tiles, 0)
            self.assertLess(third.redrawn_tiles, third.screen_tile_count)
            self.assertGreater(third.reused_tiles, 0)
            self.assertTrue(third.shadow_reused)

            manifest = json.loads((root / "out" / "cache" / "manifest.json").read_text())
            entries = list(manifest["screen_tiles"].values())
            self.assertTrue(entries)
            for entry in entries:
                self.assertIn("lighting_version", entry)
                self.assertIn("height_versions", entry)
                self.assertIn("material_versions", entry)

    def test_local_height_patch_redraws_only_affected_tiles_with_shadows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            raw["light"]["direction"]["elevation_deg"] = 72.0
            path = write_config(root, raw)
            with redirect_stdout(io.StringIO()):
                first = render(load_config(path))
            self.assertEqual(first.redrawn_tiles, first.screen_tile_count)

            raw["terrain"]["height_patches"] = [
                {"center": [0.5, 0.5], "radius": 0.12, "delta": 0.15}
            ]
            path = write_config(root, raw)
            output = io.StringIO()
            with redirect_stdout(output):
                second = render(load_config(path))
            self.assertGreater(second.redrawn_tiles, 0)
            self.assertLess(second.redrawn_tiles, second.screen_tile_count)
            self.assertGreater(second.reused_tiles, 0)
            self.assertTrue(second.shadow_generated)
            self.assertFalse(second.shadow_reused)
            self.assertGreater(second.shadow_check_tiles, 0)
            self.assertGreater(second.shadow_check_samples, 0)
            self.assertIn("shadow_invalidation", output.getvalue())


class MemoryChunkTests(unittest.TestCase):
    def test_low_memory_limit_uses_column_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = small_config()
            raw["terrain"].update(
                {
                    "rows": 24,
                    "cols": 24,
                    "tile_cells": 8,
                    "world_width": 60.0,
                    "world_depth": 60.0,
                }
            )
            raw["camera"].update({"distance": 130.0, "fov_y_deg": 55.0})
            raw["render"].update(
                {"width": 512, "height": 512, "tile_size": 256}
            )
            raw["light"]["shadow"].update(
                {"enabled": False, "resolution": 64, "pcf_samples": 1}
            )
            raw["limits"].update(
                {"max_memory_mb": 4.0, "max_total_samples": 10_000_000}
            )
            path = write_config(root, raw)
            output = io.StringIO()
            with redirect_stdout(output):
                stats = render(load_config(path))
            self.assertTrue(stats.memory_chunked)
            self.assertEqual(stats.batch_columns, 1)
            self.assertGreater(stats.redrawn_tiles, 1)
            self.assertIn("memory_notice", output.getvalue())
            with Image.open(root / "out" / "preview.png") as image:
                self.assertEqual(image.size, (512, 512))


if __name__ == "__main__":
    unittest.main()

