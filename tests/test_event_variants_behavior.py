import unittest
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import yaml

from offline_audio_render.random_engine import derive_variants
from offline_audio_render.renderer import Renderer
from offline_audio_render.scene import load_scene


class StartOffsetTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="uid49_offsets_"))
        sr = 48000
        frames = 4800
        t = np.arange(frames) / sr
        sf.write(
            self.dir / "a.wav",
            (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None],
            sr,
            subtype="FLOAT",
        )

    def test_negative_start_offset_can_begin_at_timeline_zero(self):
        variants = derive_variants(global_seed=123, seed_offset=9)
        # Find an offset that moves the event before the timeline origin.
        self.assertLess(variants.start_offset_s, 0.0)
        scene = {
            "sample_rate": 48000,
            "layout": "mono",
            "global_seed": 123,
            "duration": 0.2,
            "events": [
                {
                    "id": "early",
                    "time": 0.005,
                    "source": "a.wav",
                    "position": [1.0, 0.0, 0.0],
                    "seed_offset": 9,
                }
            ],
        }
        path = self.dir / "scene.yaml"
        path.write_text(yaml.safe_dump(scene))
        result = Renderer(load_scene(path), self.dir / "out").render()
        data, _ = sf.read(result.output_files["master"], dtype="float32")
        event = result.report["events"][0]
        self.assertEqual(event["start_sample"], 0)
        self.assertTrue(np.any(np.abs(data) > 1e-6))

    def test_positive_start_offset_moves_event_later(self):
        # Search for a deterministic offset producing a positive shift.
        offset = None
        for candidate in range(100):
            variants = derive_variants(global_seed=55, seed_offset=candidate)
            if variants.start_offset_s > 0.01:
                offset = candidate
                break
        self.assertIsNotNone(offset)
        scene = {
            "sample_rate": 48000,
            "layout": "mono",
            "global_seed": 55,
            "duration": 0.2,
            "events": [
                {
                    "id": "late",
                    "time": 0.05,
                    "source": "a.wav",
                    "position": [1.0, 0.0, 0.0],
                    "seed_offset": offset,
                }
            ],
        }
        path = self.dir / "scene.yaml"
        path.write_text(yaml.safe_dump(scene))
        result = Renderer(load_scene(path), self.dir / "out").render()
        event = result.report["events"][0]
        self.assertGreater(
            event["start_sample"], int(0.05 * 48000)
        )


if __name__ == "__main__":
    unittest.main()
