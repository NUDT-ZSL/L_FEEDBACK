import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from offline_audio_render.renderer import Renderer
from offline_audio_render.scene import load_scene


class RenderSpatializationTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="uid49_spatial_render_"))
        sr = 48000
        frames = 4800
        t = np.arange(frames) / sr
        sf.write(
            self.dir / "a.wav",
            (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None],
            sr,
            subtype="FLOAT",
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def render_scene(self, events, layout="stereo", duration=0.3, name="out"):
        scene = {
            "sample_rate": 48000,
            "layout": layout,
            "global_seed": 1,
            "duration": duration,
            "trajectory": [
                {"time": 0.0, "position": [0.0, 0.0, 0.0], "yaw": 0.0}
            ],
            "events": events,
        }
        path = self.dir / "scene.yaml"
        path.write_text(yaml.safe_dump(scene))
        out_dir = self.dir / name
        return Renderer(load_scene(path), out_dir).render()

    def event(self, event_id, position):
        return {
            "id": event_id,
            "time": 0.1,
            "source": "a.wav",
            "position": position,
            "seed_offset": 1,
        }

    def test_hard_left_and_right_pan(self):
        left_result = self.render_scene(
            [self.event("left", [0.0, 2.0, 0.0])], name="left"
        )
        right_result = self.render_scene(
            [self.event("right", [0.0, -2.0, 0.0])], name="right"
        )
        left_data, _ = sf.read(left_result.output_files["master"], dtype="float32")
        right_data, _ = sf.read(
            right_result.output_files["master"], dtype="float32"
        )
        left_start = left_result.report["events"][0]["start_sample"]
        right_start = right_result.report["events"][0]["start_sample"]
        left_segment = left_data[left_start : left_start + 1024]
        right_segment = right_data[right_start : right_start + 1024]
        np.testing.assert_allclose(
            left_segment[:, 1], 0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            right_segment[:, 0], 0.0, atol=1e-5
        )
        self.assertGreater(np.max(np.abs(left_segment[:, 0])), 0.0)
        self.assertGreater(np.max(np.abs(right_segment[:, 1])), 0.0)

    def test_distance_attenuation_scales_with_inverse_square(self):
        near = self.render_scene(
            [self.event("near", [2.0, 0.0, 0.0])], name="near"
        )
        far = self.render_scene(
            [self.event("far", [4.0, 0.0, 0.0])], name="far"
        )
        near_data, _ = sf.read(near.output_files["master"], dtype="float32")
        far_data, _ = sf.read(far.output_files["master"], dtype="float32")
        start = near.report["events"][0]["start_sample"]
        end = min(start + 2048, near_data.shape[0])
        near_rms = np.sqrt(np.mean(near_data[start:end] ** 2))
        far_rms = np.sqrt(np.mean(far_data[start:end] ** 2))
        # Doubling distance quarters the signal power envelope gain.
        self.assertAlmostEqual(
            far_rms / max(near_rms, 1e-12), 0.25, delta=0.04
        )


if __name__ == "__main__":
    unittest.main()
