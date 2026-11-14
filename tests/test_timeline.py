from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

from immersive_audio.scene import load_scene
from immersive_audio.timeline import (
    block_count,
    iter_metadata_blocks,
    metadata_array,
    seconds_to_frames,
)

from tests.fixtures import audio_object, keyframe, write_scene


class TimelineTests(unittest.TestCase):
    def test_half_up_frame_rounding(self) -> None:
        self.assertEqual(seconds_to_frames(Decimal("0.25"), 2), 1)
        self.assertEqual(seconds_to_frames(Decimal("0.249999"), 2), 0)
        self.assertEqual(seconds_to_frames(Decimal("0.75"), 2), 2)
        self.assertEqual(block_count(10, 4), 3)
        self.assertEqual(block_count(8, 4), 2)
        self.assertEqual(block_count(0, 4), 0)

    def test_linear_interpolation_and_frame_alignment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iapano-timeline-") as temporary:
            path = write_scene(
                Path(temporary) / "scene.json",
                [
                    audio_object(
                        "a",
                        [
                            keyframe(0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
                            keyframe(1.0, 90.0, 30.0, 2.0, 0.5, 1.0),
                        ],
                    )
                ],
                sample_rate=48_000,
                frame_length=4_800,
                duration=1.0,
            )
            scene = load_scene(path)
            metadata = metadata_array(scene, 48_000)
            self.assertEqual(metadata.shape, (10, 1, 5))
            np.testing.assert_allclose(metadata[:, 0, 0], np.arange(10) * 9.0)
            np.testing.assert_allclose(metadata[:, 0, 1], np.arange(10) * 3.0)
            np.testing.assert_allclose(metadata[:, 0, 2], 1.0 + np.arange(10) * 0.1)
            np.testing.assert_allclose(metadata[:, 0, 3], 1.0 - np.arange(10) * 0.05)
            np.testing.assert_allclose(metadata[9, 0, 4], 0.9)

    def test_one_metadata_block_per_audio_block(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iapano-timeline-") as temporary:
            path = write_scene(
                Path(temporary) / "scene.json",
                [audio_object("a", [keyframe(0.0)])],
                sample_rate=8_000,
                frame_length=1_000,
                duration=0.375,
            )
            scene = load_scene(path)
            total_samples = 3_000
            blocks = list(iter_metadata_blocks(scene, total_samples))
            self.assertEqual(len(blocks), 3)
            self.assertEqual(len(blocks), block_count(total_samples, scene.frame_length))

    def test_endpoint_clamping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iapano-timeline-") as temporary:
            path = write_scene(
                Path(temporary) / "scene.json",
                [
                    audio_object(
                        "a",
                        [
                            keyframe(0.5, 10.0, 0.0, 1.0, 0.25),
                            keyframe(1.0, 20.0, 0.0, 1.0, 0.5),
                        ],
                    )
                ],
                sample_rate=8_000,
                frame_length=800,
                duration=2.0,
            )
            scene = load_scene(path)
            metadata = metadata_array(scene, 16_000)
            np.testing.assert_allclose(metadata[:5, 0, 0], 10.0)
            np.testing.assert_allclose(metadata[-1, 0, 0], 20.0)
            np.testing.assert_allclose(metadata[:5, 0, 3], 0.25)
            np.testing.assert_allclose(metadata[-1, 0, 3], 0.5)
