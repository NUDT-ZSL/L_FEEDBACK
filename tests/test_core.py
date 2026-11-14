from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from fogcity.errors import KEYFRAME_TIME_DUPLICATE, KEYFRAME_TIME_ORDER, SCENE_FIELD
from fogcity.interpolation import CameraState, slerp
from fogcity.ordering import order_panels
from fogcity.runner import iter_frames
from fogcity.sample import build_sample_document
from fogcity.scene import GlassPanel, validate_scene_object
from fogcity.util import quantize


def _sample():
    return build_sample_document(
        width=64,
        height=36,
        fps=1.0,
        start_time=0.0,
        end_time=2.0,
        fog_steps=8,
        shadow_resolution=128,
    )


class CoreTests(unittest.TestCase):
    def test_sample_scene_is_valid(self) -> None:
        scene = validate_scene_object(_sample(), "memory", "sha")
        self.assertEqual(len(scene.frame_times), 3)
        self.assertEqual(len(scene.buildings), 8)
        self.assertEqual(len(scene.glass_panels), 5)

    def test_duplicate_keyframe_time_has_business_code(self) -> None:
        document = _sample()
        document["camera_keyframes"][2]["time"] = 2.0
        with self.assertRaises(Exception) as caught:
            validate_scene_object(document, "memory", "sha")
        self.assertEqual(caught.exception.code, KEYFRAME_TIME_DUPLICATE)  # type: ignore[attr-defined]

    def test_non_monotonic_keyframe_time_has_business_code(self) -> None:
        document = _sample()
        document["camera_keyframes"][2]["time"] = 0.5
        with self.assertRaises(Exception) as caught:
            validate_scene_object(document, "memory", "sha")
        self.assertEqual(caught.exception.code, KEYFRAME_TIME_ORDER)  # type: ignore[attr-defined]

    def test_invalid_field_has_business_code(self) -> None:
        document = _sample()
        document["render"]["width"] = "wide"
        with self.assertRaises(Exception) as caught:
            validate_scene_object(document, "memory", "sha")
        self.assertEqual(caught.exception.code, SCENE_FIELD)  # type: ignore[attr-defined]
        self.assertEqual(caught.exception.context["field"], "render.width")  # type: ignore[attr-defined]

    def test_slerp_is_shortest_path_and_normalized(self) -> None:
        half = math.sqrt(0.5)
        q0 = (0.0, 0.0, 0.0, 1.0)
        q1 = (0.0, half, 0.0, half)
        middle = slerp(q0, q1, 0.5)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in middle)), 1.0, places=12)
        self.assertAlmostEqual(middle[1], math.sin(math.pi / 8.0), places=12)
        self.assertAlmostEqual(middle[3], math.cos(math.pi / 8.0), places=12)

        q1_negative = tuple(-value for value in q1)
        self.assertEqual(slerp(q0, q1_negative, 0.5), middle)

    def test_panel_sort_uses_id_as_tie_breaker(self) -> None:
        normal = (0.0, 0.0, 1.0)
        panels = (
            GlassPanel(
                id=300,
                color=(1.0, 0.0, 0.0),
                transmittance=0.5,
                vertices=np.asarray(
                    [[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [1.0, 1.0, 5.0], [-1.0, 1.0, 5.0]]
                ),
                normal=normal,
                center=(0.0, 0.0, 5.0),
            ),
            GlassPanel(
                id=200,
                color=(0.0, 1.0, 0.0),
                transmittance=0.5,
                vertices=np.asarray(
                    [[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [1.0, 1.0, 5.0], [-1.0, 1.0, 5.0]]
                ),
                normal=normal,
                center=(0.0, 0.0, 5.0),
            ),
        )
        camera = CameraState(
            frame_index=0,
            time=0.0,
            from_keyframe_index=0,
            to_keyframe_index=0,
            alpha=0.0,
            position=(0.0, 0.0, 10.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            forward=(0.0, 0.0, -1.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_y_deg=50.0,
            near=0.1,
            far=50.0,
        )
        orders = order_panels(SimpleNamespace(glass_panels=panels), camera)  # type: ignore[arg-type]
        self.assertEqual([order.panel_id for order in orders], [200, 300])
        self.assertTrue(all(order.front_facing for order in orders))
        self.assertTrue(all(order.rendered for order in orders))

    def test_report_quantization_is_fixed(self) -> None:
        self.assertEqual(quantize(1.23456789), 1.234568)
        self.assertEqual(quantize(-0.0000001), 0.0)

    def test_serial_frame_failure_reports_index(self) -> None:
        scene = validate_scene_object(_sample(), "memory", "sha")

        class FailingPipeline:
            def __init__(self, scene: object, backend: str):
                pass

            def render_frame(self, frame_index: int):
                if frame_index == 1:
                    raise RuntimeError("synthetic frame failure")
                return SimpleNamespace(frame_index=frame_index)

            def close(self) -> None:
                pass

        with mock.patch("fogcity.runner.RenderPipeline", FailingPipeline):
            iterator = iter_frames(scene, backend="auto", parallel_frames=1)
            try:
                next(iterator)
                with self.assertRaises(Exception) as caught:
                    next(iterator)
                self.assertEqual(caught.exception.context["frame_index"], 1)  # type: ignore[attr-defined]
                self.assertEqual(caught.exception.code, "E_FRAME_RENDER")  # type: ignore[attr-defined]
            finally:
                iterator.close()


if __name__ == "__main__":
    unittest.main()

