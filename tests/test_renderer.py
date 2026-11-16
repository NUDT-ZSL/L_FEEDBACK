from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from strip_renderer.cli import main
from strip_renderer.demo import write_demo
from strip_renderer.errors import RenderError
from strip_renderer.output import write_render_outputs
from strip_renderer.report import (
    CHANNEL_FILES,
    REPORT_FILE,
    quantize_channel,
    quantize_rgb,
)
from strip_renderer.sampling import sample_bilinear, sample_nearest
from strip_renderer.scene import load_scene
from strip_renderer.texture import load_textures


class RendererTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.demo = self.root / "demo"
        write_demo(self.demo, size=64, texture_size=16)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _payload(self, scene_name: str) -> dict:
        path = self.demo / scene_name
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_variant(
        self,
        name: str,
        payload: dict,
    ) -> Path:
        path = self.demo / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _render_path(self, scene_path: Path):
        scene = load_scene(scene_path)
        textures = load_textures(scene.material)
        from strip_renderer.render import render_scene

        return scene, render_scene(scene, textures)

    def _run_cli(self, scene_path: Path) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--scene", str(scene_path)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_composite_is_exact_weighted_sum(self) -> None:
        scene, result = self._render_path(
            self.demo / "scene_bilinear_32_row_forward.json"
        )
        weights = scene.output.weights
        expected = (
            result.base_color * weights.base_color
            + result.direct_light * weights.direct
            + result.ambient_light * weights.ambient
        )
        self.assertLessEqual(float(np.max(np.abs(result.composite - expected))), 1e-12)
        self.assertTrue(np.all(np.isfinite(result.composite)))

    def test_pixel_buffers_and_background(self) -> None:
        scene, result = self._render_path(
            self.demo / "scene_bilinear_32_row_forward.json"
        )
        self.assertGreater(result.hit_pixels, 0)
        self.assertFalse(np.any(result.buffers.hit_mask & ~np.isfinite(
            result.buffers.world_position
        ).all(axis=-1)))
        miss = ~result.buffers.hit_mask
        background = np.asarray(scene.output.background_color)
        for channel in (
            result.base_color,
            result.direct_light,
            result.ambient_light,
            result.composite,
        ):
            self.assertLessEqual(
                float(np.max(np.abs(channel[miss] - background))),
                1e-12,
            )
        self.assertAlmostEqual(
            float(np.mean(np.linalg.norm(result.buffers.world_normal[hit := result.buffers.hit_mask], axis=-1))),
            1.0,
        )
        self.assertTrue(np.all(result.buffers.uv[hit] >= 0.0))
        self.assertTrue(np.all(result.buffers.uv[hit] <= 1.0))

    def test_sampling_definitions_and_edge_clamp(self) -> None:
        texture = np.asarray([[[0.0], [1.0]]], dtype=np.float64)
        u = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        nearest = sample_nearest(texture, u, u)[..., 0]
        bilinear = sample_bilinear(texture, u, u)[..., 0]
        np.testing.assert_array_equal(nearest, np.asarray([0.0, 1.0, 1.0]))
        np.testing.assert_allclose(bilinear, np.asarray([0.0, 0.5, 1.0]))

        texture_2d = np.asarray(((0, 1), (2, 3)), dtype=np.float64)
        outside_u = np.asarray([-2.0, 2.0])
        outside_v = np.asarray([-2.0, 2.0])
        clamped = sample_bilinear(texture_2d, outside_u, outside_v)
        np.testing.assert_array_equal(clamped, np.asarray([0.0, 3.0]))

    def test_nearest_and_bilinear_differ_deterministically(self) -> None:
        _, bilinear_result = self._render_path(
            self.demo / "scene_bilinear_32_row_forward.json"
        )
        _, nearest_result = self._render_path(
            self.demo / "scene_nearest_32_row_forward.json"
        )
        hit = bilinear_result.buffers.hit_mask
        difference = np.abs(
            bilinear_result.base_color[hit] - nearest_result.base_color[hit]
        )
        self.assertGreater(float(np.max(difference)), 1e-6)
        self.assertGreater(nearest_result.hit_pixels, 0)

    def test_block_sizes_axes_and_order_are_pixel_identical(self) -> None:
        _, result32 = self._render_path(
            self.demo / "scene_bilinear_32_row_forward.json"
        )
        payload = self._payload("scene_bilinear_32_row_forward.json")
        payload["output"]["block_size"] = 64
        payload["output"]["block_axis"] = "row"
        payload["output"]["block_order"] = "reverse"
        payload["output"]["directory"] = "outputs/whole_reverse"
        whole_path = self._write_variant("scene_whole_reverse.json", payload)
        _, result_whole = self._render_path(whole_path)
        _, result16 = self._render_path(
            self.demo / "scene_bilinear_16_column_reverse.json"
        )

        for other in (result_whole, result16):
            np.testing.assert_array_equal(result32.buffers.hit_mask, other.buffers.hit_mask)
            np.testing.assert_array_equal(
                result32.buffers.world_position,
                other.buffers.world_position,
            )
            np.testing.assert_array_equal(
                result32.buffers.world_normal,
                other.buffers.world_normal,
            )
            np.testing.assert_array_equal(
                result32.buffers.shading_normal,
                other.buffers.shading_normal,
            )
            np.testing.assert_array_equal(result32.buffers.uv, other.buffers.uv)
            np.testing.assert_array_equal(result32.base_color, other.base_color)
            np.testing.assert_array_equal(result32.direct_light, other.direct_light)
            np.testing.assert_array_equal(result32.ambient_light, other.ambient_light)
            np.testing.assert_array_equal(result32.composite, other.composite)
            self.assertEqual(result32.normal_fallback_count, other.normal_fallback_count)

    def test_png_quantization_uses_ties_to_even(self) -> None:
        values = np.asarray(
            [0.5, 1.5, 2.5, 3.5, 254.5, 255.0],
            dtype=np.float64,
        ) / 255.0
        quantized = quantize_channel(values)
        np.testing.assert_array_equal(
            quantized,
            np.asarray([0, 2, 2, 4, 254, 255], dtype=np.uint8),
        )

    def test_cli_success_writes_all_outputs_and_report(self) -> None:
        scene_path = self.demo / "scene_bilinear_32_row_forward.json"
        code, stdout, stderr = self._run_cli(scene_path)
        self.assertEqual(code, 0, stderr)
        self.assertIn("rendered 64x64", stdout)
        payload = self._payload("scene_bilinear_32_row_forward.json")
        output_dir = self.demo / payload["output"]["directory"]
        expected = set(CHANNEL_FILES.values()) | {REPORT_FILE}
        self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
        for filename in CHANNEL_FILES.values():
            with Image.open(output_dir / filename) as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.mode, "RGB")
        report = json.loads((output_dir / REPORT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "ok")
        self.assertGreater(report["render"]["hit_pixels"], 0)
        self.assertEqual(report["render"]["nan_count"], 0)
        self.assertEqual(report["render"]["inf_count"], 0)
        self.assertTrue(report["render"]["composite_check"]["passed"])
        scene, result = self._render_path(scene_path)
        expected_arrays = {
            "base_color": quantize_rgb(result.base_color),
            "direct": quantize_rgb(result.direct_light),
            "ambient": quantize_rgb(result.ambient_light),
            "composite": quantize_rgb(result.composite),
        }
        for key, filename in CHANNEL_FILES.items():
            with Image.open(output_dir / filename) as image:
                actual = np.asarray(image, dtype=np.uint8)
            np.testing.assert_array_equal(actual, expected_arrays[key])

    def test_rerun_overwrites_channels_without_staging_leftovers(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        scene_path = self._write_variant(
            "scene_rerun.json",
            payload,
        )
        code, _, stderr = self._run_cli(scene_path)
        self.assertEqual(code, 0, stderr)
        output_dir = self.demo / payload["output"]["directory"]
        channel_path = output_dir / CHANNEL_FILES["base_color"]
        old_bytes = channel_path.read_bytes()
        channel_path.write_bytes(b"stale")
        code, _, stderr = self._run_cli(scene_path)
        self.assertEqual(code, 0, stderr)
        self.assertNotEqual(channel_path.read_bytes(), b"stale")
        self.assertEqual(channel_path.read_bytes(), old_bytes)
        self.assertFalse(
            any(path.name.startswith(".strip_renderer") for path in output_dir.iterdir())
        )

    def test_missing_normal_texture_falls_back_and_counts(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        payload["material"]["normal_texture"] = "textures/does-not-exist.png"
        payload["output"]["directory"] = "outputs/missing_normal"
        scene_path = self._write_variant("scene_missing_normal.json", payload)
        scene, result = self._render_path(scene_path)
        self.assertGreater(result.hit_pixels, 0)
        self.assertEqual(result.normal_fallback_count, result.hit_pixels)
        self.assertEqual(
            result.normal_fallback_breakdown["missing_texture"],
            result.hit_pixels,
        )
        self.assertTrue(np.all(np.isfinite(result.composite)))

    def test_zero_tangent_normal_falls_back_without_nonfinite_values(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        zero_normal = self.demo / "textures" / "zero_normal.png"
        Image.new("RGB", (16, 16), (128, 128, 128)).save(zero_normal)
        payload["material"]["normal_texture"] = "textures/zero_normal.png"
        payload["output"]["directory"] = "outputs/zero_normal"
        scene_path = self._write_variant("scene_zero_normal.json", payload)
        _, result = self._render_path(scene_path)
        self.assertEqual(result.normal_fallback_count, result.hit_pixels)
        self.assertEqual(
            result.normal_fallback_breakdown["zero_length"],
            result.hit_pixels,
        )
        self.assertTrue(np.all(np.isfinite(result.buffers.world_normal)))
        self.assertTrue(np.all(np.isfinite(result.composite)))

    def test_extreme_normal_components_remain_finite(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        extreme_normal = self.demo / "textures" / "extreme_normal.png"
        Image.new("RGB", (16, 16), (0, 0, 0)).save(extreme_normal)
        payload["material"]["normal_texture"] = "textures/extreme_normal.png"
        payload["material"]["normal_strength"] = 4.0
        payload["output"]["directory"] = "outputs/extreme_normal"
        scene_path = self._write_variant("scene_extreme_normal.json", payload)
        _, result = self._render_path(scene_path)
        for array in (
            result.base_color,
            result.direct_light,
            result.ambient_light,
            result.composite,
            result.buffers.world_normal,
            result.buffers.shading_normal,
        ):
            self.assertTrue(np.all(np.isfinite(array)))
        self.assertTrue(np.all(result.buffers.uv >= 0.0))
        self.assertTrue(np.all(result.buffers.uv <= 1.0))

    def test_scene_with_no_hits_is_pure_background(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        payload["camera"]["position"] = [0.0, 0.0, 4.0]
        payload["camera"]["target"] = [0.0, 1.0, 4.0]
        payload["camera"]["up"] = [0.0, 0.0, -1.0]
        payload["output"]["directory"] = "outputs/no_hits"
        scene_path = self._write_variant("scene_no_hits.json", payload)
        scene, result = self._render_path(scene_path)
        self.assertEqual(result.hit_pixels, 0)
        background = np.asarray(scene.output.background_color)
        for channel in (
            result.base_color,
            result.direct_light,
            result.ambient_light,
            result.composite,
        ):
            self.assertLessEqual(
                float(np.max(np.abs(channel - background))),
                1e-12,
            )

    def _assert_business_failure(
        self,
        scene_path: Path,
        expected_code: str,
        expected_exit: int,
    ) -> str:
        code, _, stderr = self._run_cli(scene_path)
        self.assertEqual(code, expected_exit, stderr)
        self.assertIn(expected_code, stderr)
        return stderr

    def test_missing_scene_field_exits_two_and_writes_report(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        del payload["camera"]["target"]
        payload["output"]["directory"] = "outputs/field_error"
        scene_path = self._write_variant("scene_missing_field.json", payload)
        self._assert_business_failure(scene_path, "E_SCENE_FIELD", 2)
        output_dir = self.demo / payload["output"]["directory"]
        self.assertFalse((output_dir / CHANNEL_FILES["base_color"]).exists())
        report = json.loads((output_dir / REPORT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_code"], "E_SCENE_FIELD")
        self.assertEqual(report["exit_code"], 2)

    def test_invalid_texture_size_exits_three_without_channels(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        payload["output"]["directory"] = "outputs/texture_size_error"
        bad_texture = self.demo / "textures" / "tiny.png"
        Image.new("RGB", (1, 1), (255, 0, 0)).save(bad_texture)
        payload["material"]["base_color_texture"] = "textures/tiny.png"
        scene_path = self._write_variant("scene_texture_size_error.json", payload)
        self._assert_business_failure(scene_path, "E_TEXTURE_SIZE", 3)
        output_dir = self.demo / payload["output"]["directory"]
        self.assertFalse(any((output_dir / name).exists() for name in CHANNEL_FILES.values()))

    def test_camera_inside_sphere_exits_three(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        payload["camera"]["position"] = [0.0, 0.0, 0.0]
        payload["output"]["directory"] = "outputs/inside_error"
        scene_path = self._write_variant("scene_inside_error.json", payload)
        self._assert_business_failure(scene_path, "E_CAMERA_INSIDE_SPHERE", 3)

    def test_block_not_divisible_exits_three(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        payload["output"]["block_size"] = 24
        payload["output"]["directory"] = "outputs/block_error"
        scene_path = self._write_variant("scene_block_error.json", payload)
        self._assert_business_failure(scene_path, "E_BLOCK_NOT_DIVISIBLE", 3)

    def test_unwritable_output_exits_three_without_channels(self) -> None:
        payload = self._payload("scene_bilinear_32_row_forward.json")
        blocker = self.demo / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        payload["output"]["directory"] = "blocker/output"
        scene_path = self._write_variant("scene_unwritable.json", payload)
        stderr = self._assert_business_failure(
            scene_path,
            "E_OUTPUT_NOT_WRITABLE",
            3,
        )
        self.assertFalse((self.demo / "blocker" / "output").exists())
        self.assertIn("render_error_report.json", stderr)

    def test_commit_failure_rolls_back_previous_outputs(self) -> None:
        scene = load_scene(self.demo / "scene_bilinear_32_row_forward.json")
        textures = load_textures(scene.material)
        from strip_renderer.render import render_scene

        result = render_scene(scene, textures)
        write_render_outputs(scene, result)
        output_dir = scene.output.directory
        old_hashes = {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in CHANNEL_FILES.values()
        }
        real_replace = os.replace

        def failing_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.name == CHANNEL_FILES["direct"]
                and destination_path.parent == output_dir
            ):
                raise OSError("injected commit failure")
            return real_replace(source, destination)

        with mock.patch("strip_renderer.output.os.replace", side_effect=failing_replace):
            with self.assertRaises(RenderError) as caught:
                write_render_outputs(scene, result)
        self.assertEqual(caught.exception.code, "E_OUTPUT_NOT_WRITABLE")
        for name, expected_hash in old_hashes.items():
            actual_hash = hashlib.sha256(
                (output_dir / name).read_bytes()
            ).hexdigest()
            self.assertEqual(actual_hash, expected_hash, name)
        self.assertFalse(
            any(path.name.startswith(".strip_renderer") for path in output_dir.iterdir())
        )


if __name__ == "__main__":
    unittest.main()
