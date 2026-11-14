from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from generate_sample import sample_scene
from render_pipeline.batch import render_batch
from render_pipeline.scene import LightSpec
from render_pipeline.tonemap import linear_to_srgb8, tonemap_linear


ROOT = Path(__file__).resolve().parents[1]


def _write_scene(directory: Path, scene: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scene.json"
    path.write_text(
        json.dumps(scene, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _light(operator: str = "aces") -> LightSpec:
    return LightSpec(
        id="unit",
        key_direction=(0.0, 1.0, 0.0),
        key_irradiance=(1.0, 1.0, 1.0),
        ambient_irradiance=(0.0, 0.0, 0.0),
        tone_operator=operator,
        exposure=1.0,
        white_point=4.0,
    )


class PipelineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="uid48-tests-")
        cls.tmp = Path(cls._tmp.name)
        cls.scene = sample_scene()
        cls.scene_path = _write_scene(cls.tmp / "input", cls.scene)
        cls.output_a = cls.tmp / "output_a"
        cls.result_a = render_batch(cls.scene_path, cls.output_a)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_sample_batch_produces_all_cards_and_metadata(self) -> None:
        self.assertEqual(self.result_a.exit_code, 0)
        self.assertEqual(self.result_a.manifest["status"], "success")
        diagnostics = json.loads((self.output_a / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["summary"]["diagnostic_count"], 0)
        presets = [item["id"] for item in self.scene["lighting_presets"]]
        cameras = [item["id"] for item in self.scene["cameras"]]
        for preset in presets:
            for camera in cameras:
                png_path = self.output_a / "presets" / preset / f"{camera}.png"
                metadata_path = self.output_a / "presets" / preset / f"{camera}.metadata.json"
                self.assertTrue(png_path.is_file())
                self.assertTrue(metadata_path.is_file())
                with Image.open(png_path) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (160, 200))
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["pixel_order"], "row-major")
                self.assertEqual(metadata["image_size"], [160, 200])
                self.assertEqual(len(metadata["pixels"]["uv_u"]), 160 * 200)
                self.assertEqual(len(metadata["pixels"]["linear_irradiance_r"]), 160 * 200)
                self.assertEqual(len(metadata["pixels"]["tone_mapped_linear_r"]), 160 * 200)
                self.assertTrue(metadata["key_pixels"])

    def test_png_bytes_match_pixel_metadata(self) -> None:
        metadata_path = self.output_a / "presets" / "studio_aces" / "front.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with Image.open(self.output_a / metadata["image"]) as image:
            pixels = np.asarray(image, dtype=np.uint8)
        coverage = np.asarray(metadata["pixels"]["coverage"], dtype=bool)
        for channel, name in enumerate(("srgb8_r", "srgb8_g", "srgb8_b")):
            values = np.asarray(
                [pixels.reshape(-1, 3)[index, channel] if covered else 0 for index, covered in enumerate(coverage)],
                dtype=np.uint8,
            )
            metadata_values = np.asarray(
                [value if value is not None else 0 for value in metadata["pixels"][name]],
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(values, metadata_values)
        self.assertEqual(len(metadata["key_pixels"]), 4)

    def test_cross_view_and_material_consistency_checks_pass(self) -> None:
        for preset in (item["id"] for item in self.scene["lighting_presets"]):
            report = json.loads(
                (self.output_a / "presets" / preset / "consistency.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "pass", preset)
            checks = {check["id"]: check for check in report["checks"]}
            uv_check = checks["uv_cross_view_consistency"]
            self.assertEqual(uv_check["status"], "pass")
            self.assertLessEqual(uv_check["max_uv_error"], uv_check["thresholds"]["uv_l2"])
            self.assertLessEqual(uv_check["max_normal_error"], uv_check["thresholds"]["normal_l2"])
            boundary_check = checks["metallic_boundary_shading"]
            self.assertEqual(boundary_check["status"], "pass")
            self.assertLessEqual(
                boundary_check["max_linear_color_delta"],
                boundary_check["threshold"],
            )
            energy_check = checks["energy_conservation"]
            self.assertEqual(energy_check["status"], "pass")
            self.assertLessEqual(energy_check["max_reflectance"], 1.0)
            self.assertLessEqual(energy_check["max_lobe_weight_sum"], 1.0)

    def test_manifest_hashes_are_complete_and_valid(self) -> None:
        manifest = json.loads((self.output_a / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["input_sha256"], hashlib.sha256(self.scene_path.read_bytes()).hexdigest())
        listed: set[str] = set()
        for item in manifest["output_files"]:
            path = self.output_a / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            self.assertEqual(path.stat().st_size, item["bytes"])
            listed.add(item["path"])
        self.assertIn("cache/material_metallic.npy", listed)
        self.assertIn("presets/studio_aces/front.png", listed)
        manifest_hash = hashlib.sha256((self.output_a / "manifest.json").read_bytes()).hexdigest()
        self.assertEqual(
            (self.output_a / "manifest.sha256").read_text(encoding="ascii"),
            f"{manifest_hash}  manifest.json\n",
        )

    def test_tone_mapping_is_linear_then_encoded(self) -> None:
        values = np.asarray([[0.0, 0.18, 1.0], [2.0, 4.0, 8.0]], dtype=np.float64)
        reinhard = tonemap_linear(values, _light("reinhard"))
        expected_reinhard = values * (1.0 + values / 16.0) / (1.0 + values)
        np.testing.assert_allclose(reinhard, np.clip(expected_reinhard, 0.0, 1.0), rtol=0.0, atol=1e-14)
        aces = tonemap_linear(values, _light("aces"))
        expected_aces = values * (2.51 * values + 0.03) / (values * (2.43 * values + 0.59) + 0.14)
        np.testing.assert_allclose(aces, np.clip(expected_aces, 0.0, 1.0), rtol=0.0, atol=1e-14)
        encoded = linear_to_srgb8(np.asarray([0.0, 0.5, 1.0]))
        self.assertEqual(encoded.tolist(), [0, 188, 255])
        subtle = tonemap_linear(np.asarray([0.0001, 0.0002]), _light("aces"))
        self.assertGreater(subtle[1], subtle[0])

    def test_two_runs_are_byte_identical_including_cache(self) -> None:
        output_b = self.tmp / "output_b"
        result_b = render_batch(self.scene_path, output_b)
        self.assertEqual(result_b.exit_code, 0)
        self.assertEqual(_tree_hashes(self.output_a), _tree_hashes(output_b))
        manifest_a = json.loads((self.output_a / "manifest.json").read_text(encoding="utf-8"))
        manifest_b = json.loads((output_b / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_a["output_files"], manifest_b["output_files"])

    def test_existing_output_directory_is_replaced_whole(self) -> None:
        target = self.tmp / "replace_target"
        shutil.copytree(self.output_a, target)
        (target / "stale.txt").write_text("must disappear", encoding="utf-8")
        (target / "presets" / "studio_aces" / "front.png").write_bytes(b"corrupt")
        result = render_batch(self.scene_path, target)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse((target / "stale.txt").exists())
        self.assertEqual(_tree_hashes(self.output_a), _tree_hashes(target))

    def test_missing_light_field_isolated_to_one_preset(self) -> None:
        scene = json.loads(json.dumps(self.scene))
        del scene["lighting_presets"][1]["tone_mapping"]["exposure"]
        scene_path = _write_scene(self.tmp / "missing_light", scene)
        output = self.tmp / "missing_light_output"
        result = render_batch(scene_path, output)
        self.assertEqual(result.exit_code, 2)
        diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertIn("LIGHT_FIELD_MISSING", diagnostics["summary"]["codes"])
        results = {item["id"]: item for item in result.manifest["preset_results"]}
        self.assertEqual(results["studio_aces"]["status"], "success")
        self.assertEqual(results["dusk_reinhard"]["status"], "error")
        self.assertEqual(results["cool_rim_aces"]["status"], "success")
        self.assertTrue((output / "presets" / "studio_aces" / "front.png").is_file())
        self.assertTrue((output / "presets" / "cool_rim_aces" / "side.png").is_file())
        self.assertFalse((output / "presets" / "dusk_reinhard").exists())

    def test_mask_dimension_mismatch_reports_stable_code(self) -> None:
        scene = json.loads(json.dumps(self.scene))
        scene["material"]["layers"][4]["mask_size"] = [511, 512]
        scene_path = _write_scene(self.tmp / "bad_mask", scene)
        output = self.tmp / "bad_mask_output"
        result = render_batch(scene_path, output)
        self.assertEqual(result.exit_code, 2)
        diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertIn("MAT_MASK_SIZE_MISMATCH", diagnostics["summary"]["codes"])
        self.assertTrue(all(item["status"] == "error" for item in result.manifest["preset_results"]))
        self.assertFalse(any((output / "presets").glob("*/*.png")))

    def test_unknown_blend_reports_stable_code(self) -> None:
        scene = json.loads(json.dumps(self.scene))
        scene["material"]["layers"][0]["blend"] = "sparkle"
        scene_path = _write_scene(self.tmp / "bad_blend", scene)
        output = self.tmp / "bad_blend_output"
        result = render_batch(scene_path, output)
        self.assertEqual(result.exit_code, 2)
        diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertIn("MAT_BLEND_UNSUPPORTED", diagnostics["summary"]["codes"])


if __name__ == "__main__":
    unittest.main()
