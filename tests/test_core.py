"""Fast core tests; the end-to-end matrix is covered by material-baker selftest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from material_baker.config import load_config
from material_baker.errors import ToolError
from material_baker.lighting import evaluate_material
from material_baker.pngio import read_png_rgb, write_rgb_png
from material_baker.resample import decode_normal_map, normalize_vectors


class PNGTests(unittest.TestCase):
    def test_8bit_gamma_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "constant.png"
            write_rgb_png(path, np.full((4, 4, 3), 0.25), 8, 2.2)
            image = read_png_rgb(path)
            expected = 0.25 ** (1.0 / 2.2)
            self.assertEqual((image.width, image.height, image.bit_depth), (4, 4, 8))
            self.assertLessEqual(float(np.max(np.abs(image.rgb - expected))), 1.0 / 255.0)

    def test_16bit_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "constant16.png"
            write_rgb_png(path, np.full((3, 5, 3), 0.5), 16, 1.0)
            image = read_png_rgb(path)
            self.assertEqual((image.width, image.height, image.bit_depth), (5, 3, 16))
            self.assertLessEqual(
                float(np.max(np.abs(image.rgb - 0.5))), 1.0 / 65535.0
            )


class NormalTests(unittest.TestCase):
    def test_decode_flip_and_zero_fallback(self) -> None:
        encoded = np.asarray(
            [[[0.5, 0.5, 1.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]]],
            dtype=np.float64,
        )
        decoded, zero_count = decode_normal_map(encoded, flip_green=False)
        self.assertAlmostEqual(float(decoded[0, 0, 2]), 1.0)
        self.assertAlmostEqual(float(decoded[0, 1, 1]), -1.0)
        self.assertEqual(zero_count, 1)
        self.assertTrue(np.array_equal(decoded[0, 2], np.asarray([0.0, 0.0, 1.0])))

        flipped, _ = decode_normal_map(encoded, flip_green=True)
        self.assertAlmostEqual(float(flipped[0, 1, 1]), 1.0)

    def test_gaussian_resampled_normals_are_unit_length(self) -> None:
        rng = np.random.default_rng(7)
        vectors, _ = normalize_vectors(rng.normal(size=(9, 9, 3)))
        from material_baker.resample import resize_image

        reduced, _ = normalize_vectors(resize_image(vectors, 5, 5))
        lengths = np.linalg.norm(reduced, axis=2)
        self.assertLess(float(np.max(np.abs(lengths - 1.0))), 1.0e-10)


class LightingTests(unittest.TestCase):
    def _config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "resolution": 256,
                        "output_bits": 8,
                        "output_gamma": 2.2,
                        "directional_light": {
                            "direction": [0.4, 0.5, 0.8],
                            "color": [1.0, 0.9, 0.8],
                            "intensity": 2.0,
                        },
                        "hemisphere_light": {
                            "sky_color": [0.3, 0.55, 0.9],
                            "ground_color": [0.12, 0.08, 0.06],
                            "intensity": 0.7,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return load_config(path)

    def test_direct_energy_bound_and_metal_diffuse(self) -> None:
        config = self._config()
        albedo = np.asarray([[[0.8, 0.5, 0.2]]], dtype=np.float64)
        normal = np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float64)
        roughness = np.asarray([[0.0]], dtype=np.float64)
        metallic = np.asarray([[1.0]], dtype=np.float64)
        result = evaluate_material(
            albedo,
            roughness,
            metallic,
            normal,
            config.view_direction,
            config.directional,
            config.hemisphere,
            alpha_epsilon=config.alpha_epsilon,
        )
        self.assertLessEqual(float(np.max(np.abs(result.direct_diffuse))), 1.0e-12)
        self.assertLessEqual(float(np.max(np.abs(result.environment_diffuse))), 1.0e-12)
        self.assertTrue(np.all(np.isfinite(result.final_color)))
        incident = np.asarray(config.directional.color) * config.directional.intensity
        direct_total = result.direct_diffuse + result.direct_specular
        self.assertTrue(np.all(direct_total <= incident + 1.0e-10))


class ConfigTests(unittest.TestCase):
    def test_invalid_gamma_is_business_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "output_gamma": 1.8,
                        "directional_light": {"direction": [0.0, 0.0, 1.0]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ToolError) as context:
                load_config(path)
            self.assertEqual(context.exception.code, "E_INVALID_CONFIG")


if __name__ == "__main__":
    unittest.main()

