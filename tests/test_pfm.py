from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from soft_renderer.imageio import (
    aces_tone_map,
    linear_to_srgb_preview,
    read_pfm,
    write_pfm,
)


class PfmTests(unittest.TestCase):
    def test_color_pfm_roundtrip_and_little_endian_marker(self) -> None:
        height, width = 5, 7
        image = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
        image = (image / 17.0) - 2.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pfm"
            write_pfm(str(path), image)
            loaded = read_pfm(str(path))
            header = path.read_bytes()[:40]
        self.assertTrue(header.startswith(b"PF\n7 5\n-1.0\n"))
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(loaded, image)

    def test_grayscale_depth_pfm_roundtrip(self) -> None:
        depth = np.array([[0.0, 1.5], [np.inf, 3.25]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.pfm"
            write_pfm(str(path), depth)
            loaded = read_pfm(str(path))
        np.testing.assert_array_equal(loaded, depth)

    def test_aces_gamma_preview_matches_png_pixels(self) -> None:
        linear = np.array(
            [
                [[0.0, 0.25, 1.0], [4.0, 16.0, 64.0]],
                [[0.01, 0.1, 0.5], [0.75, 2.0, 8.0]],
            ],
            dtype=np.float32,
        )
        expected = linear_to_srgb_preview(linear, 0.8)
        mapped = aces_tone_map(linear, 0.8)
        self.assertTrue(np.all(mapped >= 0.0) and np.all(mapped <= 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.png"
            Image.fromarray(expected, mode="RGB").save(path, format="PNG")
            loaded = np.asarray(Image.open(path).convert("RGB"))
        np.testing.assert_array_equal(loaded, expected)


if __name__ == "__main__":
    unittest.main()
