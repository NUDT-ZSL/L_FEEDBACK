import os
import tempfile
import unittest

import numpy as np

from terrain_baker.errors import MemoryBudgetExceeded, ParameterError
from terrain_baker.heightfield import (
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    HeightfieldSpec,
    build_heightfield,
    height_values_for_rows,
    open_heightfield,
    validate_resolution,
)
from terrain_baker.memory import MemoryBudget
from terrain_baker.noise import fbm_2d, value_noise_2d


class NoiseTests(unittest.TestCase):
    def test_value_noise_bounds_and_determinism(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 8, 2000)
        y = rng.uniform(0, 8, 2000)
        a = value_noise_2d(x, y, seed=123)
        b = value_noise_2d(x, y, seed=123)
        np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(a.min(), -1.0)
        self.assertLessEqual(a.max(), 1.0)

    def test_fbm_deterministic_and_seed_sensitive(self):
        x = np.linspace(0, 3, 50)
        xx, yy = np.meshgrid(x, x)
        a = fbm_2d(xx, yy, seed=1)
        b = fbm_2d(xx, yy, seed=1)
        c = fbm_2d(xx, yy, seed=2)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


class ResolutionValidationTests(unittest.TestCase):
    def test_accepts_supported_odd_range(self):
        validate_resolution(MIN_RESOLUTION)
        validate_resolution(MAX_RESOLUTION)
        validate_resolution(1025)

    def test_rejects_even_or_out_of_range(self):
        for bad in (256, 4096, 255, 4099):
            with self.assertRaises(ParameterError):
                validate_resolution(bad)


class HeightfieldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.budget = MemoryBudget(64 * 1024 * 1024)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bit_identical_across_runs_and_block_sizes(self):
        spec = HeightfieldSpec(seed=99, resolution=257, height_scale=50.0)
        path_a = os.path.join(self.tmp.name, "a.dat")
        path_b = os.path.join(self.tmp.name, "b.dat")
        grid_a = build_heightfield(path_a, spec, self.budget, block_rows=16)
        grid_b = build_heightfield(path_b, spec, self.budget, block_rows=64)
        np.testing.assert_array_equal(np.asarray(grid_a), np.asarray(grid_b))
        raw_a = open(path_a, "rb").read()
        raw_b = open(path_b, "rb").read()
        self.assertEqual(raw_a, raw_b)

    def test_rows_match_full_formula(self):
        spec = HeightfieldSpec(seed=5, resolution=257, height_scale=12.0)
        path = os.path.join(self.tmp.name, "h.dat")
        grid = build_heightfield(path, spec, self.budget)
        rows = np.arange(0, 257, dtype=np.int64)
        expected = height_values_for_rows(rows, spec)
        np.testing.assert_array_equal(np.asarray(grid), expected)

    def test_memmap_roundtrip(self):
        spec = HeightfieldSpec(seed=3, resolution=257, height_scale=5.0)
        path = os.path.join(self.tmp.name, "m.dat")
        build_heightfield(path, spec, self.budget)
        reopened = open_heightfield(path, 257)
        self.assertEqual(reopened.shape, (257, 257))
        self.assertEqual(reopened.dtype, np.dtype("<f4"))

    def test_tiny_budget_rejected(self):
        spec = HeightfieldSpec(seed=3, resolution=257, height_scale=5.0)
        path = os.path.join(self.tmp.name, "x.dat")
        with self.assertRaises(MemoryBudgetExceeded):
            build_heightfield(path, spec, MemoryBudget(1024))


if __name__ == "__main__":
    unittest.main()
