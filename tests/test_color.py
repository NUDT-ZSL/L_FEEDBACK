from __future__ import annotations

import math
import unittest

import numpy as np

from matrender.color import linear_to_srgb8, srgb_decode, srgb_encode


def _scalar_reference(value: float) -> int:
    x = max(float(value), 0.0)
    aces = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    aces = min(max(aces, 0.0), 1.0)
    if aces <= 0.0031308:
        encoded = 12.92 * aces
    else:
        encoded = 1.055 * math.pow(aces, 1.0 / 2.4) - 0.055
    return min(max(int(math.floor(encoded * 255.0 + 0.5)), 0), 255)


class ColorTests(unittest.TestCase):
    def test_aces_srgb_against_independent_scalar_reference(self) -> None:
        values = np.array(
            [
                0.0,
                1.0e-6,
                0.0002,
                0.003,
                0.01,
                0.05,
                0.18,
                0.5,
                1.0,
                2.0,
                8.0,
                100.0,
                10000.0,
            ],
            dtype=np.float64,
        )
        hdr = np.stack([values, np.sqrt(values), values[::-1]], axis=-1)
        actual = linear_to_srgb8(hdr)
        expected = np.empty(hdr.shape, dtype=np.uint8)
        for index in np.ndindex(hdr.shape[:-1]):
            for channel in range(3):
                expected[index + (channel,)] = _scalar_reference(
                    hdr[index + (channel,)]
                )
        maximum_error = int(
            np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))
        )
        self.assertLessEqual(maximum_error, 2)

    def test_srgb_transfer_roundtrip(self) -> None:
        encoded = np.linspace(0.0, 1.0, 10001)
        decoded = srgb_decode(encoded)
        roundtrip = srgb_encode(decoded)
        self.assertLess(float(np.max(np.abs(roundtrip - encoded))), 1.0e-12)

    def test_endpoints(self) -> None:
        actual = linear_to_srgb8(
            np.array([[[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]])
        )
        np.testing.assert_array_equal(
            actual, np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        )

