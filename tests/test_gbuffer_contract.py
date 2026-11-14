"""Unit tests for G-Buffer attachment layout and format resolution."""

from __future__ import annotations

import unittest

from deferred_renderer.gbuffer import (
    ATTACHMENT_SPECS,
    INVALID_INTERNALFORMAT,
    SYNTHETIC_UNSUPPORTED,
    SUPPORTED_FORMAT_TOKENS,
    resolve_format_token,
)


class AttachmentContractTests(unittest.TestCase):
    def test_slots_are_contiguous_and_match_shader_order(self) -> None:
        # Slot order is the documented MRT contract consumed by both the
        # geometry fragment shader outputs and the lighting sampler bindings.
        names = [spec["name"] for spec in ATTACHMENT_SPECS]
        self.assertEqual(
            names, ["world_position", "world_normal", "albedo_material"]
        )
        self.assertEqual([spec["slot"] for spec in ATTACHMENT_SPECS], [0, 1, 2])

    def test_default_formats_match_specification(self) -> None:
        defaults = {spec["name"]: spec["default_format"] for spec in ATTACHMENT_SPECS}
        self.assertEqual(defaults["world_position"], "RGBA32F")
        self.assertEqual(defaults["world_normal"], "RGB16F")
        self.assertEqual(defaults["albedo_material"], "RGBA8")

    def test_force_format_resolution(self) -> None:
        self.assertEqual(
            resolve_format_token(SYNTHETIC_UNSUPPORTED), INVALID_INTERNALFORMAT
        )
        self.assertEqual(
            resolve_format_token("rgba32f"), SUPPORTED_FORMAT_TOKENS["RGBA32F"]
        )
        # Unknown tokens also map to the guaranteed-invalid sentinel rather
        # than silently succeeding.
        self.assertEqual(resolve_format_token("NOT_A_FORMAT"), INVALID_INTERNALFORMAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
