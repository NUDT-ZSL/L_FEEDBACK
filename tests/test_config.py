from __future__ import annotations

import copy
import unittest

from proc_gi.config import DiagnosticLog, SceneConfig, load_scene_config
from proc_gi.scene_generator import generate_config_dict


class ConfigTests(unittest.TestCase):
    def test_unknown_material_falls_back_with_business_code(self) -> None:
        payload = copy.deepcopy(generate_config_dict())
        payload["materials"][0]["type"] = "SubsurfaceMagic"
        diagnostics = DiagnosticLog()
        config = SceneConfig.from_dict(payload, diagnostics)
        material = config.material_by_id(payload["materials"][0]["id"])
        self.assertEqual(material.type, "Lambertian")
        self.assertEqual(material.shader_type, 0)
        self.assertEqual(material.original_type, "SubsurfaceMagic")
        records = [record for record in diagnostics.records if record["code"] == "MAT_UNKNOWN_TYPE"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["context"]["fallback_type"], "Lambertian")

    def test_generated_scene_json_round_trip(self) -> None:
        config = load_scene_config("scene.json")
        self.assertEqual(config.seed, 196)
        self.assertGreaterEqual(len(config.materials), 3)
        self.assertGreaterEqual(len(config.lights), 1)
        self.assertEqual(
            {material.type for material in config.materials},
            {"Lambertian", "OrenNayar", "BlinnPhong"},
        )

    def test_light_limit_is_enforced_with_warning(self) -> None:
        payload = copy.deepcopy(generate_config_dict())
        template = payload["lights"][0]
        payload["lights"] = []
        for index in range(260):
            light = copy.deepcopy(template)
            light["id"] = f"bulk-{index:03d}"
            light["intensity"] = float(index + 1)
            payload["lights"].append(light)
        diagnostics = DiagnosticLog()
        config = SceneConfig.from_dict(payload, diagnostics)
        self.assertEqual(len(config.lights), 256)
        self.assertEqual(config.lights[0].id, "bulk-259")
        records = [record for record in diagnostics.records if record["code"] == "LIGHT_LIMIT_TRUNCATED"]
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["context"]["truncated_light_ids"]), 4)


if __name__ == "__main__":
    unittest.main()

