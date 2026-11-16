from __future__ import annotations

import json
from pathlib import Path

from material_validator.manifest import parse_manifest


def _write_shaders(directory: Path) -> None:
    shaders = directory / "shaders"
    shaders.mkdir()
    (shaders / "test.vert").write_text("#version 330 core\nvoid main() {}\n", encoding="utf-8")
    (shaders / "test.frag").write_text("#version 330 core\nvoid main() {}\n", encoding="utf-8")


def _base_material() -> dict:
    return {
        "name": "test",
        "vertex_shader": "shaders/test.vert",
        "fragment_shader": "shaders/test.frag",
        "passes": ["depth_prepass", "normal", "deferred"],
        "uniforms": {},
    }


def test_version_defaults_unknown_fields_and_derived_channels(tmp_path: Path) -> None:
    _write_shaders(tmp_path)
    material = _base_material()
    material["vendor_notes"] = {"author": "artist"}
    manifest = {
        "render": {"width": 64},
        "materials": [material],
        "vendor_root": {"kept": True},
        "scenes": [{"name": "s", "objects": []}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    parsed = parse_manifest(path)

    assert parsed.schema_version == "1.0"
    assert parsed.render.width == 64
    assert parsed.render.height == 96
    assert parsed.comparison.absolute == 5.0e-4
    assert [channel.kind for channel in parsed.ordered_channels()] == [
        "depth_prepass",
        "normal",
        "deferred",
    ]
    unknown_paths = {item["path"] for item in parsed.unknown_fields}
    assert "vendor_root" in unknown_paths
    assert "materials[0].vendor_notes" in unknown_paths
    assert any(item.code == "schema_version_missing" for item in parsed.diagnostics)
    assert sum(item.code == "unknown_field" for item in parsed.diagnostics) >= 2


def test_blend_depth_state_and_layout_errors(tmp_path: Path) -> None:
    _write_shaders(tmp_path)
    material = _base_material()
    material["blend"] = {
        "enabled": True,
        "src_factor": "src_alpha",
        "dst_factor": "one_minus_src_alpha",
    }
    material["depth"] = {"test": True, "write": True, "func": "lequal"}
    manifest = {
        "schema_version": "1.1",
        "materials": [material],
        "channels": [
            {"name": "depth", "kind": "depth_prepass", "materials": ["test"]},
            {
                "name": "normal",
                "kind": "normal",
                "materials": ["test"],
                "layout": ["albedo", "world_position", "world_normal", "metallic_roughness"],
            },
            {"name": "deferred", "kind": "deferred", "materials": ["test"]},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    parsed = parse_manifest(path)
    codes = {item.code for item in parsed.diagnostics}

    assert "blend_with_depth_write" in codes
    assert "gbuffer_layout_mismatch" in codes
    blend_error = next(item for item in parsed.diagnostics if item.code == "blend_with_depth_write")
    assert blend_error.material == "test"
    assert blend_error.path == "materials[0].depth.write"


def test_channel_uniform_conflict_reports_all_sources(tmp_path: Path) -> None:
    _write_shaders(tmp_path)
    material = _base_material()
    material["uniforms"] = {
        "u_color": {"type": "vec3", "value": [1.0, 1.0, 1.0]},
    }
    manifest = {
        "schema_version": "1.1",
        "materials": [material],
        "channels": [
            {
                "name": "depth",
                "kind": "depth_prepass",
                "materials": [{"name": "test", "uniforms": {"u_color": {"type": "vec3", "value": [1.0, 0.0, 0.0]}}}],
            },
            {
                "name": "normal",
                "kind": "normal",
                "materials": [{"name": "test", "uniforms": {"u_color": {"type": "vec3", "value": [0.0, 1.0, 0.0]}}}],
            },
            {"name": "deferred", "kind": "deferred", "materials": ["test"]},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    parsed = parse_manifest(path)
    conflict = next(item for item in parsed.diagnostics if item.code == "channel_uniform_conflict")

    assert conflict.material == "test"
    assert conflict.details["sources"] == [
        "materials[0].uniforms.u_color",
        "channels[0].materials[0].uniforms.u_color",
        "channels[1].materials[0].uniforms.u_color",
    ]
    assert parsed.materials["test"].config_valid is False
