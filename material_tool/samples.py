"""Deterministic built-in material graphs used when no input directory is given."""

from __future__ import annotations

import copy
from typing import Any


def _base_document(graph_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": graph_id,
        "metadata": {"builtin": True, "deterministic": True},
        "camera": {
            "position": [0.0, 0.15, 3.15],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_degrees": 42.0,
        },
        "light": {
            "direction": [-0.45, -0.65, -0.60],
            "color": [1.0, 0.96, 0.90],
            "intensity": 1.0,
        },
        "parameters": {},
        "nodes": [],
        "edges": [],
        "outputs": {},
    }


def valid_demo_document() -> dict[str, Any]:
    document = _base_document("demo_material")
    document["parameters"] = {
        "base_tint": {"type": "vec3", "value": [0.82, 0.28, 0.13]},
        "rough_base": {"type": "float", "value": 0.38},
        "metal_value": {"type": "float", "value": 0.18},
        "normal_strength": {"type": "float", "value": 0.82},
        "emissive_color": {"type": "vec3", "value": [0.035, 0.12, 0.28]},
        "tex_scale": {"type": "vec2", "value": [2.0, 2.0]},
    }
    document["nodes"] = [
        {"id": "uv", "type": "uv", "params": {}},
        {"id": "tex_scale", "type": "parameter", "params": {"name": "tex_scale"}},
        {"id": "scaled_uv", "type": "math", "params": {"op": "mul"}},
        {"id": "detail_tex", "type": "texture", "params": {"texture": "checker"}},
        {"id": "base_tint", "type": "parameter", "params": {"name": "base_tint"}},
        {"id": "base_color", "type": "math", "params": {"op": "mul"}},
        {"id": "rough_base", "type": "parameter", "params": {"name": "rough_base"}},
        {"id": "rough_delta", "type": "constant_float", "params": {"value": 0.08}},
        {"id": "roughness", "type": "math", "params": {"op": "add"}},
        {"id": "metal_value", "type": "parameter", "params": {"name": "metal_value"}},
        {"id": "normal_tex", "type": "texture", "params": {"texture": "normal_tile"}},
        {"id": "normal_rgb", "type": "swizzle", "params": {"components": "xyz"}},
        {"id": "two", "type": "constant_float", "params": {"value": 2.0}},
        {"id": "normal_x2", "type": "math", "params": {"op": "mul"}},
        {"id": "one", "type": "constant_float", "params": {"value": 1.0}},
        {"id": "normal_decoded", "type": "math", "params": {"op": "sub"}},
        {"id": "normal_strength", "type": "parameter", "params": {"name": "normal_strength"}},
        {"id": "normal_mapped", "type": "normal_map", "params": {}},
        {"id": "emissive_color", "type": "parameter", "params": {"name": "emissive_color"}},
        {"id": "emissive_gain", "type": "constant_float", "params": {"value": 0.35}},
        {"id": "emissive", "type": "math", "params": {"op": "mul"}},
        {"id": "unit_tint", "type": "constant_color", "params": {"value": [1.0, 1.0, 1.0]}},
        {"id": "emissive_final", "type": "math", "params": {"op": "mul"}},
    ]
    document["edges"] = [
        {"from": "uv.out", "to": "scaled_uv.a"},
        {"from": "tex_scale.out", "to": "scaled_uv.b"},
        {"from": "scaled_uv.out", "to": "detail_tex.uv"},
        {"from": "scaled_uv.out", "to": "normal_tex.uv"},
        {"from": "detail_tex.color", "to": "base_color.a"},
        {"from": "base_tint.out", "to": "base_color.b"},
        {"from": "rough_base.out", "to": "roughness.a"},
        {"from": "rough_delta.out", "to": "roughness.b"},
        {"from": "normal_tex.color", "to": "normal_rgb.in"},
        {"from": "normal_rgb.out", "to": "normal_x2.a"},
        {"from": "two.out", "to": "normal_x2.b"},
        {"from": "normal_x2.out", "to": "normal_decoded.a"},
        {"from": "one.out", "to": "normal_decoded.b"},
        {"from": "normal_decoded.out", "to": "normal_mapped.normal"},
        {"from": "normal_strength.out", "to": "normal_mapped.strength"},
        {"from": "emissive_color.out", "to": "emissive.a"},
        {"from": "emissive_gain.out", "to": "emissive.b"},
        {"from": "emissive.out", "to": "emissive_final.a"},
        {"from": "unit_tint.out", "to": "emissive_final.b"},
    ]
    document["outputs"] = {
        "base_color": "base_color.out",
        "roughness": "roughness.out",
        "metallic": "metal_value.out",
        "normal": "normal_mapped.out",
        "emissive": "emissive_final.out",
    }
    return document


def inconsistent_demo_document() -> dict[str, Any]:
    document = _base_document("inconsistent_material")
    document["parameters"] = {
        "albedo": {"type": "vec3", "value": [0.18, 0.62, 0.55]},
        "emission": {"type": "vec3", "value": [0.04, 0.08, 0.16]},
    }
    document["nodes"] = [
        {"id": "albedo", "type": "parameter", "params": {"name": "albedo"}},
        {
            "id": "rough_by_pass",
            "type": "channel_switch",
            "params": {
                "type": "float",
                "default": 0.27,
                "values": {
                    "base_color": 0.27,
                    "roughness": 0.31,
                    "metallic": 0.27,
                    "normal": 0.27,
                    "emissive": 0.27,
                },
            },
        },
        {"id": "metal", "type": "constant_float", "params": {"value": 0.68}},
        {
            "id": "normal_by_pass",
            "type": "channel_switch",
            "params": {
                "type": "vec3",
                "default": [0.0, 0.0, 1.0],
                "values": {
                    "base_color": [0.0, 0.0, 1.0],
                    "roughness": [0.0, 0.0, 1.0],
                    "metallic": [0.0, 0.0, 1.0],
                    "normal": [0.35, 0.0, 0.9367497],
                    "emissive": [0.0, 0.0, 1.0],
                },
            },
        },
        {"id": "emission", "type": "parameter", "params": {"name": "emission"}},
    ]
    document["outputs"] = {
        "base_color": "albedo.out",
        "roughness": "rough_by_pass.out",
        "metallic": "metal.out",
        "normal": "normal_by_pass.out",
        "emissive": "emission.out",
    }
    return document


def cycle_demo_document() -> dict[str, Any]:
    document = _base_document("invalid_cycle")
    document["nodes"] = [
        {"id": "a", "type": "math", "params": {"op": "add"}},
        {"id": "b", "type": "math", "params": {"op": "add"}},
        {"id": "base", "type": "constant_color", "params": {"value": [0.2, 0.4, 0.6]}},
        {"id": "metal", "type": "constant_float", "params": {"value": 0.5}},
        {"id": "normal", "type": "constant_vec3", "params": {"value": [0.0, 0.0, 1.0]}},
        {"id": "emission", "type": "constant_color", "params": {"value": [0.1, 0.0, 0.0]}},
    ]
    document["edges"] = [
        {"from": "a.out", "to": "b.a"},
        {"from": "b.out", "to": "a.a"},
    ]
    document["outputs"] = {
        "base_color": "base.out",
        "roughness": "a.out",
        "metallic": "metal.out",
        "normal": "normal.out",
        "emissive": "emission.out",
    }
    return document


def type_mismatch_demo_document() -> dict[str, Any]:
    document = _base_document("invalid_type")
    document["nodes"] = [
        {"id": "world_normal", "type": "world_normal", "params": {}},
        {"id": "zero", "type": "constant_vec3", "params": {"value": [0.0, 0.0, 0.0]}},
        {"id": "vec_add", "type": "math", "params": {"op": "add"}},
        {"id": "bad_quantize", "type": "quantize", "params": {"steps": 255}},
        {"id": "base", "type": "constant_color", "params": {"value": [0.8, 0.2, 0.1]}},
        {"id": "metal", "type": "constant_float", "params": {"value": 0.0}},
        {"id": "emission", "type": "constant_color", "params": {"value": [0.0, 0.0, 0.0]}},
    ]
    document["edges"] = [
        {"from": "world_normal.out", "to": "vec_add.a"},
        {"from": "zero.out", "to": "vec_add.b"},
        {"from": "vec_add.out", "to": "bad_quantize.value"},
    ]
    document["outputs"] = {
        "base_color": "base.out",
        "roughness": "bad_quantize.out",
        "metallic": "metal.out",
        "normal": "world_normal.out",
        "emissive": "emission.out",
    }
    return document


def missing_reference_demo_document() -> dict[str, Any]:
    document = _base_document("invalid_missing_reference")
    document["nodes"] = [
        {"id": "base", "type": "constant_color", "params": {"value": [0.2, 0.7, 0.4]}},
        {"id": "rough", "type": "constant_float", "params": {"value": 0.4}},
        {"id": "metal", "type": "constant_float", "params": {"value": 0.1}},
        {"id": "normal", "type": "constant_vec3", "params": {"value": [0.0, 0.0, 1.0]}},
        {"id": "emission", "type": "constant_color", "params": {"value": [0.0, 0.0, 0.0]}},
    ]
    document["edges"] = [
        {"from": "does_not_exist.out", "to": "base.a"},
    ]
    document["outputs"] = {
        "base_color": "base.out",
        "roughness": "rough.out",
        "metallic": "metal.out",
        "normal": "normal.out",
        "emissive": "emission.out",
    }
    return document


def builtin_sample_documents() -> dict[str, dict[str, Any]]:
    return {
        "demo_material": valid_demo_document(),
        "inconsistent_material": inconsistent_demo_document(),
        "invalid_cycle": cycle_demo_document(),
        "invalid_type": type_mismatch_demo_document(),
        "invalid_missing_reference": missing_reference_demo_document(),
    }


def builtin_sample_copy(name: str) -> dict[str, Any]:
    documents = builtin_sample_documents()
    if name not in documents:
        raise KeyError(name)
    return copy.deepcopy(documents[name])

