from __future__ import annotations

import json
from pathlib import Path

from material_tool.glsl import generate_shaders, material_uniform_layout
from material_tool.graph import CHANNELS, load_graph_file, parse_graph_document
from material_tool.samples import (
    builtin_sample_documents,
    cycle_demo_document,
    inconsistent_demo_document,
    missing_reference_demo_document,
    type_mismatch_demo_document,
    valid_demo_document,
)


def test_valid_graph_parses_and_generates_all_channels() -> None:
    graph = parse_graph_document(valid_demo_document(), source_name="valid.json")
    assert graph.valid, [error.to_dict() for error in graph.errors]
    assert set(graph.outputs) == set(CHANNELS)
    assert all(node_type in graph.node_info for node_type in graph.nodes)
    node_types = {node.node_type for node in graph.nodes.values()}
    assert {
        "constant_color",
        "texture",
        "math",
        "normal_map",
        "swizzle",
    } <= node_types
    sources = generate_shaders(graph)
    assert "#version 330 core" in sources.vertex
    assert "layout(std140) uniform MaterialParameters" in sources.fragment
    assert "layout(location = 4)" not in sources.fragment
    assert "out_roughness" in sources.fragment
    assert "out_metallic" in sources.fragment
    assert "out_normal" in sources.fragment
    assert "**" not in sources.fragment
    for node_id in graph.nodes:
        assert f"node[{node_id}]" in sources.fragment


def test_cycle_is_reported_with_path() -> None:
    graph = parse_graph_document(cycle_demo_document())
    assert not graph.valid
    cycle_errors = [error for error in graph.errors if error.code == "GRAPH_CYCLE"]
    assert cycle_errors
    assert cycle_errors[0].path[:2] == ["a", "b"]
    assert cycle_errors[0].details["cycle"][-1] == cycle_errors[0].details["cycle"][0]


def test_type_mismatch_is_reported_per_edge() -> None:
    graph = parse_graph_document(type_mismatch_demo_document())
    mismatches = [
        error for error in graph.errors if error.code == "EDGE_TYPE_MISMATCH"
    ]
    assert len(mismatches) == 1
    assert mismatches[0].details["source_type"] == "vec3"
    assert mismatches[0].details["expected"] == "float"
    assert mismatches[0].node_id == "bad_quantize"


def test_missing_reference_is_structured() -> None:
    graph = parse_graph_document(missing_reference_demo_document())
    errors = [error for error in graph.errors if error.code == "EDGE_NODE_MISSING"]
    assert len(errors) == 1
    assert errors[0].details["missing_node"] == "does_not_exist"


def test_inconsistent_graph_is_structurally_valid() -> None:
    graph = parse_graph_document(inconsistent_demo_document())
    assert graph.valid
    assert graph.outputs["roughness"] == ("rough_by_pass", "out")
    assert graph.outputs["normal"] == ("normal_by_pass", "out")


def test_invalid_json_file_becomes_structured_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    graph = load_graph_file(path)
    assert not graph.valid
    assert graph.errors[0].code == "GRAPH_JSON_INVALID"


def test_multiple_independent_errors_are_collected() -> None:
    document = valid_demo_document()
    document["edges"].append({"from": "missing.out", "to": "base_color.a"})
    document["outputs"]["metallic"] = "missing_metal.out"
    graph = parse_graph_document(document)
    codes = {error.code for error in graph.errors}
    assert "EDGE_NODE_MISSING" in codes
    assert "OUTPUT_NODE_MISSING" in codes


def test_unknown_node_type_is_rejected() -> None:
    document = valid_demo_document()
    document["nodes"].append({"id": "mystery", "type": "not_a_node", "params": {}})
    graph = parse_graph_document(document)
    assert not graph.valid
    assert any(error.code == "NODE_TYPE_UNKNOWN" for error in graph.errors)


def test_std140_offsets_follow_glsl_alignment() -> None:
    graph = parse_graph_document(valid_demo_document())
    layout = material_uniform_layout(graph)
    assert layout.offsets["base_tint"] == 0
    assert layout.offsets["emissive_color"] == 16
    assert layout.offsets["metal_value"] == 28
    assert layout.offsets["normal_strength"] == 32
    assert layout.offsets["rough_base"] == 36
    assert layout.offsets["tex_scale"] == 40
    assert layout.size == 48


def test_all_builtin_samples_have_stable_ids() -> None:
    documents = builtin_sample_documents()
    assert "demo_material" in documents
    assert "inconsistent_material" in documents
    assert json.dumps(documents["demo_material"], sort_keys=True) == json.dumps(
        valid_demo_document(), sort_keys=True
    )

