from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from material_tool import renderer as renderer_module
from material_tool.glsl import ShaderSources, generate_shaders
from material_tool.graph import CHANNELS, parse_graph_document
from material_tool.pipeline import RunSettings, run
from material_tool.renderer import Renderer, compute_png_stats
from material_tool.samples import valid_demo_document
from tests.conftest import GL_AVAILABLE


pytestmark = [
    pytest.mark.gl,
    pytest.mark.skipif(
        not GL_AVAILABLE, reason="OpenGL 3.3 context is unavailable"
    ),
]


def test_builtin_pipeline_outputs_images_stats_and_consistency(tmp_path: Path, capsys) -> None:
    summary, exit_code = run(
        RunSettings(
            output_dir=tmp_path,
            channels=list(CHANNELS),
            size=64,
            tolerance=1e-4,
            reload_check=True,
        )
    )
    stderr = capsys.readouterr().err
    assert exit_code == 0
    assert "GRAPH_CYCLE" in stderr
    assert "EDGE_TYPE_MISMATCH" in stderr
    assert (tmp_path / "summary.json").exists()

    graph_entries = {entry["id"]: entry for entry in summary["graphs"]}
    demo = graph_entries["demo_material"]
    assert demo["status"] == "rendered"
    assert demo["consistency"]["conclusion"] == "pass"
    assert demo["hot_reload"]["equal"] is True
    assert demo["hot_reload"]["resource_counts_equal"] is True
    assert demo["hot_reload"]["image_hashes_equal"] is True
    assert set(demo["channels"]) == set(CHANNELS)
    image_hashes = set()
    for channel in CHANNELS:
        channel_entry = demo["channels"][channel]
        image_path = tmp_path / channel_entry["image"]
        assert image_path.exists()
        recomputed = compute_png_stats(image_path)
        assert channel_entry["width"] == 64
        assert channel_entry["height"] == 64
        assert channel_entry["stats"] == recomputed
        image_hashes.add(channel_entry["stats"]["sha256"])
    assert len(image_hashes) == len(CHANNELS)

    inconsistent = graph_entries["inconsistent_material"]
    assert inconsistent["status"] == "rendered"
    assert inconsistent["consistency"]["conclusion"] == "fail"
    rough_differences = [
        item
        for item in inconsistent["consistency"]["differences"]
        if item["parameter"] == "roughness"
    ]
    assert rough_differences
    assert "rough_by_pass" in rough_differences[0]["node_path"]
    normal_differences = [
        item
        for item in inconsistent["consistency"]["differences"]
        if item["parameter"] == "normal"
    ]
    assert normal_differences
    assert "normal_by_pass" in normal_differences[0]["node_path"]
    assert inconsistent["consistency"]["tolerance"] == 1e-4

    for graph_id in ("invalid_cycle", "invalid_type", "invalid_missing_reference"):
        assert graph_entries[graph_id]["status"] == "invalid"
        assert graph_entries[graph_id]["errors"]
    assert summary["summary"]["rendered"] == 2
    assert summary["summary"]["invalid"] == 3
    assert summary["summary"]["consistency_failed"] == 1
    assert summary["resource_counts_after_close"] == {}
    assert summary["leak_free_after_close"] is True
    assert all(
        entry["resources"]["session_leak_free"]
        for entry in summary["graphs"]
        if entry["status"] != "invalid"
    )


def test_hot_reload_reuses_framebuffer_textures_and_buffers(
    renderer: Renderer, tmp_path: Path
) -> None:
    graph = parse_graph_document(valid_demo_document())
    baseline = renderer.resource_counts()
    session = renderer.create_session(graph, output_dir=tmp_path)
    try:
        before = session.render_channel("roughness")
        before_resources = renderer.resource_counts()
        persistent_before = {
            (entry["kind"], entry["id"])
            for entry in renderer.resource_entries()
            if entry["kind"] in {"framebuffer", "texture", "buffer"}
        }

        document = copy.deepcopy(graph.document)
        document["parameters"]["rough_base"]["value"] = 0.78
        document["metadata"]["change"] = "roughness"
        reloaded = parse_graph_document(document, source_name="reload.json")
        report = session.reload(reloaded)
        after_resources = renderer.resource_counts()
        persistent_after = {
            (entry["kind"], entry["id"])
            for entry in renderer.resource_entries()
            if entry["kind"] in {"framebuffer", "texture", "buffer"}
        }
        after = session.render_channel("roughness")

        assert report["equal"] is True
        assert before_resources == after_resources
        assert persistent_before == persistent_after
        assert before.stats["sha256"] != after.stats["sha256"]
        assert after.stats["mean"] > before.stats["mean"]
        assert renderer.resource_counts() == before_resources
    finally:
        session.release()
    assert renderer.resource_counts() == baseline


def test_compile_failure_rolls_back_and_batch_continues(
    tmp_path: Path, monkeypatch
) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    valid = valid_demo_document()
    broken = copy.deepcopy(valid)
    broken["id"] = "broken_material"
    (materials / "a_broken.json").write_text(json.dumps(broken), encoding="utf-8")
    (materials / "b_valid.json").write_text(json.dumps(valid), encoding="utf-8")

    original = generate_shaders

    def inject_error(graph):
        sources = original(graph)
        if graph.graph_id == "broken_material":
            fragment = sources.fragment.replace("void main() {", "void main( {", 1)
            return ShaderSources(sources.vertex, fragment)
        return sources

    monkeypatch.setattr(renderer_module, "generate_shaders", inject_error)
    summary, exit_code = run(
        RunSettings(
            output_dir=tmp_path / "out",
            channels=["base_color"],
            size=64,
            reload_check=False,
            input_dir=materials,
        )
    )
    assert exit_code == 0
    entries = {entry["id"]: entry for entry in summary["graphs"]}
    assert entries["broken_material"]["status"] == "failed"
    assert entries["demo_material"]["status"] == "rendered"
    broken_errors = entries["broken_material"]["errors"]
    compile_errors = [
        error for error in broken_errors if error["error_code"] == "GL_SHADER_COMPILE_ERROR"
    ]
    assert compile_errors
    diagnostics = compile_errors[0]["details"]["diagnostics"]
    assert diagnostics
    assert any(item["line"] is not None for item in diagnostics)
    assert compile_errors[0]["details"]["stage"] == "fragment"
    assert entries["broken_material"]["resources"]["session_leak_free"] is True
    assert summary["resource_counts_after_close"] == {}
    assert (tmp_path / "out" / "demo_material" / "base_color.png").exists()


def test_link_failure_rolls_back_and_batch_continues(
    tmp_path: Path, monkeypatch
) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    valid = valid_demo_document()
    broken = copy.deepcopy(valid)
    broken["id"] = "link_broken_material"
    (materials / "a_link_broken.json").write_text(json.dumps(broken), encoding="utf-8")
    (materials / "b_valid.json").write_text(json.dumps(valid), encoding="utf-8")

    original_link = renderer_module.glLinkProgram
    calls = {"count": 0}

    def skip_first_link(program):
        calls["count"] += 1
        if calls["count"] == 1:
            return
        original_link(program)

    monkeypatch.setattr(renderer_module, "glLinkProgram", skip_first_link)
    summary, exit_code = run(
        RunSettings(
            output_dir=tmp_path / "out",
            channels=["base_color"],
            size=64,
            reload_check=False,
            input_dir=materials,
        )
    )
    assert exit_code == 0
    entries = {entry["id"]: entry for entry in summary["graphs"]}
    assert entries["link_broken_material"]["status"] == "failed"
    assert entries["demo_material"]["status"] == "rendered"
    codes = {
        error["error_code"]
        for error in entries["link_broken_material"]["errors"]
    }
    assert "GL_SHADER_LINK_ERROR" in codes
    assert entries["link_broken_material"]["resources"]["session_leak_free"] is True
    assert summary["resource_counts_after_close"] == {}


def test_channel_subset_and_tolerance_are_recorded(tmp_path: Path) -> None:
    summary, exit_code = run(
        RunSettings(
            output_dir=tmp_path,
            channels=["base_color", "roughness"],
            size=32,
            tolerance=0.002,
            reload_check=False,
        )
    )
    assert exit_code == 0
    assert summary["settings"]["channels"] == ["base_color", "roughness"]
    assert summary["settings"]["consistency_tolerance"] == 0.002
    demo = next(entry for entry in summary["graphs"] if entry["id"] == "demo_material")
    assert set(demo["channels"]) == {"base_color", "roughness"}
    assert demo["consistency"]["tolerance"] == 0.002
    inconsistent = next(
        entry for entry in summary["graphs"] if entry["id"] == "inconsistent_material"
    )
    assert inconsistent["consistency"]["conclusion"] == "fail"

