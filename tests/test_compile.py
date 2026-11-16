from __future__ import annotations

from pathlib import Path

from material_validator.manifest import parse_manifest
from material_validator.samples import generate_sample
from material_validator.shader_compile import compile_materials


def test_compile_aggregates_stages_without_stopping(tmp_path: Path) -> None:
    manifest_path = generate_sample(tmp_path / "sample", "invalid")
    parsed = parse_manifest(manifest_path)

    results = compile_materials(parsed.materials.values(), workers=3)
    by_name = {item.material: item for item in results}

    assert [item.material for item in results] == list(parsed.materials)
    assert by_name["good"].status == "ok"
    assert by_name["broken"].status == "error"
    stages = {item.stage for item in by_name["broken"].diagnostics}
    assert {"vertex", "fragment"}.issubset(stages)
    assert all(item.line is not None for item in by_name["broken"].diagnostics)
    assert any(item.code == "active_uniform_not_declared" for item in by_name["missing_uniform"].diagnostics)
    assert by_name["invalid_transparent"].status == "ok"


def test_unused_manifest_uniform_policy(tmp_path: Path) -> None:
    source = tmp_path / "sample"
    manifest_path = generate_sample(source, "valid")
    import json

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["materials"][0]["uniforms"]["u_unused"] = {"type": "float", "value": 1.0}
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    parsed = parse_manifest(manifest_path)

    warning_results = compile_materials([parsed.materials["stone"]], unused_uniform_policy="warn")
    error_results = compile_materials([parsed.materials["stone"]], unused_uniform_policy="error")

    assert any(item.code == "declared_uniform_unused" for item in warning_results[0].diagnostics)
    assert warning_results[0].status == "ok"
    assert error_results[0].status == "error"
