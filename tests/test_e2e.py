from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from material_validator.pipeline import run_validation
from material_validator.samples import generate_sample


def _output_hashes(run) -> dict[str, str]:
    scene = run.report["scenes"][0]
    return {name: entry["sha256"] for name, entry in scene["outputs"].items()}


def _report_hash(run) -> str:
    return hashlib.sha256(run.report_path.read_bytes()).hexdigest()


def test_valid_e2e_parallel_and_repeat_determinism(tmp_path: Path) -> None:
    manifest = generate_sample(tmp_path / "sample", "valid")

    first = run_validation(manifest, tmp_path / "out1", parallel_compile=1, parallel_render=1)
    second = run_validation(manifest, tmp_path / "out2", parallel_compile=4, parallel_render=4)
    repeat = run_validation(manifest, tmp_path / "out1", parallel_compile=1, parallel_render=1)

    assert first.status == "passed"
    assert second.status == "passed"
    assert first.report["summary"]["normal_fallback_pixels"] > 0
    assert first.report["scenes"][0]["gbuffer"]["normal_non_unit_pixels"] == 0
    assert first.report["scenes"][0]["depth_prepass_mismatch_pixels"] == 0
    assert set(
        (
            "world_position",
            "world_normal",
            "albedo",
            "metallic_roughness",
            "deferred",
            "forward",
        )
    ).issubset(first.report["scenes"][0]["outputs"])
    assert _output_hashes(first) == _output_hashes(second)
    assert first.report["scenes"][0]["comparison"] == second.report["scenes"][0]["comparison"]
    assert _report_hash(first) == _report_hash(repeat)


def test_channel_order_does_not_change_buffers(tmp_path: Path) -> None:
    manifest = generate_sample(tmp_path / "sample", "valid")
    baseline = run_validation(manifest, tmp_path / "baseline", parallel_compile=2, parallel_render=2)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["channels"] = list(reversed(data["channels"]))
    reordered = manifest.with_name("reordered_manifest.json")
    reordered.write_text(json.dumps(data), encoding="utf-8")

    changed = run_validation(reordered, tmp_path / "reordered", parallel_compile=2, parallel_render=2)

    assert baseline.status == "passed"
    assert changed.status == "passed"
    assert _output_hashes(baseline) == _output_hashes(changed)


def test_invalid_sample_reports_all_material_failures(tmp_path: Path) -> None:
    manifest = generate_sample(tmp_path / "sample", "invalid")

    run = run_validation(manifest, tmp_path / "out", parallel_compile=3, parallel_render=2)
    codes = {item["code"] for item in run.report["diagnostics"]}

    assert run.exit_code == 1
    assert run.report["summary"]["materials_total"] == 6
    assert run.report["summary"]["materials_compiled"] == 4
    assert {
        "glsl_compile_error",
        "active_uniform_not_declared",
        "blend_with_depth_write",
        "channel_uniform_conflict",
        "metallic_out_of_range",
        "roughness_out_of_range",
    }.issubset(codes)
    assert run.report["unknown_fields"][0]["path"] == "studio_note"


def test_blend_sample_classifies_pixel_difference(tmp_path: Path) -> None:
    manifest = generate_sample(tmp_path / "sample", "blend")

    run = run_validation(manifest, tmp_path / "out", parallel_compile=2, parallel_render=2)
    comparison = run.report["scenes"][0]["comparison"]

    assert run.exit_code == 1
    assert comparison["pixels_over_threshold"] > 0
    assert comparison["categories"]["floating_blend_order"]["pixels"] > 0
    assert comparison["max_channel"] in {"R", "G", "B"}


def test_vector_matrix_scalar_and_bool_bindings_reach_shader(tmp_path: Path) -> None:
    manifest = generate_sample(tmp_path / "sample", "valid")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["materials"] = [data["materials"][0]]
    data["materials"][0]["name"] = "binding_probe"
    data["materials"][0]["uniforms"]["u_base_color"]["value"] = [0.2, 0.6, 0.4]
    data["materials"][0]["uniforms"]["u_tint_matrix"]["value"] = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    data["materials"][0]["uniforms"]["u_enable_tint"]["value"] = True
    data["materials"][0]["uniforms"]["u_metallic"]["value"] = 0.3
    data["materials"][0]["uniforms"]["u_roughness"]["value"] = 0.7
    data["channels"] = [
        {"name": "depth", "kind": "depth_prepass", "materials": ["binding_probe"]},
        {"name": "normal", "kind": "normal", "materials": ["binding_probe"]},
        {"name": "deferred", "kind": "deferred", "materials": ["binding_probe"]},
    ]
    data["scenes"][0]["camera"] = {
        "position": [0.0, 0.0, 3.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y_deg": 45.0,
        "near": 0.1,
        "far": 20.0,
    }
    data["scenes"][0]["objects"] = [
        {
            "name": "probe",
            "mesh": "cube",
            "material": "binding_probe",
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.3, 1.3, 1.3],
        }
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    run = run_validation(manifest, tmp_path / "out")
    outputs = run.report["scenes"][0]["outputs"]
    albedo = np.load(tmp_path / "out" / outputs["albedo"]["path"], allow_pickle=False)
    params = np.load(
        tmp_path / "out" / outputs["metallic_roughness"]["path"],
        allow_pickle=False,
    )
    center_y = albedo.shape[0] // 2
    center_x = albedo.shape[1] // 2

    assert run.status == "passed"
    np.testing.assert_allclose(albedo[center_y, center_x, :3], [0.6, 0.2, 0.4], atol=1.0e-6)
    assert albedo[center_y, center_x, 3] == 1.0
    np.testing.assert_allclose(params[center_y, center_x, :2], [0.3, 0.7], atol=1.0e-6)
