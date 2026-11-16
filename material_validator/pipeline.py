from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .diagnostics import Diagnostic, count_by_severity, stable_diagnostics
from .manifest import ParsedManifest, parse_manifest
from .render import RenderSceneResult, render_scenes
from .shader_compile import MaterialCompileResult, compile_materials


NONBLOCKING_MATERIAL_CODES = {
    "metallic_out_of_range",
    "roughness_out_of_range",
}


@dataclass
class ValidationRun:
    report: dict[str, Any]
    status: str
    report_path: Path
    summary_path: Path
    exit_code: int


def _safe_scene_name(name: str, index: int) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not normalized:
        normalized = "scene"
    return f"{index:02d}_{normalized[:48]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_array(
    output_dir: Path,
    relative_base: str,
    array: np.ndarray,
    write_raw: bool,
) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    npy_path = output_dir / f"{relative_base}.npy"
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, array, allow_pickle=False)
    entry: dict[str, Any] = {
        "path": npy_path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(npy_path),
        "shape": list(array.shape),
        "dtype": "float32",
        "bytes": npy_path.stat().st_size,
    }
    if write_raw:
        raw_path = output_dir / f"{relative_base}.f32"
        raw_path.write_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes(order="C"))
        entry["raw_path"] = raw_path.relative_to(output_dir).as_posix()
        entry["raw_sha256"] = _sha256(raw_path)
        entry["raw_bytes"] = raw_path.stat().st_size
    return entry


def _material_parser_errors(manifest: ParsedManifest, material_name: str) -> list[Diagnostic]:
    return [
        item
        for item in manifest.diagnostics
        if item.material == material_name and item.severity == "error"
    ]


def _renderable_materials(
    manifest: ParsedManifest,
    compile_results: dict[str, MaterialCompileResult],
) -> dict[str, Any]:
    renderable: dict[str, Any] = {}
    for name, material in manifest.materials.items():
        if not material.config_valid:
            material.renderable = False
            continue
        parser_errors = _material_parser_errors(manifest, name)
        if any(item.code not in NONBLOCKING_MATERIAL_CODES for item in parser_errors):
            material.renderable = False
            continue
        result = compile_results.get(name)
        if result is None or result.status != "ok":
            material.renderable = False
            continue
        renderable[name] = material
    channel_names = {
        ref.name for channel in manifest.channels for ref in channel.materials
    }
    return {name: material for name, material in renderable.items() if name in channel_names}


def _fill_channel_context(manifest: ParsedManifest) -> None:
    for item in manifest.diagnostics:
        if item.material and not item.channels:
            material = manifest.materials.get(item.material)
            if material is not None:
                item.channels = tuple(material.channel_names)


def _comparison_totals(scene_results: list[RenderSceneResult]) -> dict[str, Any]:
    totals = {
        "pixels_total": 0,
        "pixels_over_threshold": 0,
        "max_abs_error": 0.0,
        "max_channel": None,
        "max_location": None,
        "non_finite_pixels": 0,
        "pass": True,
        "categories": {
            "normal_encoding_precision": {"pixels": 0, "max_abs_error": 0.0, "channel": None, "location": None},
            "floating_blend_order": {"pixels": 0, "max_abs_error": 0.0, "channel": None, "location": None},
            "material_parameter_range": {"pixels": 0, "max_abs_error": 0.0, "channel": None, "location": None},
            "unclassified": {"pixels": 0, "max_abs_error": 0.0, "channel": None, "location": None},
        },
    }
    for scene in scene_results:
        comparison = scene.comparison
        if comparison is None:
            continue
        totals["pixels_total"] += comparison["pixels_total"]
        totals["pixels_over_threshold"] += comparison["pixels_over_threshold"]
        totals["non_finite_pixels"] += comparison["non_finite_pixels"]
        totals["pass"] = totals["pass"] and comparison["pass"]
        if comparison["max_abs_error"] > totals["max_abs_error"]:
            totals["max_abs_error"] = comparison["max_abs_error"]
            totals["max_channel"] = comparison["max_channel"]
            totals["max_location"] = dict(comparison["max_location"]) if comparison["max_location"] else None
            totals["max_scene"] = scene.scene
        for category, stats in comparison["categories"].items():
            aggregate = totals["categories"][category]
            aggregate["pixels"] += stats["pixels"]
            if stats["max_abs_error"] > aggregate["max_abs_error"]:
                aggregate["max_abs_error"] = stats["max_abs_error"]
                aggregate["channel"] = stats["channel"]
                aggregate["location"] = (
                    dict(stats["location"]) if stats["location"] else None
                )
                aggregate["scene"] = scene.scene
    return totals


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Material Validator {report['tool_version']}",
        f"Status: {report['status']}",
        f"Schema: {report.get('schema_version')}",
        f"Manifest SHA256: {report.get('manifest', {}).get('sha256')}",
        (
            f"Materials: {summary['materials_total']} total, "
            f"{summary['materials_compiled']} compiled, "
            f"{summary['materials_renderable']} renderable"
        ),
        f"Scenes: {summary['scenes_total']} total, {summary['scenes_passed']} comparison-passed",
        (
            f"Pixels: {summary['comparison']['pixels_total']} compared, "
            f"{summary['comparison']['pixels_over_threshold']} over threshold"
        ),
        (
            f"Max deviation: {summary['comparison']['max_abs_error']} "
            f"channel={summary['comparison']['max_channel']} "
            f"location={summary['comparison']['max_location']}"
        ),
        f"Normal fallbacks: {summary['normal_fallback_pixels']} pixels",
        f"Errors: {summary['errors']}",
        f"Warnings: {summary['warnings']}",
    ]
    return "\n".join(lines) + "\n"


def run_validation(
    manifest_path: str | Path,
    output_dir: str | Path,
    parallel_compile: int = 1,
    parallel_render: int = 1,
    absolute_threshold: float | None = None,
    relative_threshold: float | None = None,
    unused_uniform_policy: str | None = None,
    render: bool = True,
    write_raw: bool = False,
    fail_on_warning: bool = False,
) -> ValidationRun:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = parse_manifest(manifest_path)
    comparison = manifest.comparison
    if absolute_threshold is not None:
        comparison = replace(comparison, absolute=absolute_threshold)
    if relative_threshold is not None:
        comparison = replace(comparison, relative=relative_threshold)
    policy = unused_uniform_policy or manifest.uniform_policy["unused_declared"]
    compile_results_list = compile_materials(
        manifest.materials.values(),
        unused_uniform_policy=policy,
        workers=max(1, parallel_compile),
    )
    compile_results = {item.material: item for item in compile_results_list}
    for result in compile_results_list:
        manifest.diagnostics.extend(result.diagnostics)
    _fill_channel_context(manifest)
    renderable = _renderable_materials(manifest, compile_results)
    scene_results: list[RenderSceneResult] = []
    if render:
        if renderable:
            scene_results = render_scenes(
                manifest.scenes,
                renderable,
                manifest.render.width,
                manifest.render.height,
                manifest.render.normal_fallback,
                manifest.render.normal_tolerance,
                manifest.render.background,
                comparison,
                workers=max(1, parallel_render),
            )
        else:
            manifest.diagnostics.append(
                Diagnostic(
                    "error",
                    "no_renderable_materials",
                    "No material passed validation, so scenes were not rendered",
                )
            )
    for scene in scene_results:
        manifest.diagnostics.extend(scene.diagnostics)
        if scene.comparison is not None and not scene.comparison["pass"]:
            manifest.diagnostics.append(
                Diagnostic(
                    "error",
                    "comparison_threshold_exceeded",
                    (
                        f"Scene {scene.scene} has {scene.comparison['pixels_over_threshold']} pixels "
                        "over the deferred/forward threshold"
                    ),
                    details={
                        "scene": scene.scene,
                        "pixels_over_threshold": scene.comparison["pixels_over_threshold"],
                        "max_abs_error": scene.comparison["max_abs_error"],
                        "max_channel": scene.comparison["max_channel"],
                        "max_location": scene.comparison["max_location"],
                        "categories": scene.comparison["categories"],
                    },
                )
            )
    scene_rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scene_results):
        base = _safe_scene_name(scene.scene, index)
        outputs: dict[str, Any] = {}
        for channel_name in (
            "world_position",
            "world_normal",
            "albedo",
            "metallic_roughness",
            "depth_prepass",
            "depth_gbuffer",
            "deferred",
            "forward",
            "blend_coverage",
        ):
            if channel_name in scene.buffers:
                outputs[channel_name] = _save_array(
                    output,
                    f"buffers/{base}.{channel_name}",
                    scene.buffers[channel_name],
                    write_raw,
                )
        if "deferred" in scene.buffers and "forward" in scene.buffers:
            difference = scene.buffers["deferred"] - scene.buffers["forward"]
            outputs["difference"] = _save_array(
                output,
                f"buffers/{base}.difference",
                difference,
                write_raw,
            )
        row = scene.to_dict()
        if scene.comparison is not None and not scene.comparison["pass"]:
            row["status"] = "error"
        row["outputs"] = outputs
        scene_rows.append(row)
    diagnostics = stable_diagnostics(manifest.diagnostics)
    errors, warnings = count_by_severity(diagnostics)
    comparison_totals = _comparison_totals(scene_results)
    scenes_passed = sum(
        1 for scene in scene_results if scene.comparison is not None and scene.comparison["pass"]
    )
    materials_compiled = sum(1 for item in compile_results_list if item.status == "ok")
    normal_fallback_pixels = sum(
        int(scene.gbuffer["normal_fallback_pixels"])
        for scene in scene_results
        if scene.gbuffer is not None
    )
    metallic_invalid_pixels = sum(
        int(scene.gbuffer["metallic_out_of_range_pixels"])
        for scene in scene_results
        if scene.gbuffer is not None
    )
    roughness_invalid_pixels = sum(
        int(scene.gbuffer["roughness_out_of_range_pixels"])
        for scene in scene_results
        if scene.gbuffer is not None
    )
    status = "failed" if errors > 0 or (fail_on_warning and warnings > 0) else "passed"
    material_rows: list[dict[str, Any]] = []
    for name, material in manifest.materials.items():
        compile_result = compile_results[name]
        row = compile_result.to_dict()
        row["renderable"] = bool(material.renderable)
        row["state"] = {"blend": material.blend.to_dict(), "depth": material.depth.to_dict()}
        row["parser_diagnostics"] = [
            item.to_dict()
            for item in stable_diagnostics(_material_parser_errors(manifest, name))
        ]
        row["uniform_declarations"] = {
            uniform_name: spec.to_dict() for uniform_name, spec in material.uniforms.items()
        }
        material_rows.append(row)
    report: dict[str, Any] = {
        "tool": "material-validator",
        "tool_version": __version__,
        "status": status,
        "schema_version": manifest.schema_version,
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.manifest_sha256,
        },
        "settings": {
            "render": manifest.render.to_dict(),
            "comparison": comparison.to_dict(),
            "uniform_policy": {"unused_declared": policy},
            "parallel_compile": max(1, parallel_compile),
            "parallel_render": max(1, parallel_render),
            "render_enabled": render,
        },
        "summary": {
            "status": status,
            "materials_total": len(manifest.materials),
            "materials_compiled": materials_compiled,
            "materials_renderable": len(renderable),
            "channels_total": len(manifest.channels),
            "scenes_total": len(scene_results),
            "scenes_passed": scenes_passed,
            "errors": errors,
            "warnings": warnings,
            "normal_fallback_pixels": normal_fallback_pixels,
            "metallic_out_of_range_pixels": metallic_invalid_pixels,
            "roughness_out_of_range_pixels": roughness_invalid_pixels,
            "comparison": comparison_totals,
        },
        "channels": [item.to_dict() for item in manifest.channels],
        "materials": material_rows,
        "scenes": scene_rows,
        "unknown_fields": manifest.unknown_fields,
        "diagnostics": [item.to_dict() for item in diagnostics],
    }
    report_path = output / "report.json"
    summary_path = output / "summary.txt"
    _write_text(report_path, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _write_text(summary_path, _summary_text(report))
    return ValidationRun(
        report=report,
        status=status,
        report_path=report_path,
        summary_path=summary_path,
        exit_code=0 if status == "passed" else 1,
    )
