from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .consistency import UV_LANDMARKS, run_consistency
from .errors import ErrorCode, PipelineError, diagnostic
from .geometry import Mesh
from .material import MaterialMaps, compile_material
from .pngio import write_rgb8_png
from .renderer import build_cached_mesh, build_gbuffers, metadata_document, render_views
from .scene import SceneSpec, inspect_lighting_preset, load_scene_bytes
from .util import sha256_file, write_json


PIPELINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class BatchResult:
    output_directory: Path
    exit_code: int
    manifest: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_cache(
    root: Path,
    scene: SceneSpec,
    mesh: Mesh,
    material: MaterialMaps,
) -> list[dict[str, Any]]:
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mesh_positions.npy": mesh.positions,
        "mesh_normals.npy": mesh.normals,
        "mesh_uvs.npy": mesh.uvs,
        "mesh_triangles.npy": mesh.triangles,
        "material_albedo_linear.npy": material.albedo_linear,
        "material_roughness.npy": material.roughness,
        "material_metallic.npy": material.metallic,
    }
    entries: list[dict[str, Any]] = []
    for name, array in arrays.items():
        path = cache_dir / name
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        entries.append(_artifact_entry(root, path))
    cache_manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "scene_id": scene.scene_id,
        "input_sha256": scene.input_sha256,
        "canonical_input_sha256": scene.canonical_sha256,
        "uv_layout": [scene.uv_layout.width, scene.uv_layout.height],
        "arrays": entries,
    }
    manifest_path = cache_dir / "cache_manifest.json"
    write_json(manifest_path, cache_manifest)
    return entries + [_artifact_entry(root, manifest_path)]


def _diagnostic_with_preset(error: dict[str, Any], preset_id: str) -> dict[str, Any]:
    result = {
        "code": error["code"],
        "severity": error.get("severity", "error"),
        "scope": error.get("scope", "material"),
        "path": error.get("path", ""),
        "message": error.get("message", ""),
        "context": dict(error.get("context", {})),
    }
    result["context"]["preset_id"] = preset_id
    return result


def _preset_label(raw: Any, index: int) -> str:
    if isinstance(raw, dict) and isinstance(raw.get("id"), str) and raw["id"]:
        return raw["id"]
    return f"preset_{index}"


def _sort_diagnostics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            str(item.get("context", {}).get("preset_id", "")),
            item.get("code", ""),
            item.get("path", ""),
            item.get("message", ""),
        ),
    )


def _write_diagnostics(root: Path, scene: SceneSpec | None, input_sha256: str, items: list[dict[str, Any]], fatal: bool) -> Path:
    path = root / "diagnostics.json"
    document = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "scene_id": scene.scene_id if scene is not None else None,
        "input_sha256": input_sha256,
        "fatal": fatal,
        "summary": {
            "diagnostic_count": len(items),
            "codes": sorted({item["code"] for item in items}),
        },
        "diagnostics": items,
    }
    write_json(path, document)
    return path


def _write_manifest(
    root: Path,
    *,
    scene: SceneSpec | None,
    input_sha256: str,
    preset_results: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    fatal: bool,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"manifest.json", "manifest.sha256"}:
            continue
        artifacts.append(_artifact_entry(root, path))
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "scene_id": scene.scene_id if scene is not None else None,
        "input_sha256": input_sha256,
        "canonical_input_sha256": scene.canonical_sha256 if scene is not None else None,
        "fatal": fatal,
        "status": "success" if not diagnostics and not fatal else ("fatal" if fatal else "partial"),
        "preset_results": preset_results,
        "diagnostics": {
            "path": "diagnostics.json",
            "count": len(diagnostics),
            "codes": sorted({item["code"] for item in diagnostics}),
        },
        "output_files": artifacts,
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    (root / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.json\n",
        encoding="ascii",
        newline="\n",
    )
    return manifest


def _publish_directory(staging: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.parent / f".{target.name}.backup-{os.getpid()}"
        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            os.replace(backup, target)
            raise
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    else:
        os.replace(staging, target)


def _render_valid_scene(
    root: Path,
    scene: SceneSpec,
    diagnostic_items: list[dict[str, Any]],
    preset_results: list[dict[str, Any]],
) -> None:
    material_result = compile_material(scene.uv_layout, scene.material)
    if material_result.maps is None:
        for index, raw_preset in enumerate(scene.lighting_presets):
            preset_label = _preset_label(raw_preset, index)
            material_diagnostics = [_diagnostic_with_preset(error, preset_label) for error in material_result.errors]
            diagnostic_items.extend(material_diagnostics)
            preset_results.append(
                {
                    "id": preset_label,
                    "status": "error",
                    "diagnostic_codes": sorted({error["code"] for error in material_diagnostics}),
                    "outputs": [],
                }
            )
        return

    material = material_result.maps
    mesh = build_cached_mesh(scene)
    _write_cache(root, scene, mesh, material)
    gbuffers = build_gbuffers(scene, mesh)

    for index, raw_preset in enumerate(scene.lighting_presets):
        preset, light_errors = inspect_lighting_preset(raw_preset, index)
        preset_id = _preset_label(raw_preset, index)
        if preset is None or light_errors:
            diagnostic_items.extend(light_errors)
            preset_results.append(
                {
                    "id": preset_id,
                    "status": "error",
                    "diagnostic_codes": sorted({error["code"] for error in light_errors}),
                    "outputs": [],
                }
            )
            continue

        views = render_views(scene, material, gbuffers, preset)
        consistency_report, consistency_errors = run_consistency(scene, material, preset, views)
        diagnostic_items.extend(consistency_errors)
        preset_dir = root / "presets" / preset.id
        preset_dir.mkdir(parents=True, exist_ok=True)
        consistency_path = preset_dir / "consistency.json"
        write_json(consistency_path, consistency_report)

        output_paths: list[Path] = [consistency_path]
        for camera in scene.cameras:
            view = views[camera.id]
            image_path = preset_dir / f"{camera.id}.png"
            write_rgb8_png(image_path, view.shading.srgb8)
            metadata_path = preset_dir / f"{camera.id}.metadata.json"
            metadata = metadata_document(
                scene=scene,
                preset=preset,
                view=view,
                materials=material,
                image_relative_path=_relative(image_path, root),
                image_path=image_path,
                consistency_relative_path=_relative(consistency_path, root),
                landmark_uvs=UV_LANDMARKS,
            )
            write_json(metadata_path, metadata)
            output_paths.extend((image_path, metadata_path))
        preset_results.append(
            {
                "id": preset.id,
                "status": "success" if consistency_report["status"] == "pass" else "partial",
                "consistency": _relative(consistency_path, root),
                "diagnostic_codes": sorted({error["code"] for error in consistency_errors}),
                "outputs": [_relative(path, root) for path in output_paths],
            }
        )


def render_batch(scene_path: Path, output_directory: Path) -> BatchResult:
    scene_path = scene_path.resolve()
    output_directory = output_directory.resolve()
    raw_input = scene_path.read_bytes()
    input_sha256 = hashlib.sha256(raw_input).hexdigest()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    diagnostics: list[dict[str, Any]] = []
    preset_results: list[dict[str, Any]] = []
    scene: SceneSpec | None = None
    fatal = False
    try:
        try:
            scene = load_scene_bytes(raw_input)
        except PipelineError as exc:
            diagnostics.append(exc.as_diagnostic())
            fatal = True
        if not fatal and scene is not None:
            _render_valid_scene(staging, scene, diagnostics, preset_results)
            diagnostics = _sort_diagnostics(diagnostics)
        _write_diagnostics(staging, scene, input_sha256, diagnostics, fatal)
        manifest = _write_manifest(
            staging,
            scene=scene,
            input_sha256=input_sha256,
            preset_results=preset_results,
            diagnostics=diagnostics,
            fatal=fatal,
        )
        _publish_directory(staging, output_directory)
        exit_code = 1 if fatal else (2 if diagnostics else 0)
        return BatchResult(output_directory, exit_code, manifest, tuple(diagnostics))
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def render_batch_with_error_output(scene_path: Path, output_directory: Path) -> BatchResult:
    """Run a batch and convert unexpected I/O failures into a stable diagnostic."""

    try:
        return render_batch(scene_path, output_directory)
    except Exception as exc:
        if isinstance(exc, PipelineError):
            raise
        output_directory = output_directory.resolve()
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_directory.name}.error-",
                dir=output_directory.parent,
            )
        )
        try:
            input_sha256 = sha256_file(scene_path)
            error_code = ErrorCode.IO_OUTPUT_FAILED if isinstance(exc, OSError) else ErrorCode.INTERNAL_ERROR
            item = diagnostic(
                error_code,
                str(exc),
                "output",
                str(output_directory),
            )
            _write_diagnostics(staging, None, input_sha256, [item], True)
            manifest = _write_manifest(
                staging,
                scene=None,
                input_sha256=input_sha256,
                preset_results=[],
                diagnostics=[item],
                fatal=True,
            )
            _publish_directory(staging, output_directory)
            return BatchResult(output_directory, 1, manifest, (item,))
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
