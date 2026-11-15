from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from .compiler import CompiledMaterial
from .evaluate import MaterialBuffers, evaluate_material
from .lighting import LightingResult, shade_scene
from .mathutil import linear_to_srgb
from .outputs import publish_directory
from .pngio import encode_gray16, encode_rgb16, encode_rgb8
from .raster import GBuffer, render_gbuffer
from .scene import Scene

CHANNEL_FILES = {
    "base_color": "base_color.png",
    "normal": "normal.png",
    "roughness": "roughness.png",
    "metallic": "metallic.png",
    "emissive": "emissive.png",
    "final": "final.png",
    "report": "report.json",
    "ir": "material_ir.json",
}


@dataclass(frozen=True)
class RenderResult:
    gbuffer: GBuffer
    material: MaterialBuffers
    lighting: LightingResult
    report: Mapping[str, Any]
    files: Mapping[str, bytes]


def _number(value: Any) -> float:
    rounded = round(float(value), 10)
    return 0.0 if rounded == 0.0 else rounded


def _stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.ndim == 1:
        selected = selected[:, None]
    if selected.shape[0] == 0:
        dimensions = selected.shape[1] if selected.ndim == 2 else 1
        zeros = [0.0] * dimensions
        return {"mean": zeros, "min": zeros, "max": zeros, "count": 0}
    return {
        "mean": [_number(value) for value in np.mean(selected, axis=0)],
        "min": [_number(value) for value in np.min(selected, axis=0)],
        "max": [_number(value) for value in np.max(selected, axis=0)],
        "count": int(selected.shape[0]),
    }


def _material_images(material: MaterialBuffers) -> Dict[str, bytes]:
    base_srgb = linear_to_srgb(np.clip(material.base_color, 0.0, 1.0))
    normal_encoded = np.clip(material.normal * 0.5 + 0.5, 0.0, 1.0)
    emissive_linear = np.clip(material.emissive, 0.0, 1.0)
    return {
        CHANNEL_FILES["base_color"]: encode_rgb8(base_srgb),
        CHANNEL_FILES["normal"]: encode_rgb16(normal_encoded),
        CHANNEL_FILES["roughness"]: encode_gray16(material.roughness),
        CHANNEL_FILES["metallic"]: encode_gray16(material.metallic),
        CHANNEL_FILES["emissive"]: encode_rgb16(emissive_linear),
    }


def _build_report(
    compiled: CompiledMaterial,
    scene: Scene,
    gbuffer: GBuffer,
    material: MaterialBuffers,
    lighting: LightingResult,
    *,
    graph_sha256: str,
    scene_sha256: str,
) -> Dict[str, Any]:
    valid = gbuffer.valid
    valid_count = int(np.count_nonzero(valid))
    total = int(gbuffer.width * gbuffer.height)
    degenerate = int(np.count_nonzero(material.degenerate_tangent & valid))
    fallback = int(np.count_nonzero(material.normal_fallback & valid))
    edge = int(np.count_nonzero(gbuffer.edge))
    final_values = lighting.ldr_srgb
    final_mask = np.ones(valid.shape, dtype=bool)
    report: Dict[str, Any] = {
        "schema": "render-report/v1",
        "ok": True,
        "graph": compiled.name,
        "scene": scene.name,
        "resolution": {"width": gbuffer.width, "height": gbuffer.height},
        "inputs": {
            "graph_sha256": graph_sha256,
            "scene_sha256": scene_sha256,
        },
        "counts": {
            "total_pixels": total,
            "valid_pixels": valid_count,
            "shaded_pixels": valid_count,
            "degenerate_tangent_pixels": degenerate,
            "normal_fallback_pixels": fallback,
            "edge_pixels": edge,
        },
        "channels": {
            "base_color": _stats(material.base_color, valid),
            "normal": _stats(material.normal, valid),
            "roughness": _stats(material.roughness, valid),
            "metallic": _stats(material.metallic, valid),
            "emissive": _stats(material.emissive, valid),
            "final_ldr": _stats(final_values, final_mask),
        },
        "g_buffer": {
            "depth_convention": "positive_ray_distance",
            "edge_depth_method": "analytic_sphere_plane_intersection",
            "world_position": _stats(gbuffer.world, valid),
            "geometric_normal": _stats(gbuffer.geometric_normal, valid),
            "depth": _stats(gbuffer.depth, valid),
        },
        "lighting": {
            "direct_lights": len(scene.lights),
            "tone_mapping": "aces_narkowicz_then_srgb",
            "radiance": _stats(lighting.radiance, valid),
        },
        "compiler": {
            "ir_sha256": compiled.sha256(),
            "instruction_count": len(compiled.instructions),
            "folded_node_count": len(compiled.folded_nodes),
            "instruction_order": compiled.instruction_order(),
            "folded_nodes": [folded.to_dict() for folded in compiled.folded_nodes],
        },
        "outputs": {
            "base_color": {
                "file": CHANNEL_FILES["base_color"],
                "bit_depth": 8,
                "encoding": "srgb_rgb",
            },
            "normal": {
                "file": CHANNEL_FILES["normal"],
                "bit_depth": 16,
                "encoding": "world_space_linear_(n*0.5+0.5)",
            },
            "roughness": {
                "file": CHANNEL_FILES["roughness"],
                "bit_depth": 16,
                "encoding": "linear_gray",
            },
            "metallic": {
                "file": CHANNEL_FILES["metallic"],
                "bit_depth": 16,
                "encoding": "linear_gray",
            },
            "emissive": {
                "file": CHANNEL_FILES["emissive"],
                "bit_depth": 16,
                "encoding": "linear_rgb_clamped_0_1",
            },
            "final": {
                "file": CHANNEL_FILES["final"],
                "bit_depth": 8,
                "encoding": "srgb_rgb",
            },
            "report": {"file": CHANNEL_FILES["report"], "encoding": "json"},
            "ir": {"file": CHANNEL_FILES["ir"], "encoding": "json"},
        },
        "limits": compiled.limits.as_dict(),
    }
    return report


def render_scene(
    compiled: CompiledMaterial,
    scene: Scene,
    *,
    graph_sha256: str = "",
    scene_sha256: str = "",
) -> RenderResult:
    gbuffer = render_gbuffer(scene)
    material = evaluate_material(compiled, gbuffer, scene.camera.position)
    lighting = shade_scene(scene, gbuffer, material)
    report = _build_report(
        compiled,
        scene,
        gbuffer,
        material,
        lighting,
        graph_sha256=graph_sha256,
        scene_sha256=scene_sha256,
    )

    files: Dict[str, bytes] = {}
    files.update(_material_images(material))
    files[CHANNEL_FILES["final"]] = encode_rgb8(lighting.ldr_srgb)

    ir_payload = compiled.to_dict()
    ir_payload["ir_sha256"] = compiled.sha256()
    ir_bytes = (
        json.dumps(ir_payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    files[CHANNEL_FILES["ir"]] = ir_bytes
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    files[CHANNEL_FILES["report"]] = report_bytes
    return RenderResult(
        gbuffer=gbuffer,
        material=material,
        lighting=lighting,
        report=report,
        files=files,
    )


def write_render(result: RenderResult, output_dir) -> None:
    publish_directory(output_dir, result.files, replace=True)
