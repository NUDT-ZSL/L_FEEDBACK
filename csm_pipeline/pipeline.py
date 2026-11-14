"""End-to-end orchestration of the deterministic offline CSM pipeline."""

from __future__ import annotations

import json
import math
import os
import platform

import numpy as np

from .camera_path import interpolate_keyframes, load_path
from .csm import build_cascades, validate_cascades
from .depth_archive import write_depth_archive
from .errors import (
    ERR_CONFIG,
    ERR_INVALID_MATRIX,
    PipelineError,
)
from .matrices import is_finite_matrix, look_at, multiply, perspective
from .metrics import (
    array_checksum,
    max_absolute_error,
    normalized_cross_correlation,
    per_cascade_stats,
    psnr,
    shadow_region_mask,
    write_pgm,
)
from .rasterizer import rasterize, validate_depth_map
from .scene import generate_scene, scene_fingerprint
from .shadows import evaluate_shadows


REPORT_FILENAME = "report.json"


def validate_config(config: dict) -> dict:
    required = {
        "seed": int,
        "cascade_count": int,
        "shadow_resolution": int,
        "width": int,
        "height": int,
        "camera_near": (int, float),
        "camera_far": (int, float),
        "frame_time": (int, float),
        "pcf_kernel": int,
        "blend_fraction": (int, float),
        "normal_bias_scale": (int, float),
        "depth_bias": (int, float),
    }
    for key, expected_type in required.items():
        if key not in config:
            raise PipelineError(ERR_CONFIG, f"missing required config '{key}'")
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise PipelineError(
                ERR_CONFIG, f"config '{key}' has an invalid type: {value!r}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise PipelineError(
                ERR_CONFIG, f"config '{key}' must be finite, got {value!r}"
            )
    if config["cascade_count"] < 1 or config["cascade_count"] > 8:
        raise PipelineError(ERR_CONFIG, "cascade_count must be in [1, 8]")
    if config["shadow_resolution"] < 16 or config["shadow_resolution"] > 4096:
        raise PipelineError(
            ERR_CONFIG, "shadow_resolution must be in [16, 4096]"
        )
    if config["width"] < 16 or config["height"] < 16:
        raise PipelineError(ERR_CONFIG, "camera width/height must be >= 16")
    if not (0.0 < config["camera_near"] < config["camera_far"]):
        raise PipelineError(ERR_CONFIG, "require 0 < camera_near < camera_far")
    if config["pcf_kernel"] < 1 or config["pcf_kernel"] % 2 == 0:
        raise PipelineError(ERR_CONFIG, "pcf_kernel must be an odd positive integer")
    if not (0.0 <= config["blend_fraction"] <= 0.5):
        raise PipelineError(ERR_CONFIG, "blend_fraction must be in [0, 0.5]")
    if config["normal_bias_scale"] < 0.0 or config["depth_bias"] < 0.0:
        raise PipelineError(ERR_CONFIG, "bias values must be non-negative")
    return dict(config)


def _ensure_output_dir(output_dir: str) -> str:
    try:
        os.makedirs(output_dir, exist_ok=True)
        probe = os.path.join(output_dir, ".write_probe")
        with open(probe, "wb"):
            pass
        os.remove(probe)
    except OSError as exc:
        raise PipelineError(
            ERR_CONFIG, f"output directory is not writable: {exc}"
        ) from exc
    return os.path.abspath(output_dir)


def _render_shadow_stack(mesh, cascades: list, resolution: int, snapped: bool):
    stack = np.zeros(
        (len(cascades), resolution, resolution), dtype=np.float32
    )
    for cascade in cascades:
        matrix = (
            cascade.snapped_view_proj if snapped else cascade.view_proj
        )
        if not is_finite_matrix(matrix):
            raise PipelineError(
                ERR_INVALID_MATRIX,
                f"cascade {cascade.index} matrix is invalid",
            )
        rendered = rasterize(
            mesh, matrix, resolution, resolution, with_attributes=False
        )
        validate_depth_map(rendered["depth"], rendered["hit"])
        stack[cascade.index] = rendered["depth"]
    return stack


def run_pipeline(config: dict, output_dir: str) -> dict:
    """Execute every stage and write all artifacts plus the JSON report."""
    config = validate_config(config)
    output_dir = _ensure_output_dir(output_dir)

    scene = generate_scene(int(config["seed"]))
    fingerprint = scene_fingerprint(scene)
    frames = load_path(config["camera_path"])
    keyframe = interpolate_keyframes(frames, float(config["frame_time"]))

    aspect = config["width"] / config["height"]
    camera_view = look_at(
        keyframe.eye, keyframe.target, np.array((0.0, 1.0, 0.0))
    )
    camera_proj = perspective(
        math.radians(keyframe.fovy_deg),
        aspect,
        float(config["camera_near"]),
        float(config["camera_far"]),
    )
    camera_view_proj = multiply(camera_proj, camera_view)
    if not is_finite_matrix(camera_view_proj):
        raise PipelineError(ERR_INVALID_MATRIX, "camera view-projection is invalid")

    mesh = scene.merged_mesh
    all_vertices = mesh.positions.reshape(-1, 3)
    scene_min = all_vertices.min(axis=0)
    scene_max = all_vertices.max(axis=0)
    offsets = np.array(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ],
        dtype=np.float64,
    )
    scene_bounds = scene_min + offsets * (scene_max - scene_min)

    cascades = build_cascades(
        eye=keyframe.eye,
        target=keyframe.target,
        up=np.array((0.0, 1.0, 0.0)),
        fovy_rad=math.radians(keyframe.fovy_deg),
        aspect=aspect,
        camera_near=float(config["camera_near"]),
        camera_far=float(config["camera_far"]),
        light_direction=scene.light_direction,
        light_up=scene.light_up,
        cascade_count=int(config["cascade_count"]),
        shadow_resolution=int(config["shadow_resolution"]),
        scene_bounds=scene_bounds,
    )
    validate_cascades(cascades, int(config["shadow_resolution"]))

    raw_stack = _render_shadow_stack(
        mesh, cascades, int(config["shadow_resolution"]), snapped=False
    )
    snapped_stack = _render_shadow_stack(
        mesh, cascades, int(config["shadow_resolution"]), snapped=True
    )

    gbuffer = rasterize(
        mesh,
        camera_view_proj,
        int(config["width"]),
        int(config["height"]),
        with_attributes=True,
    )

    common_kwargs = dict(
        camera_view=camera_view,
        normal_bias_scale=float(config["normal_bias_scale"]),
        depth_bias=float(config["depth_bias"]),
        pcf_kernel=int(config["pcf_kernel"]),
        blend_fraction=float(config["blend_fraction"]),
    )
    raw_eval = evaluate_shadows(
        gbuffer, cascades=cascades, depth_maps=list(raw_stack),
        use_snapped=False, **common_kwargs
    )
    snapped_eval = evaluate_shadows(
        gbuffer, cascades=cascades, depth_maps=list(snapped_stack),
        use_snapped=True, **common_kwargs
    )

    identical_indices = bool(
        np.array_equal(raw_eval.cascade_index, snapped_eval.cascade_index)
    )
    identical_blend = bool(np.array_equal(raw_eval.blend, snapped_eval.blend))

    diff = np.abs(snapped_eval.factor - raw_eval.factor).astype(np.float32)
    shadow_mask = shadow_region_mask(raw_eval.factor, snapped_eval.factor)
    metrics_block = {
        "psnr_db": psnr(raw_eval.factor, snapped_eval.factor),
        "max_absolute_error": max_absolute_error(
            raw_eval.factor, snapped_eval.factor
        ),
        "shadow_region_ncc": normalized_cross_correlation(
            raw_eval.factor, snapped_eval.factor, shadow_mask
        ),
        "shadow_region_pixel_count": int(shadow_mask.sum()),
        "identical_zero_error": bool(np.array_equal(
            raw_eval.factor, snapped_eval.factor
        )),
        "psnr_identical_constant_db": 99.0,
    }

    cascade_report = []
    for cascade in cascades:
        idx = cascade.index
        cascade_report.append(
            {
                "index": idx,
                "near": cascade.near,
                "far": cascade.far,
                "texel_world_size": cascade.texel_world_size,
                "raw_depth_checksum": array_checksum(raw_stack[idx]),
                "snapped_depth_checksum": array_checksum(snapped_stack[idx]),
                "snapped_equals_raw": bool(
                    np.array_equal(raw_stack[idx], snapped_stack[idx])
                ),
                "stats": per_cascade_stats(
                    snapped_eval.coverage,
                    gbuffer["hit"],
                    snapped_eval.depth_bias_ndc,
                )[idx],
            }
        )

    archive_path = os.path.join(output_dir, "shadow_depth_snapped.bin")
    write_depth_archive(archive_path, snapped_stack)

    np.save(
        os.path.join(output_dir, "shadow_factor_raw.npy"),
        raw_eval.factor.astype(np.float32),
    )
    np.save(
        os.path.join(output_dir, "shadow_factor_snapped.npy"),
        snapped_eval.factor.astype(np.float32),
    )
    np.save(os.path.join(output_dir, "shadow_diff.npy"), diff)
    write_pgm(
        os.path.join(output_dir, "shadow_diff.pgm"),
        diff,
    )
    write_pgm(
        os.path.join(output_dir, "shadow_factor_snapped.pgm"),
        1.0 - snapped_eval.factor,
    )

    report = {
        "status": "ok",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "config": {key: config[key] for key in sorted(config)},
        "scene": {
            "seed": int(config["seed"]),
            "fingerprint_sha256": fingerprint,
            "mesh_count": len(scene.meshes),
            "triangle_count": int(scene.merged_mesh.positions.shape[0]),
            "light_direction": [
                float(x) for x in scene.light_direction
            ],
        },
        "camera": {
            "frame_time": float(config["frame_time"]),
            "eye": [float(x) for x in keyframe.eye],
            "target": [float(x) for x in keyframe.target],
            "fovy_deg": float(keyframe.fovy_deg),
            "width": int(config["width"]),
            "height": int(config["height"]),
        },
        "cascades": cascade_report,
        "selection_consistency": {
            "cascade_index_identical": identical_indices,
            "blend_identical": identical_blend,
        },
        "metrics": metrics_block,
        "checksums": {
            "raw_factor_sha256": array_checksum(raw_eval.factor),
            "snapped_factor_sha256": array_checksum(snapped_eval.factor),
            "diff_sha256": array_checksum(diff),
            "snapped_depth_archive": os.path.basename(archive_path),
        },
    }
    report_path = os.path.join(output_dir, REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report

