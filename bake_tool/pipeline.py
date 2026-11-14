from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dilation import dilate_maps
from .errors import BakeError
from .mesh import normalized_rows
from .pngio import encode_png
from .preview import render_preview
from .rasterizer import rasterize_texture_space
from .raycast import build_leaf_bvh, cast_rays, interpolate_hit_normals
from .samples import generate_sample
from .tbn import build_vertex_tbn, per_pixel_tbn


REPORT_NAME = "bake_report.json"
ERROR_REPORT_NAME = "error_report.json"
RUNTIME_NAME = "runtime_profile.json"


@dataclass(frozen=True)
class BakeConfig:
    resolution: int = 128
    max_resolution: int = 512
    dilation_radius: int = 2
    miss_threshold: float = 0.02
    normal_convention: str = "opengl"
    normal_dot_threshold: float = 0.98
    scenario: str = "default"
    backface_culling: bool = True
    output_dir: str = "out"

    def snapshot(self) -> dict[str, Any]:
        return {
            "resolution": int(self.resolution),
            "max_resolution": int(self.max_resolution),
            "dilation_radius": int(self.dilation_radius),
            "miss_threshold": float(self.miss_threshold),
            "normal_convention": str(self.normal_convention),
            "normal_dot_threshold": float(self.normal_dot_threshold),
            "scenario": str(self.scenario),
            "backface_culling": bool(self.backface_culling),
            "uv_origin": "bottom_left",
            "ray_direction": "interpolated_geometric_normal_via_tbn_0001",
            "texture_row_order": "top_to_bottom",
        }


@dataclass
class RunResult:
    output_dir: str
    report: dict[str, Any]
    actual_elapsed_ms: float
    texture_bytes: dict[str, bytes]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def write_error_report(
    output_dir: str | Path,
    error: BakeError,
    parameters: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    report = error.as_report(parameters)
    path = output_path / ERROR_REPORT_NAME
    _atomic_write(path, _json_bytes(report))
    return path


def _commit_outputs(output_dir: Path, files: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".bake_stage_", dir=str(output_dir))
    )
    try:
        for name in sorted(files):
            (staging / name).write_bytes(files[name])
        for name in sorted(files):
            os.replace(staging / name, output_dir / name)
        stale_error = output_dir / ERROR_REPORT_NAME
        if stale_error.exists():
            stale_error.unlink()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_config(config: BakeConfig) -> None:
    try:
        resolution = int(config.resolution)
        max_resolution = int(config.max_resolution)
        dilation_radius = int(config.dilation_radius)
        miss_threshold = float(config.miss_threshold)
        dot_threshold = float(config.normal_dot_threshold)
    except (TypeError, ValueError) as exc:
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Bake parameters could not be converted to numeric values.",
            {"error": str(exc)},
            ["Pass integer resolution/radius and finite floating thresholds."],
        ) from exc

    if resolution < 1 or max_resolution < 1:
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Resolution and maximum resolution must be positive.",
            {"resolution": resolution, "max_resolution": max_resolution},
            ["Use resolution >= 1 and max_resolution >= 1."],
        )
    if resolution > max_resolution:
        raise BakeError(
            "E_RESOLUTION_LIMIT",
            "Requested bake resolution exceeds the configured resource cap.",
            {
                "resolution": resolution,
                "max_resolution": max_resolution,
                "requested_texels": resolution * resolution,
            },
            [
                "Lower --resolution.",
                "Raise --max-resolution only after increasing available time/memory budgets.",
            ],
        )
    if dilation_radius < 0:
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Dilation radius must be non-negative.",
            {"dilation_radius": dilation_radius},
            ["Pass --dilation-radius 0 or a positive integer."],
        )
    if not (0.0 <= miss_threshold <= 1.0):
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Miss-rate threshold must be inside [0, 1].",
            {"miss_threshold": miss_threshold},
            ["Pass a threshold such as 0.02 for at most two percent misses."],
        )
    if not (-1.0 <= dot_threshold <= 1.0):
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Normal consistency dot threshold must be inside [-1, 1].",
            {"normal_dot_threshold": dot_threshold},
            ["Use a threshold near 0.98 for an 8-bit normal map."],
        )
    convention = str(config.normal_convention).lower()
    if convention not in {"opengl", "directx"}:
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Normal convention must be 'opengl' or 'directx'.",
            {"normal_convention": str(config.normal_convention)},
            ["Pass --normal-convention opengl or --normal-convention directx."],
        )
    if str(config.scenario) not in {"default", "uv_out_of_range", "missing_high", "miss_rate"}:
        raise BakeError(
            "E_INVALID_ARGUMENT",
            "Unknown deterministic sample scenario.",
            {"scenario": str(config.scenario)},
            ["Choose default, uv_out_of_range, missing_high, or miss_rate."],
        )


def _unit_to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.clip(values, -1.0, 1.0) * 127.5 + 127.5 + 0.5), 0, 255).astype(
        np.uint8
    )


def _scalar_to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.clip(values, 0.0, 1.0) * 255.0 + 0.5), 0, 255).astype(
        np.uint8
    )


def _decode_normal_image(
    image: np.ndarray,
    pixel_y: np.ndarray,
    pixel_x: np.ndarray,
    convention: str,
) -> np.ndarray:
    values = image[pixel_y, pixel_x].astype(np.float64) / 255.0 * 2.0 - 1.0
    if convention == "directx":
        values[:, 1] *= -1.0
    return normalized_rows(values)


def _interpolate_attribute(
    mesh: Mesh,
    values: np.ndarray,
    triangle_index: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    triangles = np.asarray(mesh.triangles)[triangle_index]
    weights = np.asarray(barycentric, dtype=np.float64)
    if weights.ndim == 2 and weights.shape[1] == 2:
        weights = np.stack(
            (1.0 - weights[:, 0] - weights[:, 1], weights[:, 0], weights[:, 1]),
            axis=1,
        )
    if values.ndim == 1:
        return np.einsum("ni,ni->n", weights, values[triangles])
    return np.einsum("ni,nij->nj", weights, values[triangles])


def _channel_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = values[mask].astype(np.float64)
    mean = selected.mean(axis=0)
    variance = selected.var(axis=0)
    return {
        "mean": [float(item) for item in np.atleast_1d(mean)],
        "variance": [float(item) for item in np.atleast_1d(variance)],
        "sample_count": int(selected.shape[0]),
    }


def _normal_mismatch(
    normal_image: np.ndarray,
    convention: str,
    pixel_y: np.ndarray,
    pixel_x: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    normal: np.ndarray,
    actual_world_normal: np.ndarray,
    dot_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    tangent_normal = _decode_normal_image(
        normal_image, pixel_y, pixel_x, convention
    )
    reconstructed = (
        tangent * tangent_normal[:, :1]
        + bitangent * tangent_normal[:, 1:2]
        + normal * tangent_normal[:, 2:3]
    )
    reconstructed = normalized_rows(reconstructed)
    dot_value = np.einsum("ij,ij->i", reconstructed, actual_world_normal)
    return dot_value < dot_threshold, dot_value


def run_bake(config: BakeConfig, output_dir: str | Path | None = None) -> RunResult:
    target_dir = Path(output_dir if output_dir is not None else config.output_dir)
    parameters = config.snapshot()
    wall_start = time.perf_counter()
    try:
        _validate_config(config)
        resolution = int(config.resolution)
        dilation_radius = int(config.dilation_radius)
        convention = str(config.normal_convention).lower()

        sample = generate_sample(str(config.scenario))
        sample.low.validate(require_uvs=True)
        if sample.high is None or sample.high.triangles.size == 0:
            raise BakeError(
                "E_HIGH_MESH_MISSING",
                "The high-poly source mesh is missing or has no triangles.",
                {
                    "sample_id": sample.metadata.get("sample_id"),
                    "scenario": sample.metadata.get("scenario"),
                    "low_triangle_count": int(sample.low.triangles.shape[0]),
                    "high_triangle_count": 0,
                },
                [
                    "Regenerate or reload the high-poly source mesh.",
                    "Verify the high mesh index buffer is non-empty before baking.",
                ],
            )
        sample.high.validate(require_uvs=False)
        if int(sample.low.triangles.max()) >= 65535:
            raise BakeError(
                "E_TRIANGLE_INDEX_LIMIT",
                "Low-poly triangle count exceeds the 16-bit diagnostic index map limit.",
                {
                    "triangle_count": int(sample.low.triangles.shape[0]),
                    "maximum_supported": 65534,
                },
                ["Split the low mesh into fewer than 65535 triangles per bake job."],
            )

        tbn = build_vertex_tbn(sample.low)
        raster = rasterize_texture_space(
            sample.low,
            resolution,
            backface_culling=bool(config.backface_culling),
        )
        coverage = raster.coverage_mask
        coverage_count = int(np.count_nonzero(coverage))
        if coverage_count == 0:
            raise BakeError(
                "E_COVERAGE_EMPTY",
                "Texture-space rasterization produced no covered pixels.",
                {
                    "resolution": resolution,
                    "culled_triangles": raster.culled_triangles,
                    "degenerate_uv_triangles": raster.degenerate_uv_triangles,
                },
                [
                    "Check UV winding and triangle orientation.",
                    "Verify that UVs occupy the zero-to-one domain.",
                ],
            )
        if (raster.triangle_index.max() + 1) >= 65535:
            raise BakeError(
                "E_TRIANGLE_INDEX_LIMIT",
                "Raster ownership uses a triangle id outside the 16-bit output range.",
                {"maximum_owned_triangle_id": int(raster.triangle_index.max())},
                ["Reduce low-poly triangle count or use a smaller diagnostic mesh."],
            )

        pixel_y, pixel_x = np.nonzero(coverage)
        triangle_ids = raster.triangle_index[pixel_y, pixel_x].astype(np.int64)
        bary = raster.barycentric[pixel_y, pixel_x]
        triangle_vertices = np.asarray(sample.low.triangles)[triangle_ids]
        weights = np.clip(bary, 0.0, None)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-15)

        low_position = np.einsum(
            "ni,nij->nj", weights, sample.low.positions[triangle_vertices]
        )
        low_normal = normalized_rows(
            np.einsum("ni,nij->nj", weights, sample.low.normals[triangle_vertices])
        )
        low_tangent, low_bitangent, low_per_pixel_normal = per_pixel_tbn(
            weights, triangle_ids, sample.low, tbn
        )
        low_albedo = _interpolate_attribute(
            sample.low, sample.low.albedo, triangle_ids, weights
        )
        low_roughness = _interpolate_attribute(
            sample.low, sample.low.roughness, triangle_ids, weights
        )
        low_metallic = _interpolate_attribute(
            sample.low, sample.low.metallic, triangle_ids, weights
        )

        ray_origin = low_position + low_per_pixel_normal * 1.0e-5
        ray_direction = low_per_pixel_normal
        bvh = build_leaf_bvh(sample.high, max_leaf_size=8)
        ray_result = cast_rays(ray_origin, ray_direction, sample.high, bvh=bvh)
        hit_mask = ray_result.hit_triangle >= 0
        hit_normals, _ = interpolate_hit_normals(
            sample.high, ray_result.hit_triangle, ray_result.hit_barycentric
        )
        miss_count = int(np.count_nonzero(~hit_mask))
        miss_ratio = miss_count / coverage_count
        if miss_ratio > float(config.miss_threshold):
            raise BakeError(
                "E_MISS_RATE_EXCEEDED",
                "Ray misses exceed the configured threshold; no textures were emitted.",
                {
                    "coverage_pixels": coverage_count,
                    "hit_pixels": int(np.count_nonzero(hit_mask)),
                    "miss_pixels": miss_count,
                    "miss_ratio": float(miss_ratio),
                    "miss_threshold": float(config.miss_threshold),
                    "ray_origin_offset": 1.0e-5,
                },
                [
                    "Increase the high/low surface overlap.",
                    "Check high-poly alignment and low-poly normal orientation.",
                    "Raise --miss-threshold only when fallback material is acceptable.",
                ],
            )

        high_albedo = np.zeros_like(low_albedo)
        high_roughness = np.zeros_like(low_roughness)
        high_metallic = np.zeros_like(low_metallic)
        if np.any(hit_mask):
            high_albedo[hit_mask] = _interpolate_attribute(
                sample.high,
                sample.high.albedo,
                ray_result.hit_triangle[hit_mask],
                ray_result.hit_barycentric[hit_mask],
            )
            high_roughness[hit_mask] = _interpolate_attribute(
                sample.high,
                sample.high.roughness,
                ray_result.hit_triangle[hit_mask],
                ray_result.hit_barycentric[hit_mask],
            )
            high_metallic[hit_mask] = _interpolate_attribute(
                sample.high,
                sample.high.metallic,
                ray_result.hit_triangle[hit_mask],
                ray_result.hit_barycentric[hit_mask],
            )
        albedo = np.where(hit_mask[:, None], high_albedo, low_albedo)
        roughness = np.where(hit_mask, high_roughness, low_roughness)
        metallic = np.where(hit_mask, high_metallic, low_metallic)

        tangent_normal = np.zeros_like(low_normal)
        tangent_normal[:, 2] = 1.0
        actual_world_normal = low_per_pixel_normal.copy()
        if np.any(hit_mask):
            actual_world_normal[hit_mask] = hit_normals[hit_mask]
            tangent_normal[hit_mask] = np.stack(
                (
                    np.einsum(
                        "ij,ij->i",
                        hit_normals[hit_mask],
                        low_tangent[hit_mask],
                    ),
                    np.einsum(
                        "ij,ij->i",
                        hit_normals[hit_mask],
                        low_bitangent[hit_mask],
                    ),
                    np.einsum(
                        "ij,ij->i",
                        hit_normals[hit_mask],
                        low_per_pixel_normal[hit_mask],
                    ),
                ),
                axis=1,
            )
        tangent_normal = normalized_rows(tangent_normal)

        if convention == "directx":
            encoded_tangent_normal = tangent_normal.copy()
            encoded_tangent_normal[:, 1] *= -1.0
        else:
            encoded_tangent_normal = tangent_normal.copy()

        normal_map = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        normal_map[..., 0] = 128
        normal_map[..., 1] = 128
        normal_map[..., 2] = 255
        normal_map[pixel_y, pixel_x] = _unit_to_u8(encoded_tangent_normal)

        albedo_map = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        albedo_map[pixel_y, pixel_x] = _scalar_to_u8(albedo)
        roughness_map = np.full((resolution, resolution), 255, dtype=np.uint8)
        roughness_map[pixel_y, pixel_x] = _scalar_to_u8(roughness)
        metallic_map = np.zeros((resolution, resolution), dtype=np.uint8)
        metallic_map[pixel_y, pixel_x] = _scalar_to_u8(metallic)
        tangent_map = np.full((resolution, resolution, 3), 128, dtype=np.uint8)
        tangent_map[pixel_y, pixel_x] = _unit_to_u8(low_tangent)
        bitangent_map = np.full((resolution, resolution, 3), 128, dtype=np.uint8)
        bitangent_map[pixel_y, pixel_x] = _unit_to_u8(low_bitangent)

        mismatch_mask = np.zeros((resolution, resolution), dtype=np.uint8)
        alternate_mask = np.zeros((resolution, resolution), dtype=np.uint8)
        selected_mismatch_pixels = 0
        alternate_mismatch_pixels = 0
        hit_pixels = int(np.count_nonzero(hit_mask))
        alternate_convention = "directx" if convention == "opengl" else "opengl"
        if hit_pixels:
            selected_mask_rows, _ = _normal_mismatch(
                normal_map,
                convention,
                pixel_y[hit_mask],
                pixel_x[hit_mask],
                low_tangent[hit_mask],
                low_bitangent[hit_mask],
                low_per_pixel_normal[hit_mask],
                actual_world_normal[hit_mask],
                float(config.normal_dot_threshold),
            )
            alternate_mask_rows, _ = _normal_mismatch(
                normal_map,
                alternate_convention,
                pixel_y[hit_mask],
                pixel_x[hit_mask],
                low_tangent[hit_mask],
                low_bitangent[hit_mask],
                low_per_pixel_normal[hit_mask],
                actual_world_normal[hit_mask],
                float(config.normal_dot_threshold),
            )
            selected_mismatch_pixels = int(np.count_nonzero(selected_mask_rows))
            alternate_mismatch_pixels = int(np.count_nonzero(alternate_mask_rows))
            mismatch_mask[pixel_y[hit_mask][selected_mask_rows], pixel_x[hit_mask][selected_mask_rows]] = 255
            alternate_mask[pixel_y[hit_mask][alternate_mask_rows], pixel_x[hit_mask][alternate_mask_rows]] = 255

        stacked_maps = np.concatenate(
            (
                normal_map,
                albedo_map,
                roughness_map[..., None],
                metallic_map[..., None],
                tangent_map,
                bitangent_map,
            ),
            axis=2,
        )
        dilation = dilate_maps(stacked_maps, coverage, dilation_radius)
        (
            dilated_normal_map,
            dilated_albedo_map,
            dilated_roughness_map,
            dilated_metallic_map,
            dilated_tangent_map,
            dilated_bitangent_map,
        ) = (
            dilation.maps[..., 0:3],
            dilation.maps[..., 3:6],
            dilation.maps[..., 6],
            dilation.maps[..., 7],
            dilation.maps[..., 8:11],
            dilation.maps[..., 11:14],
        )

        preview = render_preview(
            sample.low,
            dilated_normal_map,
            dilated_albedo_map,
            dilated_roughness_map,
            dilated_metallic_map,
            dilation.coverage_after,
            tbn,
            resolution,
            convention,
        )

        triangle_index_map = np.zeros((resolution, resolution), dtype=np.uint16)
        triangle_index_map[pixel_y, pixel_x] = (
            raster.triangle_index[pixel_y, pixel_x].astype(np.uint16) + 1
        )
        coverage_before_image = coverage.astype(np.uint8) * 255
        coverage_after_image = dilation.coverage_after.astype(np.uint8) * 255
        difference_image = np.zeros((resolution, resolution), dtype=np.uint8)
        difference_image[coverage] = 255
        difference_image[dilation.added_mask] = 128

        textures: dict[str, np.ndarray] = {
            "normal.png": dilated_normal_map,
            "albedo.png": dilated_albedo_map,
            "roughness.png": dilated_roughness_map,
            "metallic.png": dilated_metallic_map,
            "tangent.png": dilated_tangent_map,
            "bitangent.png": dilated_bitangent_map,
            "coverage_mask.png": coverage_before_image,
            "coverage_mask_dilated.png": coverage_after_image,
            "triangle_index.png": triangle_index_map,
            "dilation_difference.png": difference_image,
            "normal_consistency_mask.png": mismatch_mask,
            "normal_consistency_alternate_mask.png": alternate_mask,
            "preview.png": preview.image,
        }
        encoded_files: dict[str, bytes] = {}
        png_formats: dict[str, str] = {}
        for name, array in textures.items():
            payload, png_format = encode_png(array)
            encoded_files[name] = payload
            png_formats[name] = png_format

        texture_hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(encoded_files.items())
        }
        selected_ratio = (
            selected_mismatch_pixels / hit_pixels if hit_pixels else 0.0
        )
        alternate_ratio = (
            alternate_mismatch_pixels / hit_pixels if hit_pixels else 0.0
        )
        channel_stats = {
            "normal": _channel_stats(normal_map, coverage),
            "albedo": _channel_stats(albedo_map, coverage),
            "roughness": _channel_stats(roughness_map, coverage),
            "metallic": _channel_stats(metallic_map, coverage),
            "tangent": _channel_stats(tangent_map, coverage),
            "bitangent": _channel_stats(bitangent_map, coverage),
        }
        coverage_ratio = coverage_count / float(resolution * resolution)
        fallback_ratio = miss_count / coverage_count
        work_units = (
            resolution * resolution
            + coverage_count * 7
            + raster.candidate_assignments * 4
            + ray_result.candidate_leaf_tests
            + ray_result.ray_triangle_tests
            + dilation.dilated_pixels * 5
            + preview.rendered_pixels * 24
        )
        deterministic_elapsed_ms = round(work_units / 500000.0, 6)

        report = {
            "schema": "offline-pbr-bake/v1",
            "status": "ok",
            "parameters": parameters,
            "sample": sample.metadata,
            "texture_space": {
                "width": resolution,
                "height": resolution,
                "uv_origin": "bottom_left",
                "row_order": "top_to_bottom",
                "pixel_center_rule": "u=(x+0.5)/W, v=1-(y+0.5)/H",
            },
            "outputs": {
                "texture_files": sorted(encoded_files),
                "png_formats": png_formats,
                "texture_sha256": texture_hashes,
            },
            "rasterization": {
                "coverage_pixels": coverage_count,
                "coverage_ratio": float(coverage_ratio),
                "rasterized_triangles": int(raster.rasterized_triangles),
                "culled_backface_triangles": int(raster.culled_triangles),
                "clipped_triangles": int(raster.clipped_triangles),
                "degenerate_uv_triangles": int(raster.degenerate_uv_triangles),
                "pixel_assignments": int(raster.candidate_assignments),
                "one_triangle_per_pixel": True,
            },
            "tbn": {
                "vertex_count": int(sample.low.positions.shape[0]),
                "degenerate_uv_triangles": int(tbn.degenerate_triangles),
                "mirrored_uv_triangles": int(tbn.mirrored_triangles),
                "fallback_vertex_count": int(tbn.fallback_vertices),
                "adjacent_tangent_flips": int(tbn.adjacent_tangent_flips),
                "handedness": "right_handed_B_equals_cross_N_T",
            },
            "raycast": {
                "ray_count": coverage_count,
                "hit_pixels": hit_pixels,
                "fallback_pixels": miss_count,
                "fallback_ratio": float(fallback_ratio),
                "miss_threshold": float(config.miss_threshold),
                "bvh_leaf_count": int(ray_result.leaf_count),
                "candidate_leaf_tests": int(ray_result.candidate_leaf_tests),
                "ray_triangle_tests": int(ray_result.ray_triangle_tests),
                "nearest_hit": True,
            },
            "dilation": {
                "radius": dilation_radius,
                "seam_pixels": int(dilation.seam_pixels),
                "dilated_pixels": int(dilation.dilated_pixels),
                "coverage_before": coverage_count,
                "coverage_after": int(np.count_nonzero(dilation.coverage_after)),
                "added_mask_pixels": int(np.count_nonzero(dilation.added_mask)),
                "valid_pixels_overwritten": 0,
            },
            "normal_consistency": {
                "selected_convention": convention,
                "selected_green_multiplier": -1 if convention == "directx" else 1,
                "selected_y_axis_consistent": selected_ratio
                <= (1.0 - float(config.normal_dot_threshold)),
                "compare_pixels": hit_pixels,
                "mismatch_pixels": selected_mismatch_pixels,
                "mismatch_ratio": float(selected_ratio),
                "dot_threshold": float(config.normal_dot_threshold),
                "alternate_convention": alternate_convention,
                "alternate_y_axis_mismatch_detected": alternate_ratio
                > selected_ratio,
                "alternate_mismatch_pixels": alternate_mismatch_pixels,
                "alternate_mismatch_ratio": float(alternate_ratio),
                "selected_mask_file": "normal_consistency_mask.png",
                "alternate_mask_file": "normal_consistency_alternate_mask.png",
            },
            "channels": channel_stats,
            "preview": {
                "file": "preview.png",
                "width": resolution,
                "height": resolution,
                "rendered_pixels": int(preview.rendered_pixels),
                "rasterized_triangle_pixels": int(preview.rasterized_triangles),
                "culled_backface_triangles": int(preview.culled_triangles),
                "mean_luminance": float(preview.mean_luminance),
                "renderer": "numpy_software_orthographic",
                "normal_convention": convention,
                "uses_same_tbn": True,
            },
            "timing": {
                "timing_model": "deterministic_cost_model_v1",
                "work_units": int(work_units),
                "total_elapsed_ms": float(deterministic_elapsed_ms),
                "actual_elapsed_ms_file": RUNTIME_NAME,
                "note": "Physical wall time is recorded in runtime_profile.json and is excluded from byte determinism.",
            },
        }
        report_bytes = _json_bytes(report)

        encoded_files[REPORT_NAME] = report_bytes
        _commit_outputs(target_dir, encoded_files)
        actual_elapsed_ms = round((time.perf_counter() - wall_start) * 1000.0, 3)
        runtime_profile = {
            "schema": "offline-pbr-bake-runtime/v1",
            "actual_elapsed_ms": actual_elapsed_ms,
            "output_directory": target_dir.name,
            "status": "ok",
        }
        _atomic_write(target_dir / RUNTIME_NAME, _json_bytes(runtime_profile))
        return RunResult(
            output_dir=str(target_dir),
            report=report,
            actual_elapsed_ms=actual_elapsed_ms,
            texture_bytes={
                name: payload
                for name, payload in encoded_files.items()
            },
        )
    except BakeError as error:
        write_error_report(target_dir, error, parameters)
        raise

