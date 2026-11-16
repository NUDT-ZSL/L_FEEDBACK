from __future__ import annotations

import platform
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .errors import RendererError, SceneConfigError
from .gbuffer import GBUFFER_LAYOUT
from .lighting import shade_deferred
from .linear import json_safe
from .output import write_outputs
from .raster import camera_matrices, prepare_geometry, rasterize_gbuffer
from .scene import load_scene
from .shadows import build_shadow_maps


def _elapsed_ms(start_ns: int) -> float:
    return round((time.perf_counter_ns() - start_ns) / 1.0e6, 6)


def _rss_peak_mb() -> Optional[float]:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            value /= 1024.0 * 1024.0
        else:
            value /= 1024.0
        return round(value, 6)
    except (ImportError, AttributeError, OSError):
        return None


def _buffer_bytes(gbuffer, hdr, geometry, shadows) -> int:
    total = int(hdr.nbytes + geometry.batch.byte_size())
    total += int(sum(array.nbytes for array in gbuffer.arrays().values()))
    total += int(sum(shadow.depth.nbytes for shadow in shadows.maps))
    return total


def render_scene(
    scene_path: str,
    output_directory: str,
    overrides: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    total_start = time.perf_counter_ns()
    tracemalloc.start()
    stage_start = time.perf_counter_ns()
    scene = load_scene(scene_path, overrides)
    scene_load_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter_ns()
    try:
        geometry = prepare_geometry(scene)
    except ValueError as exc:
        raise SceneConfigError(str(exc))
    geometry_prepare_ms = _elapsed_ms(stage_start)

    camera_view, camera_projection = camera_matrices(scene)
    view_projection = camera_projection @ camera_view
    stage_start = time.perf_counter_ns()
    gbuffer, raster_stats = rasterize_gbuffer(
        geometry.visible_batch,
        view_projection,
        scene.camera.position,
        scene.width,
        scene.height,
    )
    gbuffer_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter_ns()
    shadows = build_shadow_maps(scene, geometry.batch)
    shadows_ms = _elapsed_ms(stage_start)

    stage_start = time.perf_counter_ns()
    try:
        hdr, interactions, interaction_details, shadow_queries = shade_deferred(
            gbuffer, scene, shadows, camera_view
        )
    except ValueError as exc:
        raise RendererError("deferred shading failed: %s" % exc)
    lighting_ms = _elapsed_ms(stage_start)

    current_bytes, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracked_bytes = _buffer_bytes(gbuffer, hdr, geometry, shadows)
    tracemalloc_peak_mb = tracemalloc_peak / (1024.0 * 1024.0)
    tracked_mb = tracked_bytes / (1024.0 * 1024.0)
    peak_estimate_mb = max(tracemalloc_peak_mb, tracked_mb)
    tracemalloc.stop()

    covered = int(np.count_nonzero(gbuffer.coverage))
    hdr_min = float(np.min(hdr))
    hdr_max = float(np.max(hdr))
    hdr_mean = float(np.mean(hdr))
    culling = geometry.report_dict()
    culling.update(
        {
            "triangles_after_culling": raster_stats["triangles_submitted"],
            "backface_culled_triangles": raster_stats["backface_culled_triangles"],
            "near_far_culled_triangles": raster_stats["near_far_culled_triangles"],
            "offscreen_culled_triangles": raster_stats["offscreen_culled_triangles"],
            "rasterized_triangles": raster_stats["rasterized_triangles"],
            "fragments_written": raster_stats["fragments_written"],
        }
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "scene": {
            "path": str(scene.source_path),
            "sha256": scene.source_sha256,
            "camera": {
                "position": scene.camera.position.tolist(),
                "target": scene.camera.target.tolist(),
                "up": scene.camera.up.tolist(),
                "fov_y_deg": scene.camera.fov_y_deg,
                "near": scene.camera.near,
                "far": scene.camera.far,
            },
            "lights": [light.report_dict() for light in scene.lights],
            "materials": [material.report_dict() for material in scene.materials],
            "mesh_count": len(scene.meshes),
        },
        "resolution": {"width": scene.width, "height": scene.height},
        "gbuffer": {
            "layout": GBUFFER_LAYOUT,
            "covered_pixels": covered,
            "background_depth": 1.0,
        },
        "shadow_maps": shadows.report_dict(),
        "depth_parameters": {
            "bias": scene.shadows.depth_bias,
            "slope_scale": scene.shadows.slope_scale,
            "compare_tolerance": scene.shadows.compare_tolerance,
            "used_for_render_and_query": True,
        },
        "culling": culling,
        "light_pixel_interactions": interactions,
        "light_interaction_details": interaction_details,
        "shadow_queries": int(shadow_queries),
        "hdr": {
            "file": "hdr_color.npy",
            "shape": list(hdr.shape),
            "dtype": "float32",
            "tone_mapping": "none",
            "min": hdr_min,
            "max": hdr_max,
            "mean": hdr_mean,
            "finite": bool(np.all(np.isfinite(hdr))),
        },
        "timings_ms": {
            "scene_load": scene_load_ms,
            "geometry_prepare": geometry_prepare_ms,
            "gbuffer_rasterization": gbuffer_ms,
            "shadow_map_generation": shadows_ms,
            "deferred_shading": lighting_ms,
            "total_render": _elapsed_ms(total_start),
        },
        "timing_method": "time.perf_counter_ns single interval per stage",
        "peak_memory": {
            "estimate_mb": round(peak_estimate_mb, 6),
            "tracemalloc_peak_mb": round(tracemalloc_peak_mb, 6),
            "tracked_numpy_buffers_mb": round(tracked_mb, 6),
            "tracemalloc_current_mb": round(current_bytes / (1024.0 * 1024.0), 6),
            "rss_peak_mb": _rss_peak_mb(),
        },
        "outputs": {
            "hdr": "hdr_color.npy",
            "depth_image": "depth.png",
            "gbuffer_directory": "gbuffer",
            "shadow_map_directory": "shadow_maps",
            "shadow_map_files": [
                "shadow_maps/" + shadow.filename for shadow in shadows.maps
            ],
            "report": "report.json",
        },
    }
    report = json_safe(report)

    output_path = Path(output_directory).expanduser().resolve()
    write_outputs(
        output_path,
        gbuffer,
        hdr,
        gbuffer.depth,
        shadows,
        report,
        overwrite=overwrite,
    )
    return report
