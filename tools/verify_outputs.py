#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from softdeferred.gbuffer import GBUFFER_LAYOUT, GBuffer


def verify(output: Path) -> dict:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    gbuffer = GBuffer.load(output / "gbuffer")
    hdr = np.load(output / "hdr_color.npy", allow_pickle=False)
    coverage = gbuffer.coverage.astype(bool)
    covered = int(np.count_nonzero(coverage))

    assert report["status"] == "ok"
    assert report["resolution"] == {
        "width": gbuffer.width,
        "height": gbuffer.height,
    }
    assert hdr.shape == (gbuffer.height, gbuffer.width, 3)
    assert hdr.dtype == np.float32
    assert np.all(np.isfinite(hdr))
    assert np.all(hdr >= 0.0)
    assert report["gbuffer"]["covered_pixels"] == covered
    assert report["hdr"]["shape"] == list(hdr.shape)
    assert report["hdr"]["tone_mapping"] == "none"
    assert np.all(np.isfinite(gbuffer.position[coverage]))
    assert np.all(gbuffer.material_index[coverage] >= 0)
    assert np.all(gbuffer.material_index[coverage] < len(report["scene"]["materials"]))
    assert np.all(np.abs(np.linalg.norm(gbuffer.normal[coverage], axis=1) - 1.0) < 2.0e-4)
    assert np.all(np.abs(np.linalg.norm(gbuffer.tangent[coverage], axis=1) - 1.0) < 2.0e-4)
    assert np.all(
        np.abs(np.linalg.norm(gbuffer.bitangent[coverage], axis=1) - 1.0) < 2.0e-4
    )
    assert np.allclose(gbuffer.depth[~coverage], 1.0)

    shadow_files = set((output / "shadow_maps").glob("*.png"))
    expected_files = {
        output / relative for relative in report["outputs"]["shadow_map_files"]
    }
    assert shadow_files == expected_files
    assert len(shadow_files) == report["shadow_maps"]["map_count"]
    assert report["shadow_maps"]["map_count"] >= 2 + 6 + 1
    assert (
        report["depth_parameters"]["compare_tolerance"]
        == report["shadow_maps"]["compare_tolerance"]
    )
    assert report["depth_parameters"]["used_for_render_and_query"] is True
    assert report["depth_parameters"]["bias"] == report["shadow_maps"]["depth_bias"]
    assert (
        report["depth_parameters"]["slope_scale"]
        == report["shadow_maps"]["slope_scale"]
    )

    interactions = report["light_pixel_interactions"]
    light_types = {
        light["id"]: light["type"] for light in report["scene"]["lights"]
    }
    assert set(interactions) == set(light_types)
    for light_id, count in interactions.items():
        assert 0 <= count <= covered
        if light_types[light_id] == "directional":
            assert count == covered

    culling = report["culling"]
    assert culling["triangles_before_culling"] >= culling["triangles_after_culling"]
    assert (
        culling["triangles_after_frustum_culling"]
        + culling["mesh_frustum_culled_triangles"]
        == culling["triangles_before_culling"]
    )
    assert culling["fragments_written"] >= covered
    assert set(GBUFFER_LAYOUT) == set(report["gbuffer"]["layout"])

    for stage in (
        "scene_load",
        "geometry_prepare",
        "gbuffer_rasterization",
        "shadow_map_generation",
        "deferred_shading",
        "output_write",
    ):
        assert stage in report["timings_ms"]
        assert report["timings_ms"][stage] >= 0.0
    assert report["peak_memory"]["estimate_mb"] > 0.0
    assert (output / "depth.png").is_file()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a renderer output directory.")
    parser.add_argument("output", help="renderer output directory")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        report = verify(output)
    except Exception as exc:
        print("verification failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(
        "verification OK: %dx%d, %d covered pixels, %d shadow maps, %d interactions"
        % (
            report["resolution"]["width"],
            report["resolution"]["height"],
            report["gbuffer"]["covered_pixels"],
            report["shadow_maps"]["map_count"],
            sum(report["light_pixel_interactions"].values()),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
