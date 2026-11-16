from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import glfw
import numpy as np
from PIL import Image

from .app import _create_window
from .clustering import VirtualPointLight, cluster_lights
from .config import DiagnosticLog, load_scene_config
from .diagnostics_cli import configure_logging
from .lighting import accumulate_reference, vpl_contribution
from .renderer import Renderer
from .scene_generator import generate_scene


def _png_difference(path_a: Path, path_b: Path) -> float:
    first = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float64)
    second = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float64)
    return float(np.mean(np.abs(first - second)))


def _energy_checks() -> dict[str, float | bool]:
    from proc_gi.config import Material

    material = Material(
        id=1,
        name="energy-test",
        type="BlinnPhong",
        albedo=(0.55, 0.42, 0.30),
        roughness=0.4,
        specular=0.35,
        shininess=72.0,
    )
    base = VirtualPointLight(
        id="energy-base",
        position=(0.0, 2.0, 0.0),
        color=(1.0, 0.8, 0.6),
        intensity=10.0,
        direction=(0.0, -1.0, 0.0),
        direction_range_deg=0.0,
        is_spot=False,
        range=5.0,
        radius=1.0,
        member_ids=("a",),
        source_count=1,
        signature="a",
    )
    half_a = VirtualPointLight(
        **{**base.__dict__, "id": "half-a", "intensity": 5.0, "signature": "ha"}
    )
    half_b = VirtualPointLight(
        **{**base.__dict__, "id": "half-b", "intensity": 5.0, "signature": "hb"}
    )
    point = np.asarray((0.25, 0.75, 0.15))
    normal = np.asarray((0.0, 1.0, 0.0))
    view = np.asarray((0.0, 1.0, 0.4))
    single = accumulate_reference([base], material, point, normal, view)
    split = accumulate_reference([half_a, half_b], material, point, normal, view)
    additive = accumulate_reference([base, half_a], material, point, normal, view)
    return {
        "split_equals_single_max_abs": float(np.abs(single - split).max()),
        "additive_gt_single": bool(np.all(additive >= single - 1.0e-9)),
        "additive_equals_sum_max_abs": float(
            np.abs(additive - (single + accumulate_reference([half_a], material, point, normal, view))).max()
        ),
        "single_rgb": [float(value) for value in single],
        "split_rgb": [float(value) for value in split],
        "additive_rgb": [float(value) for value in additive],
    }


def run_validation(args: argparse.Namespace) -> int:
    configure_logging()
    diagnostics = DiagnosticLog()
    config = load_scene_config(args.scene, diagnostics)
    _, mesh = generate_scene(config.seed, diagnostics)
    width = args.width
    height = args.height
    window = _create_window(width, height, visible=False)
    renderer = Renderer(config, mesh, diagnostics, width=width, height=height)
    try:
        initial_stats = renderer.render()
        initial_hdr = renderer.read_hdr()
        assert np.isfinite(initial_hdr).all()
        assert float(initial_hdr.max()) > 0.0
        color_path = Path("artifacts/frame_tonemapped.png")
        linear_path = Path("artifacts/frame_linear.png")
        renderer.save_screenshot(color_path, tonemap=True)
        renderer.save_screenshot(linear_path, tonemap=False)
        tone_difference = _png_difference(color_path, linear_path)

        selected = config.lights[0]
        x, y, z = selected.position
        renderer.update_light(
            selected.id,
            intensity=selected.intensity * 1.20,
            position=[x + 0.08, y, z],
        )
        incremental_stats = renderer.render()
        incremental_hdr = renderer.read_hdr()
        full_stats = renderer.render(rebuild=True)
        full_hdr = renderer.read_hdr()
        incremental_error = np.abs(incremental_hdr - full_hdr)
        assert float(incremental_error.max()) < 0.08
        assert incremental_stats.cube_builds < initial_stats.cube_builds
        assert incremental_stats.cube_reused > 0

        renderer.save_screenshot(Path("artifacts/frame_after_incremental.png"), tonemap=True)

        clustered = cluster_lights(config.lights, config.clustering)
        cluster_path = Path("artifacts/vpl_clusters.json")
        cluster_path.write_text(
            json.dumps(
                {
                    "source_light_count": len(config.lights),
                    "vpl_count": len(clustered),
                    "vpls": [vpl.as_dict() for vpl in clustered],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        renderer.set_clustering(max_vpls=12, vpl_capacity=8)
        degraded_stats = renderer.render()
        degraded_path = Path("artifacts/frame_degraded.png")
        renderer.save_screenshot(degraded_path, tonemap=True)
        diagnostics.write(config.render.warning_path)
        warning_payload = json.loads(Path(config.render.warning_path).read_text(encoding="utf-8"))
        truncation_records = [
            record
            for record in warning_payload["warnings"]
            if record["code"] == "VPL_CAPACITY_TRUNCATED"
        ]
        assert truncation_records
        assert degraded_stats.truncated_vpl_count > 0
        assert degraded_stats.active_vpl_count == 8
        assert truncation_records[0]["context"]["truncated_source_light_ids"]
        assert truncation_records[0]["context"]["degraded_state"]["render_continues"]

        material_types = sorted({material.type for material in config.materials})
        assert {"Lambertian", "OrenNayar", "BlinnPhong"}.issubset(set(material_types))
        mesh_hash = hashlib.sha256(mesh.vertices.tobytes() + mesh.indices.tobytes()).hexdigest()
        _config_again, mesh_again = generate_scene(config.seed)
        mesh_hash_again = hashlib.sha256(
            mesh_again.vertices.tobytes() + mesh_again.indices.tobytes()
        ).hexdigest()
        assert mesh_hash == mesh_hash_again

        energy = _energy_checks()
        assert energy["split_equals_single_max_abs"] < 1.0e-9
        assert energy["additive_gt_single"]
        assert energy["additive_equals_sum_max_abs"] < 1.0e-9

        report = {
            "status": "passed",
            "seed": config.seed,
            "mesh": {
                "vertices": len(mesh.vertices),
                "triangles": mesh.triangle_count,
                "material_groups": len(mesh.groups),
                "group_index_ranges": [
                    {
                        "material": group.material_name,
                        "index_offset": group.index_offset,
                        "index_count": group.index_count,
                    }
                    for group in mesh.groups
                ],
                "sha256": mesh_hash,
                "deterministic_regeneration": True,
            },
            "materials": {
                "count": len(config.materials),
                "types": material_types,
                "unknown_fallback_code": "MAT_UNKNOWN_TYPE (covered by unit test)",
            },
            "lights": {
                "source_count": len(config.lights),
                "initial_vpl_clusters": len(clustered),
                "cluster_output": str(cluster_path),
                "initial_stats": initial_stats.as_dict(),
            },
            "incremental_update": {
                "stats": incremental_stats.as_dict(),
                "incremental_vs_full_max_abs": float(incremental_error.max()),
                "incremental_vs_full_mean_abs": float(incremental_error.mean()),
                "full_rebuild_stats": full_stats.as_dict(),
                "shadow_consistency_verified_against_full_rebuild": True,
            },
            "hdr": {
                "minimum": float(initial_hdr.min()),
                "maximum": float(initial_hdr.max()),
                "mean": float(initial_hdr.mean()),
                "finite": bool(np.isfinite(initial_hdr).all()),
            },
            "tonemapping": {
                "ace_and_srgb_png": str(color_path),
                "linear_debug_png": str(linear_path),
                "same_hdr_source": True,
                "mean_absolute_png_difference": tone_difference,
            },
            "energy_conservation": energy,
            "truncation": {
                "stats": degraded_stats.as_dict(),
                "warning_path": config.render.warning_path,
                "warning": truncation_records[0],
                "degraded_png": str(degraded_path),
            },
            "screenshots": [
                str(color_path),
                str(linear_path),
                "artifacts/frame_after_incremental.png",
                str(degraded_path),
            ],
        }
        report_path = Path("artifacts/validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        renderer.close()
        glfw.destroy_window(window)
        glfw.terminate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run actual OpenGL validation and write a report.")
    parser.add_argument("--scene", default="scene.json")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    return parser


def main() -> int:
    return run_validation(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

