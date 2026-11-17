"""Run the deterministic CPU-renderer acceptance checks and write a report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lrt.errors import SceneError
from lrt.materials import calibration_report
from lrt.renderer import Renderer
from lrt.scene import create_default_scene, parse_scene


def render_with(
    width: int,
    height: int,
    spp: int,
    aa: int,
    shadow_samples: int,
) -> Any:
    description = create_default_scene(seed=158, width=width, height=height)
    description["render"].update(
        {
            "samples_per_pixel": spp,
            "aa_samples": aa,
            "shadow_samples": shadow_samples,
        }
    )
    return Renderer(parse_scene(description)).render()


def check_error_codes() -> List[Dict[str, Any]]:
    cases = []
    mutations = {
        "missing_field": (3, lambda data: data["camera"].pop("look_at")),
        "negative_radius": (4, lambda data: data["geometry"][2].update({"radius": -1.0})),
        "unknown_material_type": (
            5,
            lambda data: data["materials"].update({"unknown": {"type": "glass"}}),
        ),
        "empty_lights": (6, lambda data: data.update({"lights": []})),
        "ior_out_of_range": (
            7,
            lambda data: data["materials"]["ceramic"]["coating"].update({"ior": 0.5}),
        ),
        "thickness_out_of_range": (
            7,
            lambda data: data["materials"]["ceramic"]["coating"].update({"thickness": 1.0}),
        ),
        "invalid_aa": (8, lambda data: data["render"].update({"aa_samples": 3})),
    }
    for name, (expected, mutate) in mutations.items():
        data = copy.deepcopy(create_default_scene(seed=158, width=16, height=16))
        mutate(data)
        actual = 0
        try:
            parse_scene(data)
        except SceneError as exc:
            actual = exc.code
        cases.append(
            {
                "case": name,
                "expected_error_code": expected,
                "actual_error_code": actual,
                "passed": actual == expected,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="skip the two 256x256 default renders (useful for a quick smoke run)",
    )
    args = parser.parse_args()

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report: Dict[str, Any] = {"schema_version": 1}

    print("[1/5] layer-energy calibration")
    default_scene = parse_scene(create_default_scene(seed=158, width=256, height=256))
    calibration = calibration_report(default_scene.materials.values())
    report["energy"] = {
        "all_in_range": calibration["all_in_range"],
        "minimum_reflectance": calibration["minimum_reflectance"],
        "maximum_reflectance": calibration["maximum_reflectance"],
        "warnings": calibration["warnings"],
    }

    print("[2/5] integer input-error codes")
    report["error_codes"] = check_error_codes()

    print("[3/5] shadow convergence (96x96, 1 vs 16 area samples)")
    low = render_with(96, 96, spp=2, aa=1, shadow_samples=1)
    high = render_with(96, 96, spp=2, aa=1, shadow_samples=16)
    report["shadow_convergence"] = {
        "low_shadow_samples": 1,
        "high_shadow_samples": 16,
        "low_shadow_noise": low.stats["shadow_noise"],
        "high_shadow_noise": high.stats["shadow_noise"],
        "low_average_shadow_rays_per_pixel": low.stats["average_shadow_rays_per_pixel"],
        "high_average_shadow_rays_per_pixel": high.stats["average_shadow_rays_per_pixel"],
        "noise_decreased": high.stats["shadow_noise"] < low.stats["shadow_noise"],
    }

    print("[4/5] deterministic AA (64x64, 1x vs 4x; 4x rendered twice)")
    aa1 = render_with(64, 64, spp=1, aa=1, shadow_samples=4)
    aa4 = render_with(64, 64, spp=1, aa=4, shadow_samples=4)
    aa4_repeat = render_with(64, 64, spp=1, aa=4, shadow_samples=4)
    report["anti_aliasing"] = {
        "aa1_edge_gradient": aa1.stats["edge_gradient"],
        "aa4_edge_gradient": aa4.stats["edge_gradient"],
        "edge_gradient_ratio": aa4.stats["edge_gradient"] / aa1.stats["edge_gradient"],
        "aa4_serial_pixel_reproducible": bool(np.array_equal(aa4.rgb, aa4_repeat.rgb)),
        "aa4_serial_stats_reproducible": aa4.stats == aa4_repeat.stats,
    }

    if not args.skip_full:
        print("[5/5] default 256x256 render, rendered twice for exact comparison")
        first = Renderer(default_scene).render()
        second = Renderer(default_scene).render()
        first.save_png(str(artifacts / "default_256.png"))
        first.save_stats(str(artifacts / "default_256.json"))
        image_hash = hashlib.sha256(first.rgb.tobytes()).hexdigest()
        report["default_256"] = {
            "pixel_reproducible": bool(np.array_equal(first.rgb, second.rgb)),
            "stats_reproducible": first.stats == second.stats,
            "image_sha256": image_hash,
            "stats_sha256": hashlib.sha256(
                json.dumps(first.stats, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "total_rays": first.stats["total_rays"],
            "average_rays_per_pixel": first.stats["average_rays_per_pixel"],
            "average_shadow_rays_per_pixel": first.stats["average_shadow_rays_per_pixel"],
            "depth_clip_count": first.stats["depth_clip_count"],
            "nan_pixels": first.stats["nan_pixels"],
            "max_layer_reflectance": first.stats["max_layer_reflectance"],
            "shadow_noise": first.stats["shadow_noise"],
            "edge_gradient": first.stats["edge_gradient"],
        }
    else:
        print("[5/5] skipped full default render")

    checks = [
        report["energy"]["all_in_range"],
        all(item["passed"] for item in report["error_codes"]),
        report["shadow_convergence"]["noise_decreased"],
        report["anti_aliasing"]["edge_gradient_ratio"] < 0.8,
        report["anti_aliasing"]["aa4_serial_pixel_reproducible"],
        report["anti_aliasing"]["aa4_serial_stats_reproducible"],
    ]
    if "default_256" in report:
        checks.extend(
            [
                report["default_256"]["pixel_reproducible"],
                report["default_256"]["stats_reproducible"],
                report["default_256"]["nan_pixels"] == 0,
            ]
        )
    report["all_checks_passed"] = all(checks)
    with open(artifacts / "acceptance_report.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

