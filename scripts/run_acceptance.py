from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "acceptance"


def run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "deferred_renderer", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
        check=False,
    )


def load_stats(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "stats.json").read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    failures: list[str] = []

    scene_path = OUT / "scene.json"
    result = run(["--write-default-scene", str(scene_path)])
    checks["scene_generation"] = {"passed": result.returncode == 0}
    if result.returncode != 0:
        failures.append("scene_generation")

    common = [
        "--scene",
        str(scene_path),
        "--width",
        "160",
        "--height",
        "90",
    ]
    outputs = {
        "headless_a": OUT / "headless_a",
        "headless_b": OUT / "headless_b",
        "window": OUT / "window",
    }
    runs = {
        "headless_a": run([*common, "--out", str(outputs["headless_a"]), "--mode", "headless"]),
        "headless_b": run([*common, "--out", str(outputs["headless_b"]), "--mode", "headless"]),
        "window": run(
            [
                *common,
                "--out",
                str(outputs["window"]),
                "--mode",
                "window",
                "--window-hidden",
            ]
        ),
    }
    for name, completed in runs.items():
        passed = completed.returncode == 0
        checks[f"render_{name}"] = {
            "passed": passed,
            "exit_code": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
        if not passed:
            failures.append(f"render_{name}")

    if not failures:
        headless_a = load_stats(outputs["headless_a"])
        headless_b = load_stats(outputs["headless_b"])
        window = load_stats(outputs["window"])
        images = {
            name: np.asarray(Image.open(outputs[name] / "frame_0000.png"))
            for name in outputs
        }
        pixel_equal = (
            np.array_equal(images["headless_a"], images["headless_b"])
            and np.array_equal(images["headless_a"], images["window"])
        )
        hash_equal = (
            headless_a["output"]["pixel_sha256"]
            == headless_b["output"]["pixel_sha256"]
            == window["output"]["pixel_sha256"]
        )
        checks["repeat_png_pixel_hash_stable"] = {
            "passed": hash_equal,
            "pixel_sha256": headless_a["output"]["pixel_sha256"],
        }
        checks["headless_window_pixel_identical"] = {
            "passed": pixel_equal,
            "max_channel_delta": int(
                np.abs(
                    images["headless_a"].astype(np.int16)
                    - images["window"].astype(np.int16)
                ).max()
            ),
        }
        checks["reference_threshold"] = {
            "passed": bool(headless_a["reference_check"]["passed"]),
            **headless_a["reference_check"],
        }
        checks["g_buffer"] = {
            "passed": bool(
                headless_a["g_buffer"]["measured"]["valid_geometry_pixels"] > 0
                and headless_a["g_buffer"]["measured"]["front_side_pixels"] > 0
                and headless_a["g_buffer"]["measured"]["back_side_pixels"] > 0
            ),
            "attachments": [
                {
                    "name": item["name"],
                    "format": item["format"],
                    "range": item["range"],
                }
                for item in headless_a["g_buffer"]["attachments"]
            ],
            "measured": {
                "valid_geometry_pixels": headless_a["g_buffer"]["measured"][
                    "valid_geometry_pixels"
                ],
                "front_side_pixels": headless_a["g_buffer"]["measured"][
                    "front_side_pixels"
                ],
                "back_side_pixels": headless_a["g_buffer"]["measured"][
                    "back_side_pixels"
                ],
                "linear_depth": headless_a["g_buffer"]["measured"][
                    "linear_depth_geometry"
                ],
                "normal_length": headless_a["g_buffer"]["measured"]["normal_length"],
                "material_ids": headless_a["g_buffer"]["measured"][
                    "material_ids_present"
                ],
            },
        }
        tiling = headless_a["tile_lighting"]
        checks["tile_light_lists"] = {
            "passed": bool(
                len(tiling["lists"]) == tiling["tile_count"]
                and tiling["max_lights_in_tile"] > 0
                and tiling["average_lights_per_tile"] > 0.0
            ),
            "tile_size": tiling["tile_size"],
            "tile_count": tiling["tile_count"],
            "average_lights_per_tile": tiling["average_lights_per_tile"],
            "max_lights_in_tile": tiling["max_lights_in_tile"],
            "light_count_histogram": tiling["light_count_histogram"],
        }
        checks["dof_histogram"] = {
            "passed": bool(
                sum(headless_a["dof"]["all_pixel_counts"])
                == headless_a["resolution"]["width"]
                * headless_a["resolution"]["height"]
            ),
            "composition_order": headless_a["dof"]["composition_order"],
            "bin_edges_px": headless_a["dof"]["bin_edges_px"],
            "all_pixel_counts": headless_a["dof"]["all_pixel_counts"],
            "valid_radius_mean_px": headless_a["dof"]["valid_radius_mean_px"],
            "valid_radius_max_px": headless_a["dof"]["valid_radius_max_px"],
        }
        checks["hashes"] = {
            "passed": headless_a["scene"]["geometry_sha256"]
            == window["scene"]["geometry_sha256"]
            and headless_a["scene"]["texture_sha256"]
            == window["scene"]["texture_sha256"],
            "scene_sha256": headless_a["scene"]["sha256"],
            "geometry_sha256": headless_a["scene"]["geometry_sha256"],
            "texture_sha256": headless_a["scene"]["texture_sha256"],
            "combined_asset_sha256": headless_a["scene"]["combined_asset_sha256"],
        }
        for name, check in checks.items():
            if isinstance(check, dict) and check.get("passed") is False:
                failures.append(name)

    source = json.loads(scene_path.read_text(encoding="utf-8"))
    mutations = {
        "missing_attachment": (
            lambda data: data["render"]["g_buffer"]["attachments"].remove(
                "albedo_roughness"
            ),
            "E_MISSING_ATTACHMENT",
        ),
        "bad_light": (
            lambda data: data["lights"][0].__setitem__("radius", 0.0),
            "E_BAD_LIGHT",
        ),
        "zero_area": (
            lambda data: data["objects"][-1]["size"].__setitem__(2, 0.0),
            "E_ZERO_AREA_GEOMETRY",
        ),
    }
    for name, (mutate, expected_code) in mutations.items():
        invalid = json.loads(json.dumps(source))
        mutate(invalid)
        invalid_path = OUT / f"invalid_{name}.json"
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        stats_path = OUT / "errors" / name
        completed = run(
            [
                "--scene",
                str(invalid_path),
                "--out",
                str(stats_path),
                "--mode",
                "headless",
            ]
        )
        error_doc = json.loads((stats_path / "stats.json").read_text())
        code = error_doc.get("error", {}).get("code")
        passed = completed.returncode != 0 and code == expected_code
        checks[f"error_{name}"] = {
            "passed": passed,
            "exit_code": completed.returncode,
            "code": code,
        }
        if not passed:
            failures.append(f"error_{name}")

    if sys.platform.startswith("linux"):
        env = dict(os.environ)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        stats_path = OUT / "errors" / "window_create"
        completed = run(
            [
                "--scene",
                str(scene_path),
                "--out",
                str(stats_path),
                "--mode",
                "window",
                "--window-hidden",
            ],
            env=env,
        )
        error_doc = json.loads((stats_path / "stats.json").read_text())
        code = error_doc.get("error", {}).get("code")
        passed = completed.returncode != 0 and code == "E_WINDOW_CREATE"
        checks["error_window_create"] = {
            "passed": passed,
            "exit_code": completed.returncode,
            "code": code,
        }
        if not passed:
            failures.append("error_window_create")

    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    checks["pytest"] = {
        "passed": pytest_result.returncode == 0,
        "exit_code": pytest_result.returncode,
        "summary": pytest_result.stdout.strip().splitlines()[-1]
        if pytest_result.stdout.strip()
        else "",
    }
    if pytest_result.returncode != 0:
        failures.append("pytest")

    report = {
        "ok": not failures,
        "failures": failures,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "python": sys.version,
        "checks": checks,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    print(f"REPORT {OUT / 'report.json'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

