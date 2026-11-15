#!/usr/bin/env python3
"""Run the acceptance checks and write reproducible verification artifacts."""

from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matrender.color import linear_to_srgb8, luminance
from matrender.renderer import RenderOptions, make_render_plan, render
from matrender.scene import load_scene


VERIFICATION = ROOT / "verification"


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _scalar_display_reference(value: float) -> int:
    x = max(float(value), 0.0)
    aces = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    aces = min(max(aces, 0.0), 1.0)
    encoded = (
        12.92 * aces
        if aces <= 0.0031308
        else 1.055 * math.pow(aces, 1.0 / 2.4) - 0.055
    )
    return min(max(int(math.floor(encoded * 255.0 + 0.5)), 0), 255)


def _color_metrics() -> dict[str, object]:
    values = np.array(
        [
            0.0,
            1.0e-6,
            0.0002,
            0.003,
            0.01,
            0.05,
            0.18,
            0.5,
            1.0,
            2.0,
            8.0,
            100.0,
            10000.0,
        ]
    )
    hdr = np.stack([values, np.sqrt(values), values[::-1]], axis=-1)
    actual = linear_to_srgb8(hdr)
    expected = np.empty(hdr.shape, dtype=np.uint8)
    for index in np.ndindex(hdr.shape[:-1]):
        for channel in range(3):
            expected[index + (channel,)] = _scalar_display_reference(
                hdr[index + (channel,)]
            )
    return {
        "samples": int(hdr.size),
        "maximum_quantization_error": int(
            np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))
        ),
        "tolerance_quantization_units": 2,
    }


def _thread_metrics() -> dict[str, object]:
    scene = load_scene(ROOT / "scenes" / "material_grid.json")
    results = {}
    for threads in (1, 2, 4):
        plan = make_render_plan(
            scene,
            RenderOptions(
                samples=2,
                threads=threads,
                tile_size=16,
                max_memory_bytes=1024 * 1024 * 1024,
                width=96,
                height=72,
            ),
        )
        results[threads] = render(scene, plan)
    reference = results[1].linear_hdr.astype(np.float64)
    differences = {
        str(threads): float(
            np.max(np.abs(reference - results[threads].linear_hdr.astype(np.float64)))
        )
        for threads in (2, 4)
    }
    return {
        "resolution": [96, 72],
        "samples": 2,
        "max_abs_pixel_difference_vs_1_thread": differences,
        "bitwise_equal": all(value == 0.0 for value in differences.values()),
        "reports_equal": (
            results[1].material_regions == results[2].material_regions
            and results[1].material_regions == results[4].material_regions
        ),
    }


def _repeatability_metrics() -> dict[str, object]:
    scene = load_scene(ROOT / "scenes" / "material_grid.json")
    plan = make_render_plan(
        scene,
        RenderOptions(
            samples=2,
            threads=4,
            tile_size=16,
            max_memory_bytes=1024 * 1024 * 1024,
            width=96,
            height=72,
        ),
    )
    first = render(scene, plan).linear_hdr.astype(np.float64)
    second = render(scene, plan).linear_hdr.astype(np.float64)
    return {
        "maximum_absolute_pixel_difference": float(
            np.max(np.abs(first - second))
        ),
        "tolerance": 1.0e-6,
    }


def _ssao_metrics() -> dict[str, object]:
    on_image = np.load(
        VERIFICATION / "ssao_on" / "linear_hdr.npy", allow_pickle=False
    ).astype(np.float64)
    off_image = np.load(
        VERIFICATION / "ssao_off" / "linear_hdr.npy", allow_pickle=False
    ).astype(np.float64)
    return {
        "resolution": [int(on_image.shape[1]), int(on_image.shape[0])],
        "samples": 2,
        "mean_absolute_linear_difference": float(
            np.mean(np.abs(on_image - off_image))
        ),
        "required_minimum": 0.02,
    }


def _report_metrics() -> dict[str, object]:
    output_dir = VERIFICATION / "material_grid"
    hdr = np.load(output_dir / "linear_hdr.npy", allow_pickle=False)
    report = json.loads(
        (output_dir / "material_report.json").read_text(encoding="utf-8")
    )
    masks = np.load(output_dir / "sphere_masks.npz", allow_pickle=False)
    maximum_relative_error = 0.0
    maximum_hue_error = 0.0
    for entry in report["regions"]:
        mask = masks[f"sphere_{entry['index']:03d}"]
        values = hdr[mask].astype(np.float64)
        average = float(np.mean(luminance(values)))
        maximum = float(np.max(luminance(values)))
        maximum_relative_error = max(
            maximum_relative_error,
            abs(average - float(entry["average_luminance"]))
            / max(abs(average), 1.0e-12),
            abs(maximum - float(entry["max_luminance"]))
            / max(abs(maximum), 1.0e-12),
        )
        mean_rgb = np.mean(values, axis=0)
        normalized = mean_rgb / np.max(mean_rgb)
        hue = colorsys.rgb_to_hsv(*normalized.tolist())[0] * 360.0
        maximum_hue_error = max(
            maximum_hue_error,
            abs(hue - float(entry["dominant_hue_deg"])),
        )
    return {
        "entries": len(report["regions"]),
        "mask_entries": len(masks.files),
        "maximum_relative_stat_error": maximum_relative_error,
        "maximum_hue_error_degrees": maximum_hue_error,
        "required_maximum_relative_error": 0.01,
    }


def main() -> int:
    VERIFICATION.mkdir(parents=True, exist_ok=True)

    test_process = _run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    )
    if test_process.returncode != 0:
        print(test_process.stdout, file=sys.stderr)
        print(test_process.stderr, file=sys.stderr)
        return 1

    grid_process = _run(
        [
            sys.executable,
            "-m",
            "matrender",
            "render",
            "scenes/material_grid.json",
            "-o",
            str(VERIFICATION / "material_grid"),
            "--samples",
            "8",
            "--threads",
            "4",
            "--tile-size",
            "64",
            "--quiet",
        ]
    )
    if grid_process.returncode != 0:
        print(grid_process.stderr, file=sys.stderr)
        return 1

    for state in ("on", "off"):
        process = _run(
            [
                sys.executable,
                "-m",
                "matrender",
                "render",
                "scenes/ssao_contrast.json",
                "-o",
                str(VERIFICATION / f"ssao_{state}"),
                "--samples",
                "2",
                "--threads",
                "4",
                "--ssao",
                state,
                "--quiet",
            ]
        )
        if process.returncode != 0:
            print(process.stderr, file=sys.stderr)
            return 1

    resource_process = _run(
        [
            sys.executable,
            "-m",
            "matrender",
            "render",
            "scenes/material_grid.json",
            "-o",
            str(VERIFICATION / "resource_fallback"),
            "--width",
            "96",
            "--height",
            "72",
            "--samples",
            "8",
            "--threads",
            "2",
            "--tile-size",
            "16",
            "--max-memory-mb",
            "0.5",
            "--quiet",
        ]
    )
    if resource_process.returncode != 0:
        print(resource_process.stderr, file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as directory:
        invalid_output = Path(directory) / "invalid_output"
        invalid_process = _run(
            [
                sys.executable,
                "-m",
                "matrender",
                "render",
                "scenes/invalid_roughness.json",
                "-o",
                str(invalid_output),
            ]
        )
        invalid_no_output = not invalid_output.exists()

    resource_log = json.loads(
        (VERIFICATION / "resource_fallback" / "render_log.json").read_text(
            encoding="utf-8"
        )
    )
    results = {
        "environment": {
            "python": platform.python_version(),
            "python_bits": platform.architecture()[0],
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "logical_cores": __import__("os").cpu_count(),
        },
        "unit_tests": {
            "return_code": test_process.returncode,
            "output": (
                test_process.stderr.strip().splitlines()[-3:]
                if test_process.stderr.strip()
                else []
            ),
        },
        "color": _color_metrics(),
        "threads": _thread_metrics(),
        "repeatability": _repeatability_metrics(),
        "ssao": _ssao_metrics(),
        "reports": _report_metrics(),
        "invalid_scene": {
            "return_code": invalid_process.returncode,
            "stderr_contains_E_INVALID_SCENE": (
                "E_INVALID_SCENE" in invalid_process.stderr
            ),
            "output_directory_absent": invalid_no_output,
        },
        "resource_fallback": {
            "return_code": resource_process.returncode,
            "stderr_contains_E_RESOURCE_LIMIT": (
                "E_RESOURCE_LIMIT" in resource_process.stderr
            ),
            "degraded": resource_log["resource"]["degraded"],
            "requested_resolution": resource_log["render"][
                "requested_resolution"
            ],
            "effective_resolution": resource_log["render"][
                "effective_resolution"
            ],
            "requested_samples": resource_log["render"]["requested_samples"],
            "effective_samples": resource_log["render"]["effective_samples"],
            "initial_estimated_bytes": resource_log["resource"][
                "initial_estimated_bytes"
            ],
            "effective_estimated_bytes": resource_log["resource"][
                "effective_estimated_bytes"
            ],
        },
    }
    results_path = VERIFICATION / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

