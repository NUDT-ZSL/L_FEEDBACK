#!/usr/bin/env python3
"""End-to-end acceptance checks for generated assets and CLI contracts."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=60,
        env=env,
    )


def generate(generator: Path, preset: str, output: Path) -> None:
    result = run([str(generator), "--preset", preset, "--out", str(output)])
    check(
        result.returncode == 0,
        f"generator creates {preset} asset (rc={result.returncode})",
    )
    check(output.is_file(), f"{preset} asset exists")


def deterministic_report(document: dict) -> dict:
    clone = json.loads(json.dumps(document))
    clone.pop("timing", None)
    for frame in clone.get("frames", []):
        frame.pop("elapsed_ms", None)
    return clone


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: acceptance.py <asset_generator> <skelpreview> <workdir>")
        return 2

    generator = Path(sys.argv[1]).resolve()
    tool = Path(sys.argv[2]).resolve()
    workdir = Path(sys.argv[3]).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    normal = workdir / "normal.ska"
    generate(generator, "normal", normal)

    first = workdir / "run_a"
    second = workdir / "run_b"
    common = [
        str(tool),
        "offline",
        str(normal),
        "--frame",
        "3",
        "--width",
        "176",
        "--height",
        "144",
    ]
    result_a = run(common + ["--output-dir", str(first)])
    check(
        result_a.returncode == 0,
        f"offline frame render succeeds (rc={result_a.returncode})",
    )
    result_b = run(common + ["--output-dir", str(second)])
    check(
        result_b.returncode == 0,
        f"second offline render succeeds (rc={result_b.returncode})",
    )

    for name in ("depth_0003.pfm", "color_0003.ppm"):
        check((first / name).is_file(), f"zero-padded output {name} exists")
        check(
            (first / name).read_bytes() == (second / name).read_bytes(),
            f"{name} is byte-identical across runs",
        )

    ppm = (first / "color_0003.ppm").read_bytes()
    check(ppm.startswith(b"P6\n176 144\n255\n"), "PPM header and size are correct")
    check(
        len(ppm) == len(b"P6\n176 144\n255\n") + 176 * 144 * 3,
        "PPM payload length is correct",
    )
    ppm_payload = ppm[len(b"P6\n176 144\n255\n") :]
    background = bytes((14, 17, 22))
    check(
        any(ppm_payload[i : i + 3] != background for i in range(0, len(ppm_payload), 3)),
        "PPM contains rendered geometry",
    )
    pfm = (first / "depth_0003.pfm").read_bytes()
    check(pfm.startswith(b"PF\n176 144\n-1.0\n"), "PFM header and size are correct")
    check(
        len(pfm) == len(b"PF\n176 144\n-1.0\n") + 176 * 144 * 4,
        "PFM payload length is correct",
    )
    pfm_payload = pfm[len(b"PF\n176 144\n-1.0\n") :]
    depths = struct.unpack(f"<{176 * 144}f", pfm_payload)
    check(
        any(0.0 <= value < 1.0 for value in depths),
        "PFM contains covered depth samples",
    )

    report_a = json.loads((first / "report.json").read_text())
    report_b = json.loads((second / "report.json").read_text())
    check(
        deterministic_report(report_a) == deterministic_report(report_b),
        "JSON deterministic fields match; timing is isolated",
    )
    check(report_a["schema"] == "skelpreview.statistics.v1", "JSON schema identifier")
    check(report_a["render"]["total_frames"] == 1, "JSON total frame count")
    check(report_a["render"]["vertex_count"] == 16, "JSON vertex count")
    check(report_a["render"]["triangle_count"] == 24, "JSON triangle count")
    check(
        report_a["render"]["normalized_vertex_count"] == 1,
        "JSON normalized vertex count",
    )
    check(report_a["render"]["nan_positions"] == 0, "JSON position NaN count")
    check(report_a["render"]["nan_normals"] == 0, "JSON normal NaN count")
    check("timing" in report_a, "JSON timing field is present")
    check(
        "elapsed_ms" in report_a["frames"][0],
        "JSON per-frame timing field is present",
    )

    ranged = workdir / "range"
    range_result = run(
        [
            str(tool),
            "offline",
            str(normal),
            "--start-frame",
            "10",
            "--end-frame",
            "11",
            "--width",
            "64",
            "--height",
            "48",
            "--output-dir",
            str(ranged),
        ]
    )
    check(range_result.returncode == 0, "multi-frame range render succeeds")
    ranged_report = json.loads((ranged / "report.json").read_text())
    check(ranged_report["render"]["total_frames"] == 2, "range report frame count")
    check((ranged / "depth_0010.pfm").is_file(), "range pads frame 10")
    check((ranged / "color_0011.ppm").is_file(), "range pads frame 11")

    expected_codes = {
        "cycle": 18,
        "oob-index": 27,
        "oob-bone": 26,
        "nonmonotonic": 20,
        "zero-quat": 22,
        "bad-weight-count": 23,
        "bad-weights": 25,
        "bad-order": 28,
        "truncated": 14,
    }
    for preset, expected in expected_codes.items():
        asset = workdir / f"{preset}.ska"
        generate(generator, preset, asset)
        result = run(
            [
                str(tool),
                "offline",
                str(asset),
                "--frame",
                "0",
                "--width",
                "32",
                "--height",
                "32",
                "--output-dir",
                str(workdir / f"{preset}_out"),
            ]
        )
        check(
            result.returncode == expected,
            f"{preset} exits with stable code {expected}",
        )

    singular = workdir / "singular.ska"
    generate(generator, "singular", singular)
    singular_out = workdir / "singular_out"
    singular_result = run(
        [
            str(tool),
            "offline",
            str(singular),
            "--frame",
            "0",
            "--width",
            "64",
            "--height",
            "48",
            "--output-dir",
            str(singular_out),
        ]
    )
    check(
        singular_result.returncode == 70,
        "normal NaNs return distinct offline failure code 70",
    )
    singular_report = json.loads((singular_out / "report.json").read_text())
    check(
        singular_report["render"]["nan_normals"] > 0,
        "singular report counts normal NaNs",
    )
    check(
        singular_report["render"]["nan_positions"] == 0,
        "singular report keeps position NaN count separate",
    )

    force_out = workdir / "force_offline"
    force_result = run(
        [
            str(tool),
            "preview",
            str(normal),
            "--force-offline",
            "--frame",
            "2",
            "--width",
            "64",
            "--height",
            "48",
            "--output-dir",
            str(force_out),
        ]
    )
    check(force_result.returncode == 0, "preview force-offline fallback succeeds")
    check(
        "falling back to deterministic offline" in force_result.stderr,
        "preview fallback prints diagnostic",
    )
    check((force_out / "depth_0002.pfm").is_file(), "fallback emits PFM")
    check((force_out / "color_0002.ppm").is_file(), "fallback emits PPM")
    check((force_out / "report.json").is_file(), "fallback emits report")

    if sys.platform.startswith("linux"):
        headless_out = workdir / "headless_fallback"
        env = os.environ.copy()
        env["DISPLAY"] = ""
        env["WAYLAND_DISPLAY"] = ""
        headless = run(
            [
                str(tool),
                "preview",
                str(normal),
                "--frame",
                "1",
                "--width",
                "64",
                "--height",
                "48",
                "--output-dir",
                str(headless_out),
            ],
            env=env,
        )
        check(
            headless.returncode == 0,
            "headless window-creation failure falls back offline",
        )
        check(
            "falling back" in headless.stderr,
            "headless fallback diagnostic is emitted",
        )

    print("\nacceptance: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:  # noqa: BLE001 - CLI test must report clearly.
        print(f"[FAIL] {exception}", file=sys.stderr)
        raise
