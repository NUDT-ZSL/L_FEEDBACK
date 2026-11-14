from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _process(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, check=False, text=True, capture_output=True
    )


def _normalized_report(report: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(report))
    for frame in normalized["frames"]:
        frame["render_time_ms"] = -1.0
    return normalized


def _png_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("frame_*.png"))
    }


def _clear_render_artifacts(directory: Path) -> None:
    if not directory.exists():
        return
    for pattern in ("frame_*.png", "report.json", "report.partial.json"):
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full local acceptance flow")
    parser.add_argument(
        "--root",
        default="validation/acceptance",
        help="directory for generated inputs and outputs",
    )
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=135)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / args.root).resolve()
    input_dir = root / "input"
    first_dir = root / "render_a"
    second_dir = root / "render_b"
    failure_input = root / "failure_input"
    failure_output = root / "failure_output"
    _clear_render_artifacts(first_dir)
    _clear_render_artifacts(second_dir)
    _clear_render_artifacts(failure_output)

    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "generate",
            "--output",
            str(input_dir),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--frames",
            str(args.frames),
            "--fps",
            str(args.fps),
            "--seed",
            str(args.seed),
        ],
        repo_root,
    )
    config_path = input_dir / "config.json"
    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "render",
            "--config",
            str(config_path),
            "--input",
            str(input_dir),
            "--output",
            str(first_dir),
            "--quiet",
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "render",
            "--config",
            str(config_path),
            "--input",
            str(input_dir),
            "--output",
            str(second_dir),
            "--quiet",
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "verify-slices",
            "--config",
            str(config_path),
            "--input",
            str(input_dir),
            "--json-output",
            str(root / "slice_verification.json"),
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "export-kernel",
            "--blades",
            "0",
            "--radius",
            "10",
            "--output",
            str(root / "kernel_circle.npy"),
        ],
        repo_root,
    )
    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "export-kernel",
            "--blades",
            "6",
            "--radius",
            "10",
            "--output",
            str(root / "kernel_hex.npy"),
        ],
        repo_root,
    )

    first_hashes = _png_hashes(first_dir)
    second_hashes = _png_hashes(second_dir)
    first_report = json.loads((first_dir / "report.json").read_text(encoding="utf-8"))
    second_report = json.loads((second_dir / "report.json").read_text(encoding="utf-8"))
    slice_verification = json.loads(
        (root / "slice_verification.json").read_text(encoding="utf-8")
    )

    _run(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "generate",
            "--output",
            str(failure_input),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--frames",
            "6",
            "--fps",
            str(args.fps),
            "--seed",
            str(args.seed),
        ],
        repo_root,
    )
    bad_track = {
        "time_unit": "seconds",
        "keyframes": [
            {"time": 0.0, "focus_distance": 2.0},
            {"time": 0.25, "focus_distance": 8.0},
            {"time": 0.125, "focus_distance": 4.0},
        ],
    }
    (failure_input / "focus_track.json").write_text(
        json.dumps(bad_track, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    failure_process = _process(
        [
            sys.executable,
            "-m",
            "dofpipe",
            "render",
            "--config",
            str(failure_input / "config.json"),
            "--input",
            str(failure_input),
            "--output",
            str(failure_output),
            "--quiet",
        ],
        repo_root,
    )
    partial_path = failure_output / "report.partial.json"
    conventional_path = failure_output / "report.json"
    partial_report = (
        json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.is_file()
        else None
    )
    conventional_report = (
        json.loads(conventional_path.read_text(encoding="utf-8"))
        if conventional_path.is_file()
        else None
    )

    kernel_circle = np.load(root / "kernel_circle.npy", allow_pickle=False)
    kernel_hex = np.load(root / "kernel_hex.npy", allow_pickle=False)
    failure_frame_count = len(
        [path for path in failure_output.glob("frame_*.png") if path.is_file()]
    )
    checks: dict[str, bool] = {
        "png_sequence_count": len(first_hashes) == args.frames,
        "png_hashes_byte_identical": first_hashes == second_hashes,
        "report_equal_excluding_timing": _normalized_report(first_report)
        == _normalized_report(second_report),
        "slice_consistency_passed": slice_verification["status"] == "passed",
        "circle_kernel_normalized": bool(
            np.isclose(float(np.sum(kernel_circle)), 1.0, atol=1e-12)
        ),
        "polygon_kernel_normalized": bool(
            np.isclose(float(np.sum(kernel_hex)), 1.0, atol=1e-12)
        ),
        "failure_exit_nonzero": failure_process.returncode != 0,
        "partial_report_preserved": partial_report is not None,
        "partial_and_conventional_match": partial_report == conventional_report,
        "rendered_frames_preserved_on_failure": (
            partial_report is not None
            and partial_report["total_frames"] == failure_frame_count
            and failure_frame_count > 0
        ),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "sample": {
            "width": args.width,
            "height": args.height,
            "frame_count": args.frames,
            "fps": args.fps,
            "seed": args.seed,
        },
        "slice_consistency": {
            "max_abs_error_8bit": slice_verification["max_abs_error_8bit"],
            "threshold_max_abs_error_8bit": slice_verification[
                "threshold_max_abs_error_8bit"
            ],
            "mean_abs_error_8bit": slice_verification["mean_abs_error_8bit"],
        },
        "kernels": {
            "circle_sum": float(np.sum(kernel_circle)),
            "polygon_sum": float(np.sum(kernel_hex)),
            "circle_shape": list(kernel_circle.shape),
            "polygon_shape": list(kernel_hex.shape),
        },
        "failure_case": {
            "exit_code": int(failure_process.returncode),
            "status": None if partial_report is None else partial_report["status"],
            "error_frame_index": (
                None
                if partial_report is None
                else partial_report["error"]["frame_index"]
            ),
            "partial_frames": failure_frame_count,
            "partial_total_frames": (
                None if partial_report is None else partial_report["total_frames"]
            ),
        },
        "artifacts_root": str(root),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
