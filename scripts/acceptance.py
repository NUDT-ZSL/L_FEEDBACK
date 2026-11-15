"""End-to-end acceptance: serial vs parallel, single vs multiple, errors."""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
OUT = os.path.join(ROOT, "output", "acceptance")


def run_cli(*extra):
    proc = subprocess.run(
        [PY, "-m", "volfog", "--output-dir", OUT, *extra],
        capture_output=True, text=True, cwd=ROOT,
    )
    return proc


def load(tag, channel):
    return np.load(os.path.join(OUT, f"fog_{tag}_{channel}.npy"))


def main():
    results = []

    run_cli("--mode", "single", "--output-prefix", "sser")
    run_cli("--mode", "single", "--parallel", "--threads", "4", "--output-prefix", "spar")
    run_cli("--mode", "multiple", "--output-prefix", "mser")
    run_cli("--mode", "multiple", "--parallel", "--threads", "4", "--output-prefix", "mpar")

    for mode in ("single", "multiple"):
        for channel in ("color_linear", "transmittance", "depth"):
            ser_prefix = "sser" if mode == "single" else "mser"
            par_prefix = "spar" if mode == "single" else "mpar"
            serial = np.load(os.path.join(OUT, f"{ser_prefix}_{mode}_{channel}.npy"))
            parallel = np.load(os.path.join(OUT, f"{par_prefix}_{mode}_{channel}.npy"))
            diff_sp = float(np.max(np.abs(serial - parallel)))
            results.append((mode, channel, diff_sp))

    print("== serial vs parallel max abs diff (tolerance 1e-6) ==")
    for mode, channel, diff_sp in results:
        status = "PASS" if diff_sp <= 1e-6 else "FAIL"
        print(f"[{status}] {mode:8s} {channel:14s} diff={diff_sp:.3e}")

    s_color = np.load(os.path.join(OUT, "sser_single_color_linear.npy"))
    m_color = np.load(os.path.join(OUT, "mser_multiple_color_linear.npy"))
    color_delta = float(np.max(np.abs(m_color - s_color)))
    trans_delta = float(np.max(np.abs(
        np.load(os.path.join(OUT, "sser_single_transmittance.npy"))
        - np.load(os.path.join(OUT, "mser_multiple_transmittance.npy")))))
    depth_delta = float(np.max(np.abs(
        np.load(os.path.join(OUT, "sser_single_depth.npy"))
        - np.load(os.path.join(OUT, "mser_multiple_depth.npy")))))
    print("\n== mode comparison ==")
    print(f"color max delta        : {color_delta:.6f} (expected > 1e-6)")
    print(f"transmittance max delta: {trans_delta:.3e} (expected 0)")
    print(f"depth max delta        : {depth_delta:.3e} (expected 0)")

    print("\n== invalid input handling ==")
    for flag, bad, param in [
        ("--focal-length", "-1", "focal_length"),
        ("--focal-length", "nan", "focal_length"),
        ("--step-size", "inf", "step_size"),
        ("--width", "0", "width"),
        ("--voxel-res", "1", "voxel_res"),
        ("--light-color", "1 1 nan", "light_color"),
    ]:
        argv = [flag] + bad.split()
        proc = run_cli(*argv)
        stderr_lines = [line for line in proc.stderr.strip().splitlines() if line]
        ok = (
            proc.returncode == 1
            and len(stderr_lines) == 1
            and json.loads(stderr_lines[0])["error_code"] == "E_INVALID_INPUT"
            and json.loads(stderr_lines[0])["param"] == param
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {flag} {bad}: rc={proc.returncode} stderr={stderr_lines}")


if __name__ == "__main__":
    main()
