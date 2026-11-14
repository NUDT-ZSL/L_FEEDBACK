"""Deterministic local acceptance run for the terrain baker.

Runs a small (257) and a tiled (>1025) bake, verifies every requirement and
writes a human-readable report to ``artifacts/acceptance_report.txt``.
Software rendering is selected through the standard Mesa environment
variables; on Windows the OSMesa/llvmpipe DLLs are used automatically.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from terrain_baker.heightfield import open_heightfield
from terrain_baker.memory import MemoryBudget
from terrain_baker.normal import bake_normal_map
from terrain_baker.verifier import verify_output

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(ROOT, "artifacts")
REPORT = os.path.join(ARTIFACTS, "acceptance_report.txt")


def run_cli(output: str, resolution: int, extra: list[str]) -> int:
    cmd = [
        sys.executable, "-m", "terrain_baker",
        "--seed", "2026",
        "--resolution", str(resolution),
        "--height-scale", "8",
        "--frequency", "2",
        "--levels", "0", "1", "2",
        "--tile", "512",
        "--output", output,
    ] + extra
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    os.makedirs(ARTIFACTS, exist_ok=True)
    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    small = os.path.join(ARTIFACTS, "accept_257")
    tiled = os.path.join(ARTIFACTS, "accept_1025")
    code_small = run_cli(small, 257, ["--preview-width", "256", "--preview-height", "256"])
    code_tiled = run_cli(tiled, 1025, ["--preview-width", "256", "--preview-height", "256"])
    log(f"CLI exit codes -> 257: {code_small}, 1025: {code_tiled}")

    for label, path in (("257", small), ("1025", tiled)):
        manifest = json.load(open(os.path.join(path, "metadata.json")))
        verification = verify_output(path)
        log(f"\n[{label}] verification ok={verification['ok']} "
            f"files={verification['files_checked']} lod={verification['lod_checked']}")
        log(f"  memory peak={manifest['memory']['peak_live_bytes'] / 1048576:.2f} MiB "
            f"limit={manifest['memory']['limit_bytes'] / 1048576:.0f} MiB")
        log(f"  normalized normal length error="
            f"{manifest['normal_map']['quality']['max_normalized_length_error']:.2e}")
        for report in manifest["consistency_checks"]:
            log(f"  LOD{report['level']} max angle={report['max_angle_deg']:.4f} deg "
                f"samples={report['sample_count']} passed={report['passed']}")

    # Tiled vs monolithic numerical equivalence and seam check on the 1025 grid.
    grid = open_heightfield(os.path.join(tiled, "data", "heightfield.dat"), 1025)
    mono, _ = bake_normal_map(grid, MemoryBudget(64 * 1024 * 1024), tile=1025)
    pieces, _ = bake_normal_map(grid, MemoryBudget(64 * 1024 * 1024), tile=512)
    exact = bool((mono == pieces).all())
    border = abs(pieces[512].astype(int) - pieces[511].astype(int)).max()
    interior = abs(pieces[100].astype(int) - pieces[99].astype(int)).max()
    log(f"\n[tiling] exact tile==monolithic={exact} "
        f"border discontinuity={border} interior={interior}")

    success = code_small == 0 and code_tiled == 0 and exact and border <= interior
    log(f"\nOVERALL: {'PASS' if success else 'FAIL'}")
    with open(REPORT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
