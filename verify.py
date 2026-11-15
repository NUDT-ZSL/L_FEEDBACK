from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import PIL

from matgraph.outputs import write_atomic_file
from matgraph.samples import generate_samples, sample_graph

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
VERIFICATION = ROOT / "verification"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hashes(directory: Path) -> dict:
    return {
        path.name: sha256(path.read_bytes())
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def run_command(arguments) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    generate_samples(SAMPLES)
    run_a = VERIFICATION / "run_a"
    run_b = VERIFICATION / "run_b"
    render_command = [
        str(ROOT / "run.py"),
        "render",
        "--graph",
        str(SAMPLES / "material_graph.json"),
        "--scene",
        str(SAMPLES / "scene.json"),
    ]
    first = run_command([*render_command, "--out", str(run_a)])
    second = run_command([*render_command, "--out", str(run_b)])
    if first.returncode != 0 or second.returncode != 0:
        sys.stderr.write(first.stderr + second.stderr)
        return 1

    first_hashes = hashes(run_a)
    second_hashes = hashes(run_b)
    deterministic = first_hashes == second_hashes
    report = json.loads((run_a / "report.json").read_text(encoding="utf-8"))

    invalid_graph = sample_graph()
    del invalid_graph["nodes"][1]["inputs"]
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        invalid_path = temporary_path / "invalid_graph.json"
        invalid_path.write_text(
            json.dumps(invalid_graph, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        invalid_output = temporary_path / "must_not_exist"
        invalid_run = run_command(
            [
                str(ROOT / "run.py"),
                "render",
                "--graph",
                str(invalid_path),
                "--scene",
                str(SAMPLES / "scene.json"),
                "--out",
                str(invalid_output),
            ]
        )
        invalid_error = json.loads(invalid_run.stderr)
        no_partial_output = not invalid_output.exists()

    tests = run_command(["-m", "unittest", "discover", "-s", "tests", "-v"])
    test_summary = [
        line for line in tests.stderr.splitlines() if line.strip()
    ][-1] if tests.stderr.strip() else ""
    compact = {
        "schema": "verification/v1",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "samples": {
            "graph_sha256": sha256((SAMPLES / "material_graph.json").read_bytes()),
            "scene_sha256": sha256((SAMPLES / "scene.json").read_bytes()),
            "checker_sha256": sha256((SAMPLES / "textures/checker.png").read_bytes()),
            "bumps_sha256": sha256((SAMPLES / "textures/bumps.png").read_bytes()),
        },
        "render": {
            "resolution": report["resolution"],
            "counts": report["counts"],
            "ir_sha256": report["compiler"]["ir_sha256"],
            "file_sha256": first_hashes,
        },
        "determinism": {
            "two_independent_cli_runs_byte_identical": deterministic,
        },
        "invalid_graph": {
            "exit_code": invalid_run.returncode,
            "error_code": invalid_error["error"]["code"],
            "node_index": invalid_error["error"]["node_index"],
            "node_id": invalid_error["error"]["node_id"],
            "no_partial_output_directory": no_partial_output,
        },
        "tests": {
            "exit_code": tests.returncode,
            "summary": test_summary,
        },
    }
    verification_path = VERIFICATION / "verification.json"
    write_atomic_file(
        verification_path,
        (
            json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8"),
    )
    print(json.dumps(compact, indent=2, sort_keys=True))
    if not deterministic or tests.returncode != 0 or invalid_run.returncode != 2 or not no_partial_output:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
