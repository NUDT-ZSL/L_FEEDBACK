from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from material_validator.samples import generate_sample


def test_cli_nonzero_exit_and_reports(tmp_path: Path) -> None:
    manifest = generate_sample(tmp_path / "sample", "invalid")
    output = tmp_path / "out"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "material_validator",
            "--manifest",
            str(manifest),
            "--out",
            str(output),
            "--parallel",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert process.returncode == 1
    assert (output / "report.json").is_file()
    assert (output / "summary.txt").is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["summary"]["errors"] > 0
