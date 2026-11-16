from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def run_cli(arguments: Sequence[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "soft_renderer"] + list(arguments),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=environment,
    )


def temporary_files(directory: Path) -> list:
    return sorted(
        str(path)
        for path in directory.rglob("*")
        if path.is_file() and (".tmp-" in path.name or path.name.endswith(".backup"))
    )
