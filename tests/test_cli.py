from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import GL_AVAILABLE


pytestmark = pytest.mark.gl


@pytest.mark.skipif(not GL_AVAILABLE, reason="OpenGL 3.3 context is unavailable")
def test_cli_without_input_uses_builtins(tmp_path: Path) -> None:
    output = tmp_path / "cli-output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "material_tool",
            "--size",
            "32",
            "--channels",
            "roughness,normal",
            "--no-reload-check",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["input_dir"] == "builtin"
    assert summary["summary"]["rendered"] == 2
    assert summary["summary"]["invalid"] == 3
    assert summary["resource_counts_after_close"] == {}
    assert "GRAPH_CYCLE" in result.stderr
    for filename in ("roughness.png", "normal.png"):
        assert (output / "demo_material" / filename).exists()
        assert not (output / "demo_material" / "base_color.png").exists()


def test_cli_returns_zero_for_invalid_material_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "material_tool",
            str(tmp_path / "missing"),
            "--size",
            "16",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["errors"][0]["error_code"] == "INPUT_DIRECTORY_INVALID"

