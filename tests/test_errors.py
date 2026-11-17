from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "deferred_renderer", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )


def base_scene(tmp_path: Path) -> dict:
    scene_path = tmp_path / "base.json"
    result = run_cli("--write-default-scene", str(scene_path))
    assert result.returncode == 0, result.stderr
    return json.loads(scene_path.read_text())


@pytest.mark.parametrize(
    ("name", "mutate", "code"),
    [
        (
            "missing_attachment",
            lambda data: data["render"]["g_buffer"]["attachments"].remove(
                "material_id"
            ),
            "E_MISSING_ATTACHMENT",
        ),
        (
            "bad_light",
            lambda data: data["lights"][0].__setitem__("radius", 0.0),
            "E_BAD_LIGHT",
        ),
        (
            "zero_area",
            lambda data: data["objects"][-1]["size"].__setitem__(1, 0.0),
            "E_ZERO_AREA_GEOMETRY",
        ),
    ],
)
def test_invalid_scene_fails_with_stable_error_code(
    tmp_path: Path,
    name: str,
    mutate,
    code: str,
) -> None:
    data = copy.deepcopy(base_scene(tmp_path))
    mutate(data)
    scene_path = tmp_path / f"{name}.json"
    scene_path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / name
    result = run_cli(
        "--scene",
        str(scene_path),
        "--out",
        str(output),
        "--mode",
        "headless",
    )
    assert result.returncode != 0
    assert result.stderr
    stats = json.loads((output / "stats.json").read_text())
    assert stats["ok"] is False
    assert stats["error"]["code"] == code


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses X11 display failure")
def test_window_creation_failure_has_nonzero_exit(tmp_path: Path) -> None:
    scene = tmp_path / "scene.json"
    result = run_cli("--write-default-scene", str(scene))
    assert result.returncode == 0
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    output = tmp_path / "window_failure"
    result = run_cli(
        "--scene",
        str(scene),
        "--out",
        str(output),
        "--mode",
        "window",
        "--window-hidden",
        env=env,
    )
    assert result.returncode != 0
    stats = json.loads((output / "stats.json").read_text())
    assert stats["error"]["code"] == "E_WINDOW_CREATE"

