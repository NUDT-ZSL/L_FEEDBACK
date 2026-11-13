from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from ruin_bake.layout import LayoutParams, build_probe_layout
from ruin_bake.scene import (
    Scene,
    SceneGeneratorConfig,
    apply_scene_patch,
    generate_scene,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_TMP = ROOT / ".tmp" / "tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)

SMALL_GENERATOR = SceneGeneratorConfig(
    seed=1234,
    bounds_min=(-5.0, -4.0, 0.0),
    bounds_max=(5.0, 4.0, 4.0),
    extra_boxes=8,
    point_light_count=2,
    emissive_box_count=1,
)
SMALL_LAYOUT = LayoutParams(spacing=2.0, cull_distance=0.4)


def make_scene(patch: dict[str, Any] | None = None) -> Scene:
    scene = generate_scene(SMALL_GENERATOR)
    return apply_scene_patch(scene, patch)


def make_layout(scene: Scene, params: LayoutParams = SMALL_LAYOUT):
    return build_probe_layout(scene, params)


def config_dict(
    *,
    generator: SceneGeneratorConfig = SMALL_GENERATOR,
    layout: LayoutParams = SMALL_LAYOUT,
    samples: int = 16,
    workers: int = 1,
    chunk_size: int = 7,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "config_version": 1,
        "scene": generator.to_dict(),
        "layout": layout.to_dict(),
        "bake": {
            "samples": samples,
            "ray_epsilon": 1.0e-6,
            "workers": workers,
            "chunk_size": chunk_size,
        },
        "patch": patch or {},
    }


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def cli_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing else str(SRC) + os.pathsep + existing
    env["TMPDIR"] = str(TEST_TMP)
    env["PYTHONHASHSEED"] = "0"
    env.update(extra)
    return env


def run_cli(
    *arguments: str | Path,
    timeout: float = 60.0,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "ruin_bake", *(str(argument) for argument in arguments)]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=cli_env(**(env_extra or {})),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def float_bits(values: tuple[tuple[float, float, float], ...]) -> bytes:
    return b"".join(struct.pack("<3d", *value) for value in values)

