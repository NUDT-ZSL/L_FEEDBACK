from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.generate_samples import build_scene  # noqa: E402
from skin_bake.scene import write_scene  # noqa: E402


@pytest.fixture
def scene_document():
    return build_scene("loop")


@pytest.fixture
def scene_path(tmp_path, scene_document):
    path = tmp_path / "scene_loop.json"
    write_scene(path, scene_document)
    return path


@pytest.fixture
def clamp_scene_path(tmp_path):
    path = tmp_path / "scene_clamp.json"
    write_scene(path, build_scene("clamp"))
    return path


@pytest.fixture
def baked_dir(tmp_path, scene_path):
    from skin_bake.bake import bake_scene

    output = tmp_path / "bake"
    bake_scene(scene_path, output)
    return output

