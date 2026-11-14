from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from skin_bake.errors import BusinessError
from skin_bake.scene import (
    compute_content_hash,
    load_scene,
    parse_scene,
    seal_scene,
)


def _reseal(document):
    document = copy.deepcopy(document)
    document["contentHash"] = compute_content_hash(document)
    return document


def test_scene_hash_is_canonical_and_round_trips(tmp_path, scene_document):
    first = seal_scene(scene_document)
    second = seal_scene(copy.deepcopy(scene_document))
    assert first["contentHash"] == second["contentHash"]
    assert first["contentHash"] == compute_content_hash(first)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    scene = load_scene(path)
    assert scene.vertex_count == 135
    assert scene.bones[0].name == "root"
    assert scene.bake.atlas_width == 128
    assert scene.loop_mode == "loop"


def test_hash_mismatch_has_stable_code(scene_document):
    document = seal_scene(scene_document)
    document["description"] += " tampered"
    with pytest.raises(BusinessError) as error:
        parse_scene(json.dumps(document).encode(), Path("scene.json"))
    assert error.value.code == "E_SCENE_HASH"


def test_unsupported_major_version_has_stable_code(scene_document):
    document = seal_scene(scene_document)
    document["version"] = "2.0.0"
    document = _reseal(document)
    with pytest.raises(BusinessError) as error:
        parse_scene(json.dumps(document).encode(), Path("scene.json"))
    assert error.value.code == "E_VERSION"


def test_invalid_weights_are_rejected(scene_document):
    document = seal_scene(scene_document)
    document["mesh"]["boneWeights"][0] = [0.0, 0.0, 0.0, 0.0]
    document = _reseal(document)
    with pytest.raises(BusinessError) as error:
        parse_scene(json.dumps(document).encode(), Path("scene.json"))
    assert error.value.code == "E_SCHEMA"


def test_fewer_than_four_weights_are_padded(scene_document):
    document = seal_scene(scene_document)
    raw_weights = document["mesh"]["boneWeights"][0][:2]
    total = sum(raw_weights)
    document["mesh"]["boneWeights"][0] = [value / total for value in raw_weights]
    document["mesh"]["boneIndices"][0] = document["mesh"]["boneIndices"][0][:2]
    document = _reseal(document)
    scene = parse_scene(json.dumps(document).encode(), Path("scene.json"))
    assert np.array_equal(scene.bone_weights[0, 2:], np.zeros(2))
    assert abs(float(scene.bone_weights[0].sum()) - 1.0) < 1e-12


def test_cli_parse_error_writes_business_code_to_stderr(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    output = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "skin_bake", "bake", str(bad), "-o", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "E_SCENE_JSON" in result.stderr
    assert not output.exists()


def test_timeline_alignment_is_validated(scene_document):
    document = seal_scene(scene_document)
    document["animation"]["duration"] = 2.01
    document = _reseal(document)
    with pytest.raises(BusinessError) as error:
        parse_scene(json.dumps(document).encode(), Path("scene.json"))
    assert error.value.code == "E_TIMELINE"

