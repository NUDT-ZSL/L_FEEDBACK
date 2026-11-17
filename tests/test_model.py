from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from character_preview.errors import ModelError
from character_preview.model import load_model, model_from_dict, select_point_lights

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "character.json"


def _raw_model() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sample_model_has_required_structure() -> None:
    model = load_model(SAMPLE)
    assert len(model.nodes) >= 3
    assert len(model.materials) >= 3
    assert len({node.name for node in model.nodes}) == len(model.nodes)
    assert len(model.point_lights) >= 3
    assert model.color_space == "linear"
    assert model.units == "meters"
    for node in model.nodes:
        assert node.material.name in model.materials
        assert node.transform.shape == (4, 4)
        np.testing.assert_allclose(np.linalg.norm(node.normals, axis=1), 1.0, atol=1e-3)


def test_generated_triangle_winding_matches_normals() -> None:
    model = load_model(SAMPLE)
    for node in model.nodes:
        triangles = node.indices.reshape(-1, 3)
        p0, p1, p2 = (node.positions[triangles[:, i]] for i in range(3))
        cross = np.cross(p1 - p0, p2 - p0)
        area = np.linalg.norm(cross, axis=1)
        average_normal = (
            node.normals[triangles[:, 0]]
            + node.normals[triangles[:, 1]]
            + node.normals[triangles[:, 2]]
        )
        valid = area > 1e-10
        alignment = np.einsum("ij,ij->i", cross[valid], average_normal[valid])
        assert np.all(alignment > 0.0), node.name


def test_point_light_selection_is_deterministic_and_limited() -> None:
    model = load_model(SAMPLE)
    first = select_point_lights(model.point_lights, 3, model.camera.target)
    for _ in range(5):
        assert [light.name for light in select_point_lights(model.point_lights, 3, model.camera.target)] == [
            light.name for light in first
        ]
    assert len(first) == 3
    scores = []
    for light in first:
        delta = light.position - model.camera.target
        scores.append(light.intensity / float(np.dot(delta, delta)))
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda raw: raw.update(schema_version=99), "E_MODEL_VERSION"),
        (lambda raw: raw["nodes"][0].update(name=raw["nodes"][1]["name"]), "E_MODEL_NAME"),
        (lambda raw: raw["nodes"][0].update(material="missing"), "E_MODEL_MATERIAL_REF"),
        (lambda raw: raw["nodes"][0].update(transform=[0.0] * 16), "E_MODEL_TRANSFORM"),
        (lambda raw: raw["nodes"][0]["mesh"]["indices"].append(999), "E_MODEL_GEOMETRY"),
        (lambda raw: raw["materials"][0].update(roughness=1.5), "E_MODEL_MATERIAL"),
        (lambda raw: raw["scene"]["point_lights"][0].update(intensity=-1), "E_MODEL_LIGHT"),
        (
            lambda raw: raw["scene"]["camera"].update(up=[0.0, 0.0, 0.0]),
            "E_MODEL_SCHEMA",
        ),
        (lambda raw: raw.update(nodes=raw["nodes"][:2]), "E_MODEL_NODE_COUNT"),
    ],
)
def test_model_validation_error_codes(mutation, code: str) -> None:
    raw = copy.deepcopy(_raw_model())
    mutation(raw)
    with pytest.raises(ModelError) as error:
        model_from_dict(raw)
    assert error.value.code == code


def test_invalid_json_reports_model_json_code(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1,', encoding="utf-8")
    with pytest.raises(ModelError) as error:
        load_model(path)
    assert error.value.code == "E_MODEL_JSON"

