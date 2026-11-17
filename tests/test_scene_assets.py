from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deferred_renderer.scene import (
    build_assets,
    default_scene_dict,
    parse_scene,
    write_default_scene,
)


def test_default_scene_serialization_is_byte_stable(tmp_path: Path) -> None:
    first = write_default_scene(tmp_path / "first.json")
    second = write_default_scene(tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    assert parse_scene(json.loads(first.read_text())).scene_sha256 == parse_scene(
        default_scene_dict()
    ).scene_sha256


def test_geometry_and_texture_hashes_are_deterministic() -> None:
    scene = parse_scene(default_scene_dict())
    first = build_assets(scene)
    second = build_assets(scene)
    assert first.geometry_sha256 == second.geometry_sha256
    assert first.texture_sha256 == second.texture_sha256
    assert first.combined_sha256 == second.combined_sha256
    assert len(first.meshes) == len(scene.objects)
    for left, right in zip(first.meshes, second.meshes, strict=True):
        assert np.array_equal(left.vertices, right.vertices)
        assert np.array_equal(left.indices, right.indices)


def test_meshes_contain_tangent_space_and_non_degenerate_triangles() -> None:
    scene = parse_scene(default_scene_dict())
    assets = build_assets(scene)
    for mesh in assets.meshes:
        assert mesh.vertices.shape[1] == 12
        triangles = mesh.indices.reshape(-1, 3)
        positions = mesh.vertices[:, :3]
        for triangle in triangles:
            p0, p1, p2 = positions[triangle]
            assert np.linalg.norm(np.cross(p1 - p0, p2 - p0)) > 1e-7
        tangents = mesh.vertices[:, 6:9]
        assert np.all(np.linalg.norm(tangents, axis=1) > 0.9)

