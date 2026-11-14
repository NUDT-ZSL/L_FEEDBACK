from __future__ import annotations

import math

import numpy as np

from skin_bake.basis import build_bind_basis
from skin_bake.math3d import quat_to_matrix
from skin_bake.scene import load_scene
from skin_bake.skinning import AnimationSampler


def test_degenerate_tangent_uses_spherical_fallback_and_basis_is_orthonormal(scene_path):
    scene = load_scene(scene_path)
    basis = build_bind_basis(scene.normals, scene.tangents)
    assert 7 in basis.fallback_vertices
    assert basis.fallback_reasons[7] == "degenerate_tangent"
    dots = np.einsum("vi,vi->v", basis.normals, basis.tangents)
    assert np.max(np.abs(dots)) < 1e-10
    handed = np.einsum(
        "vi,vi->v", np.cross(basis.normals, basis.tangents), basis.bitangents
    )
    signs = np.where(scene.tangents[:, 3] >= 0.0, 1.0, -1.0)
    assert np.allclose(handed, signs, atol=1e-9)


def test_non_orthogonal_tangent_uses_spherical_fallback():
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    tangents = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.5, 1.0]])
    basis = build_bind_basis(normals, tangents)
    assert basis.fallback_vertices == (1,)
    assert basis.fallback_reasons[1] == "non_orthogonal_basis_sphere_fallback"
    assert abs(float(np.dot(basis.normals[1], basis.tangents[1]))) < 1e-12


def test_multibone_linear_blend_skinning_matches_weighted_matrices(scene_path):
    scene = load_scene(scene_path)
    sampler = AnimationSampler(scene)
    time = 0.35
    skin = sampler.skin(time)
    matrices = sampler.skin_matrices(time)
    vertex = 42
    expected = np.zeros(3)
    for weight, bone in zip(scene.bone_weights[vertex], scene.bone_indices[vertex]):
        point = np.append(scene.positions[vertex], 1.0)
        expected += weight * (matrices[bone] @ point)[:3]
    assert np.allclose(skin.positions[vertex], expected, atol=1e-12)
    assert abs(float(scene.bone_weights[vertex].sum()) - 1.0) < 1e-12
    assert np.count_nonzero(scene.bone_weights[vertex] > 0.0) >= 3


def test_loop_closes_bone_tracks_at_duration(scene_path):
    scene = load_scene(scene_path)
    sampler = AnimationSampler(scene)
    start = sampler.world_matrices(0.0)
    end = sampler.world_matrices(scene.duration)
    assert np.allclose(start, end, atol=1e-10)


def test_quaternion_slerp_uses_short_path():
    from skin_bake.math3d import slerp_short

    angle = math.radians(20.0)
    q_positive = np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
    q_negative = -q_positive
    expected = np.array(
        [0.0, 0.0, math.sin(math.radians(10.0) / 2), math.cos(math.radians(10.0) / 2)]
    )
    result = slerp_short(np.array([0.0, 0.0, 0.0, 1.0]), q_negative, 0.5)
    assert np.allclose(result, expected, atol=1e-12)
    assert np.allclose(quat_to_matrix(result), quat_to_matrix(expected), atol=1e-12)

