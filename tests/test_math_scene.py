import math

import numpy as np
import pytest

from rfr import mathutils
from rfr.errors import RenderError
from rfr.scene import Scene
from rfr.sample_scene import build_sample_scene


def test_compose_is_float64_and_identity_order():
    m = mathutils.compose([1, 2, 3], [0, 0, 0], [2, 2, 2])
    assert m.dtype == np.float64
    assert np.allclose(m[:3, 3], [1, 2, 3])
    assert np.allclose(np.diag(m)[:3], [2, 2, 2])


def test_world_matrix_parent_child_accumulation():
    scene = Scene(build_sample_scene())
    parent = next(n for n in scene.nodes if n.name == "parent_cube")
    child = next(n for n in scene.nodes if n.name == "child_sphere")
    expected = parent.world @ child.local
    assert child.world.dtype == np.float64
    assert np.allclose(child.world, expected)
    # float32 cast is available for rendering and stays finite
    rendered = child.world.astype(np.float32)
    assert rendered.dtype == np.float32 and np.isfinite(rendered).all()


def test_animation_is_deterministic():
    a = Scene(build_sample_scene())
    b = Scene(build_sample_scene())
    a.update(7)
    b.update(7)
    node_a = next(n for n in a.nodes if n.name == "child_sphere")
    node_b = next(n for n in b.nodes if n.name == "child_sphere")
    assert np.allclose(node_a.world, node_b.world)


def test_perspective_matches_known_opengl_form():
    proj = mathutils.perspective(math.radians(90.0), 2.0, 1.0, 10.0)
    assert proj.dtype == np.float64
    assert proj[3, 2] == -1.0 and proj[3, 3] == 0.0
    assert proj[0, 0] == pytest.approx(0.5)


def test_invalid_perspective_raises_business_error():
    with pytest.raises(RenderError) as info:
        mathutils.perspective(1.0, 1.0, 5.0, 1.0)
    assert info.value.code == "E_DIMENSION"


def test_missing_scene_file_is_structured_error(tmp_path):
    with pytest.raises(RenderError) as info:
        Scene.load(str(tmp_path / "absent.json"))
    assert info.value.code == "E_SCENE_INVALID"


def test_undefined_mesh_reference():
    data = build_sample_scene()
    data["nodes"][0]["mesh"] = "does_not_exist"
    with pytest.raises(RenderError) as info:
        Scene(data)
    assert info.value.code == "E_SCENE_MESH"
