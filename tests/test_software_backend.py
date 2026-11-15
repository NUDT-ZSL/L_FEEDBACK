import numpy as np
import pytest

from rfr.renderer import Renderer
from rfr.backends.software import SoftwareBackend
from rfr.errors import RenderError


def test_framebuffer_shape_and_dtype(scene, software_frame):
    assert software_frame.shape == (scene.height, scene.width, 4)
    assert software_frame.dtype == np.uint8


def test_background_is_cleared(scene, draw_list):
    backend = SoftwareBackend(scene.width, scene.height)
    Renderer(backend, scene).render_frame([])
    px = backend.read_pixels()[0, 0]
    expected = np.floor(np.clip(scene.background, 0, 1) * 255 + 0.5)
    assert np.allclose(px, expected, atol=0)


def test_state_machine_is_tracked_in_software(scene, draw_list):
    backend = SoftwareBackend(scene.width, scene.height)
    renderer = Renderer(backend, scene)
    renderer.render_frame(draw_list)
    assert renderer.state.total_switches > 0
    assert renderer.state.switch_counts["blend"] >= 1
    assert backend.backend_statistics()["rasterized_triangles"] > 0


def test_invalid_backend_size():
    with pytest.raises(RenderError) as info:
        SoftwareBackend(-1, 10)
    assert info.value.code == "E_SIZE_MISMATCH"


def test_blend_modes_all_render(scene):
    from rfr.pipeline import DrawCall
    pos = np.array([[-1, -1, -3], [1, -1, -3], [0, 1, -3]], np.float32)
    nrm = np.array([[0, 0, 1]] * 3, np.float32)
    idx = np.array([0, 1, 2], np.int32)
    for mode, color in (("alpha", [0.5, 0.6, 0.7, 0.5]),
                        ("additive", [0.8, 0.4, 0.2, 0.6]),
                        ("opaque", [0.9, 0.9, 0.9, 1.0])):
        col = np.array([color] * 3, np.float32)
        draw = DrawCall(mode, pos, nrm, col, idx, mode, True, "back")
        backend = SoftwareBackend(64, 64)
        scene_small = _minimal_scene()
        Renderer(backend, scene_small).render_frame([draw])
        image = backend.read_pixels()
        assert (image[:, :, :3].sum(axis=2) > 0).any()


def _minimal_scene():
    from rfr.scene import Scene
    return Scene({
        "width": 64, "height": 64, "background": [0, 0, 0, 1],
        "camera": {"eye": [0, 0, 0], "target": [0, 0, -1], "up": [0, 1, 0],
                   "fov_y": 1.047, "near": 0.1, "far": 50},
        "nodes": [{"name": "dummy"}]})
