"""Pixel-level parity between the ModernGL and NumPy backends."""

import numpy as np
import pytest

from rfr.renderer import Renderer
from rfr.backends.software import SoftwareBackend
from rfr.compare import compare_framebuffers


def test_full_scene_backends_match(scene, draw_list, gl_backend_or_skip):
    gl = gl_backend_or_skip
    if (gl.width, gl.height) != (scene.width, scene.height):
        from rfr.backends.gl_backend import GLBackend
        gl = GLBackend(scene.width, scene.height)
    Renderer(gl, scene).render_frame(draw_list)
    gl_pixels = gl.read_pixels()

    software = SoftwareBackend(scene.width, scene.height)
    Renderer(software, scene).render_frame(draw_list)
    sw_pixels = software.read_pixels()

    report = compare_framebuffers(gl_pixels, sw_pixels, tolerance=2)
    assert report["mismatched_pixels"] <= 2, report
    assert report["mismatched_pixels"] / report["total_pixels"] < 0.001


def test_animated_frames_stay_consistent(scene, gl_backend_or_skip):
    from rfr.backends.gl_backend import GLBackend
    from rfr.renderer import build_draw_list
    for frame in (0, 3, 8):
        scene.update(frame)
        draws = build_draw_list(scene)
        gl = GLBackend(scene.width, scene.height)
        Renderer(gl, scene).render_frame(draws)
        gl_pixels = gl.read_pixels()
        gl.release()
        software = SoftwareBackend(scene.width, scene.height)
        Renderer(software, scene).render_frame(draws)
        report = compare_framebuffers(gl_pixels, software.read_pixels(), 2)
        # The acceptance contract allows per-channel differences up to 2; only
        # isolated sub-pixel silhouette/coverage decisions may differ beyond
        # that, and they must remain extremely rare (well under 0.1% pixels).
        assert report["mismatched_pixels"] <= 2, (frame, report)
        assert report["mismatched_pixels"] / report["total_pixels"] < 0.001


def test_compare_reports_coordinate_range_on_mismatch():
    a = np.zeros((4, 4, 4), np.uint8)
    b = a.copy()
    b[2, 3, 0] = 200
    report = compare_framebuffers(a, b, tolerance=2)
    assert report["mismatched_pixels"] == 1
    assert report["coordinate_range"] == {
        "min_row": 2, "max_row": 2, "min_col": 3, "max_col": 3}


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        compare_framebuffers(np.zeros((2, 2, 4), np.uint8),
                             np.zeros((3, 3, 4), np.uint8))
