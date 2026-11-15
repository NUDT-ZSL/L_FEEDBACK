"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rfr.sample_scene import build_sample_scene
from rfr.scene import Scene
from rfr.renderer import Renderer, build_draw_list
from rfr.backends.software import SoftwareBackend


@pytest.fixture
def scene():
    return Scene(build_sample_scene(width=160, height=120))


@pytest.fixture
def draw_list(scene):
    return build_draw_list(scene)


@pytest.fixture
def software_frame(scene, draw_list):
    backend = SoftwareBackend(scene.width, scene.height)
    renderer = Renderer(backend, scene)
    renderer.render_frame(draw_list)
    return backend.read_pixels()


@pytest.fixture
def gl_backend_or_skip():
    try:
        from rfr.backends.gl_backend import GLBackend
        ctx = GLBackend(16, 16)
    except Exception as exc:  # pragma: no cover - environment without GL
        pytest.skip(f"OpenGL backend unavailable: {exc}")
    return ctx


def assert_rgba8(image):
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 4
