from __future__ import annotations

import pytest

from material_tool.errors import GLContextError
from material_tool.renderer import Renderer


def _can_create_gl() -> bool:
    try:
        renderer = Renderer(size=16)
    except Exception:
        return False
    renderer.close()
    return True


GL_AVAILABLE = _can_create_gl()


@pytest.fixture
def renderer():
    if not GL_AVAILABLE:
        pytest.skip("OpenGL 3.3 context is not available in this environment")
    instance = Renderer(size=64)
    try:
        yield instance
    finally:
        instance.close()


def pytest_configure(config):
    config.addinivalue_line("markers", "gl: requires an OpenGL 3.3 context")

