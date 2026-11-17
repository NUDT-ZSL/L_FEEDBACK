from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from character_preview._gl import GL
from character_preview.config import load_config
from character_preview.errors import ContextError, RenderError
from character_preview.gl_context import OffscreenGLContext
from character_preview.model import load_model
from character_preview.renderer import CharacterRenderer
from character_preview.sequence import render_sequence

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "character.json"
CONFIG = ROOT / "configs" / "sample_render.json"


@pytest.fixture
def gl_context():
    try:
        with OffscreenGLContext():
            yield
    except ContextError as exc:
        pytest.skip(f"OpenGL context unavailable: {exc}")


def _small_config():
    config, _ = load_config(CONFIG)
    return replace(
        config,
        width=128,
        height=96,
        shadow_map_size=256,
        point_shadow_size=64,
        max_point_lights=2,
        frame_count=2,
        time_step=0.25,
    )


def test_renderer_state_restore_and_lifecycle(gl_context) -> None:
    model = load_model(SAMPLE)
    config = replace(_small_config(), frame_count=1)
    GL.glViewport(7, 9, 33, 44)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
    GL.glEnable(GL.GL_BLEND)
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glEnable(GL.GL_STENCIL_TEST)
    GL.glEnable(GL.GL_RASTERIZER_DISCARD)
    GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
    GL.glColorMask(GL.GL_FALSE, GL.GL_TRUE, GL.GL_FALSE, GL.GL_TRUE)
    GL.glDepthRange(0.1, 0.9)
    GL.glClearDepth(0.75)
    before = {
        "viewport": tuple(int(x) for x in GL.glGetIntegerv(GL.GL_VIEWPORT)),
        "depth_test": bool(GL.glIsEnabled(GL.GL_DEPTH_TEST)),
        "cull": bool(GL.glIsEnabled(GL.GL_CULL_FACE)),
        "depth_func": int(GL.glGetIntegerv(GL.GL_DEPTH_FUNC)),
        "pack_alignment": int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT)),
        "blend": bool(GL.glIsEnabled(GL.GL_BLEND)),
        "scissor": bool(GL.glIsEnabled(GL.GL_SCISSOR_TEST)),
        "stencil": bool(GL.glIsEnabled(GL.GL_STENCIL_TEST)),
        "rasterizer_discard": bool(GL.glIsEnabled(GL.GL_RASTERIZER_DISCARD)),
        "framebuffer_srgb": bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB)),
        "color_mask": tuple(bool(x) for x in GL.glGetBooleanv(GL.GL_COLOR_WRITEMASK)),
        "depth_range": tuple(float(x) for x in GL.glGetFloatv(GL.GL_DEPTH_RANGE)),
        "depth_clear": float(GL.glGetFloatv(GL.GL_DEPTH_CLEAR_VALUE)),
    }
    renderer = CharacterRenderer(model, config)
    renderer.initialize()
    first_counts = renderer.resource_counts
    renderer.initialize()
    assert renderer.resource_counts == first_counts
    image, metadata = renderer.render_frame(0, 0.0, "frame_0000.png")
    after = {
        "viewport": tuple(int(x) for x in GL.glGetIntegerv(GL.GL_VIEWPORT)),
        "depth_test": bool(GL.glIsEnabled(GL.GL_DEPTH_TEST)),
        "cull": bool(GL.glIsEnabled(GL.GL_CULL_FACE)),
        "depth_func": int(GL.glGetIntegerv(GL.GL_DEPTH_FUNC)),
        "pack_alignment": int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT)),
        "blend": bool(GL.glIsEnabled(GL.GL_BLEND)),
        "scissor": bool(GL.glIsEnabled(GL.GL_SCISSOR_TEST)),
        "stencil": bool(GL.glIsEnabled(GL.GL_STENCIL_TEST)),
        "rasterizer_discard": bool(GL.glIsEnabled(GL.GL_RASTERIZER_DISCARD)),
        "framebuffer_srgb": bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB)),
        "color_mask": tuple(bool(x) for x in GL.glGetBooleanv(GL.GL_COLOR_WRITEMASK)),
        "depth_range": tuple(float(x) for x in GL.glGetFloatv(GL.GL_DEPTH_RANGE)),
        "depth_clear": float(GL.glGetFloatv(GL.GL_DEPTH_CLEAR_VALUE)),
    }
    assert after == before
    assert image.shape == (config.height, config.width, 4)
    assert len(np.unique(image.reshape(-1, 4), axis=0)) > 50
    assert metadata["frame_index"] == 0
    renderer.close()
    renderer.close()
    assert renderer.resource_counts == {
        "textures": 0,
        "buffers": 0,
        "vertex_arrays": 0,
        "framebuffers": 0,
        "programs": 0,
    }


def test_render_sequence_writes_complete_png_and_metadata(tmp_path: Path) -> None:
    model = load_model(SAMPLE)
    config = _small_config()
    output = tmp_path / "sequence"
    try:
        manifest = render_sequence(model, config, output)
    except ContextError as exc:
        pytest.skip(f"OpenGL context unavailable: {exc}")
    assert manifest["status"] == "complete"
    assert len(manifest["frames"]) == config.frame_count
    images = []
    for index in range(config.frame_count):
        image_path = output / f"frame_{index:04d}.png"
        metadata_path = output / f"frame_{index:04d}.json"
        assert image_path.exists()
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["frame_index"] == index
        assert metadata["time_seconds"] == pytest.approx(index * config.time_step)
        assert len(metadata["camera"]["view_projection_matrix"]) == 16
        assert len(metadata["lights"]["point_lights"]) <= config.max_point_lights
        assert metadata["image_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
        images.append(image_path.read_bytes())
    assert images[0] != images[1]
    assert (output / "manifest.json").exists()
    assert not (output / "render_report.json").exists()


def test_shadow_params_reduce_light_without_adding_light(gl_context) -> None:
    model = load_model(SAMPLE)
    config = replace(
        _small_config(),
        frame_count=1,
        width=96,
        height=96,
        max_point_lights=2,
    )
    no_shadow = replace(
        model,
        directional=replace(model.directional, cast_shadow=False),
        point_lights=tuple(
            replace(light, cast_shadow=False) for light in model.point_lights
        ),
    )
    with_shadow = replace(
        model,
        directional=replace(model.directional, cast_shadow=True),
        point_lights=tuple(
            replace(light, cast_shadow=True) for light in model.point_lights
        ),
    )
    rendered = []
    for scene in (no_shadow, with_shadow):
        renderer = CharacterRenderer(scene, config)
        renderer.initialize()
        try:
            image, _ = renderer.render_frame(0, 0.0, "shadow.png")
            rendered.append(image[..., :3].astype(np.int16))
        finally:
            renderer.close()
    darker = np.count_nonzero(
        np.any(rendered[1] < rendered[0] - 2, axis=2)
    )
    brighter = np.count_nonzero(
        np.any(rendered[1] > rendered[0] + 2, axis=2)
    )
    assert darker > 100
    assert brighter == 0


def test_failed_frame_preserves_previous_outputs(tmp_path: Path, monkeypatch) -> None:
    model = load_model(SAMPLE)
    config = replace(_small_config(), width=96, height=96, frame_count=2)
    output = tmp_path / "failed_sequence"
    original = CharacterRenderer.render_frame

    def fail_second(self, frame_index, time_seconds, image_name):
        if frame_index == 1:
            raise RenderError("E_TEST_FRAME", "injected frame failure")
        return original(self, frame_index, time_seconds, image_name)

    monkeypatch.setattr(CharacterRenderer, "render_frame", fail_second)
    with pytest.raises(RenderError) as error:
        try:
            render_sequence(model, config, output)
        except ContextError as exc:
            pytest.skip(f"OpenGL context unavailable: {exc}")
    assert error.value.code == "E_TEST_FRAME"
    assert (output / "frame_0000.png").exists()
    assert (output / "frame_0000.json").exists()
    assert not (output / "frame_0001.png").exists()
    report = json.loads((output / "render_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"]["code"] == "E_TEST_FRAME"
    assert report["completed_frame_count"] == 1

