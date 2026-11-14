from __future__ import annotations

import sys

import numpy as np
import pytest

from skin_bake.runtime import load_baked_asset
from skin_bake.software import render_software_frame, write_png


def test_software_viewer_renders_distinct_animation_frames(scene_path, baked_dir, tmp_path):
    asset = load_baked_asset(scene_path, baked_dir)
    first = render_software_frame(asset, 0, 480, 360)
    middle = render_software_frame(asset, 24, 480, 360)
    assert first.covered_pixels > 1000
    assert middle.covered_pixels > 1000
    assert first.sample_index != middle.sample_index
    assert not np.array_equal(first.image, middle.image)
    output = tmp_path / "preview.png"
    write_png(output, middle.image)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="EGL check is Linux-only")
def test_egl_opengl_renderer_samples_rg16f(scene_path, baked_dir, tmp_path):
    from skin_bake.egl_headless import render_egl_frame

    asset = load_baked_asset(scene_path, baked_dir)
    try:
        result = render_egl_frame(asset, asset.frame_time(12), 320, 240)
    except Exception as exc:
        pytest.skip(f"EGL/OpenGL context unavailable: {exc}")
    assert result.version.split()[0] >= "3.3" or tuple(
        int(part) for part in result.version.split()[0].split(".")[:2]
    ) >= (3, 3)
    assert result.sample_index == 16
    assert result.image.shape == (240, 320, 3)
    assert np.unique(result.image.reshape(-1, 3), axis=0).shape[0] > 8
    output = tmp_path / "egl.png"
    write_png(output, result.image)
    assert output.stat().st_size > 1000

