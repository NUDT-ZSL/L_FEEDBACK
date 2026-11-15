import numpy as np
import pytest

from rfr.lighting import compute_vertex_colors
from rfr.pipeline import StateCache, DrawCall
from rfr.scene import Light


def test_lighting_clamps_and_keeps_alpha():
    positions = np.zeros((2, 3), dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
    base = np.array([1.0, 1.0, 1.0, 0.4], dtype=np.float32)
    light = Light(kind="point",
                  direction=np.array([0.0, -1.0, 0.0]),
                  position=np.array([0.0, 0.0, 2.0]),
                  color=np.ones(3), intensity=50.0, light_range=100.0)
    colors = compute_vertex_colors(
        positions, normals, base, 0.5, 0.0, [light],
        ambient=np.zeros(3), camera_position=np.array([0.0, 0.0, 5.0]))
    assert colors.shape == (2, 4) and colors.dtype == np.float32
    assert np.all(colors[:, :3] <= 1.0) and np.all(colors[:, :3] >= 0.0)
    assert np.all(colors[:, 3] == pytest.approx(0.4))


def test_back_facing_light_gives_no_diffuse():
    positions = np.zeros((1, 3), dtype=np.float32)
    normals = np.array([[0, 0, -1]], dtype=np.float32)
    directional = Light(kind="directional",
                        direction=np.array([0.0, 0.0, 1.0]),
                        position=np.zeros(3),
                        color=np.ones(3), intensity=2.0)
    colors = compute_vertex_colors(
        positions, normals, np.array([1, 1, 1, 1], np.float32),
        0.5, 0.0, [directional], ambient=np.zeros(3),
        camera_position=np.array([0.0, 0.0, 5.0]))
    # Light travel direction +z means it shines toward -z; normal points -z,
    # so the face is lit. Flip the normal to confirm the dark case instead.
    dark_normals = np.array([[0, 0, 1]], dtype=np.float32)
    dark = compute_vertex_colors(
        positions, dark_normals, np.array([1, 1, 1, 1], np.float32),
        0.5, 0.0, [directional], ambient=np.zeros(3),
        camera_position=np.array([0.0, 0.0, 5.0]))
    assert np.allclose(dark[0, :3], 0.0, atol=1e-6)
    assert colors.shape == (1, 4)


def test_state_cache_counts_only_real_changes():
    cache = StateCache()
    draw = DrawCall("n", np.zeros((3, 3), np.float32), np.zeros((3, 3), np.float32),
                    np.zeros((3, 4), np.float32), np.array([0, 1, 2], np.int32),
                    "opaque", True, "back")
    cache.apply(draw)
    first = cache.total_switches
    cache.apply(draw)  # identical state must not add switches
    assert cache.total_switches == first
    draw.blend = "alpha"
    cache.apply(draw)
    assert cache.switch_counts["blend"] == 2
    assert cache.total_switches == first + 1


def test_state_cache_rejects_bad_blend():
    import pytest
    with pytest.raises(ValueError):
        StateCache().set_blend("nonsense")
