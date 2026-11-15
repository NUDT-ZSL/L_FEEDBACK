import numpy as np

from terrain_gen.config import NoiseConfig
from terrain_gen.noise import generate_heightfield


def make_noise(seed):
    return NoiseConfig(seed=seed, base_frequency=8, octaves=5)


def test_shape_dtype_and_range():
    field = generate_heightfield(256, 300, make_noise(1))
    assert field.shape == (300, 256)
    assert field.dtype == np.float64
    assert field.min() == 0.0
    assert field.max() == 1.0
    assert np.isfinite(field).all()


def test_deterministic_same_seed():
    first = generate_heightfield(256, 256, make_noise(99))
    second = generate_heightfield(256, 256, make_noise(99))
    np.testing.assert_array_equal(first, second)


def test_seed_changes_output():
    first = generate_heightfield(256, 256, make_noise(1))
    second = generate_heightfield(256, 256, make_noise(2))
    assert not np.array_equal(first, second)


def test_periodic_seam_is_continuous():
    field = generate_heightfield(256, 256, make_noise(3))
    vertical_jumps = np.abs(np.diff(field, axis=0))
    horizontal_jumps = np.abs(np.diff(field, axis=1))
    wrap_vertical = np.abs(field[0, :] - field[-1, :])
    wrap_horizontal = np.abs(field[:, 0] - field[:, -1])
    # Seam neighbours are lattice-adjacent just like interior neighbours.
    assert wrap_vertical.mean() <= 2.0 * vertical_jumps.mean()
    assert wrap_horizontal.mean() <= 2.0 * horizontal_jumps.mean()
    assert wrap_vertical.max() <= vertical_jumps.max() + 1e-12
    assert wrap_horizontal.max() <= horizontal_jumps.max() + 1e-12
