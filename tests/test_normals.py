import numpy as np

from terrain_gen.normals import compute_normals, encode_normals_rgb, sobel_gradients


def gaussian_bump(size=64):
    axis = np.arange(size, dtype=np.float64) - size / 2
    xx, yy = np.meshgrid(axis, axis)
    return np.exp(-(xx**2 + yy**2) / (2.0 * (size / 8.0) ** 2))


def test_normals_are_unit_length_and_up_facing():
    height = gaussian_bump()
    normals = compute_normals(height)
    lengths = np.linalg.norm(normals, axis=2)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-12)
    assert (normals[..., 2] > 0.0).all()


def test_normal_agrees_with_sobel_gradient_angle():
    height = gaussian_bump()
    dz_dx, dz_dy = sobel_gradients(height)
    reference = np.stack([-dz_dx, -dz_dy, np.ones_like(height)], axis=2)
    reference /= np.linalg.norm(reference, axis=2, keepdims=True)
    normals = compute_normals(height)
    cosine = np.clip(np.sum(normals * reference, axis=2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine))
    assert angle_deg.max() < 2.0


def test_encoded_normal_rgb_is_8bit_and_centered():
    height = gaussian_bump()
    rgb = encode_normals_rgb(compute_normals(height))
    assert rgb.dtype == np.uint8
    assert rgb.shape == (*height.shape, 3)
    # Flat regions encode to the neutral OpenGL tangent normal (128, 128, 255).
    flat = np.zeros((32, 32))
    flat_rgb = encode_normals_rgb(compute_normals(flat))
    np.testing.assert_array_equal(flat_rgb[..., 0], 128)
    np.testing.assert_array_equal(flat_rgb[..., 1], 128)
    np.testing.assert_array_equal(flat_rgb[..., 2], 255)


def test_normal_quantization_angle_still_within_tolerance():
    height = gaussian_bump()
    normals = compute_normals(height)
    rgb = encode_normals_rgb(normals).astype(np.float64)
    decoded = rgb / 255.0 * 2.0 - 1.0
    decoded /= np.linalg.norm(decoded, axis=2, keepdims=True)
    cosine = np.clip(np.sum(decoded * normals, axis=2), -1.0, 1.0)
    assert np.degrees(np.arccos(cosine)).max() < 2.0
