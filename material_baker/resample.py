"""Normal decoding and separable Gaussian material downsampling."""

from __future__ import annotations

import math

import numpy as np


def _length_squared(vectors: np.ndarray) -> np.ndarray:
    return np.sum(vectors * vectors, axis=-1)


def normalize_vectors(
    vectors: np.ndarray, *, epsilon: float = 1.0e-2
) -> tuple[np.ndarray, int]:
    vectors = np.asarray(vectors, dtype=np.float64)
    lengths_sq = _length_squared(vectors)
    invalid = lengths_sq <= epsilon * epsilon
    count = int(np.count_nonzero(invalid))
    safe_lengths = np.sqrt(np.maximum(lengths_sq, epsilon * epsilon))
    normalized = vectors / safe_lengths[..., None]
    if count:
        normalized[invalid] = (0.0, 0.0, 1.0)
    return normalized, count


def decode_normal_map(
    encoded_rgb: np.ndarray, *, flip_green: bool = False
) -> tuple[np.ndarray, int]:
    """Decode tangent-space normals with Y-up and normalize them."""

    encoded = np.asarray(encoded_rgb, dtype=np.float64)
    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError("normal map must be an HxWx3 array")
    vectors = encoded * 2.0 - 1.0
    if flip_green:
        vectors = vectors.copy()
        vectors[:, :, 1] *= -1.0
    return normalize_vectors(vectors)


def _gaussian_kernel_for_scale(source_size: int, target_size: int) -> np.ndarray:
    scale = target_size / float(source_size)
    if scale >= 1.0:
        return np.asarray([1.0], dtype=np.float64)
    sigma = 0.5 * math.sqrt(1.0 / (scale * scale) - 1.0)
    if sigma < 1.0e-8:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _blur_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size <= 1:
        return image
    radius = kernel.size // 2
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="edge")
    result = np.zeros_like(image, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        if axis == 0:
            result += float(weight) * padded[offset : offset + image.shape[0], :, :]
        else:
            result += float(weight) * padded[:, offset : offset + image.shape[1], :]
    return result


def _bilinear_resize(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    if source_h == target_h and source_w == target_w:
        return image.copy()

    source_y = (np.arange(target_h, dtype=np.float64) + 0.5) * source_h / target_h - 0.5
    source_x = (np.arange(target_w, dtype=np.float64) + 0.5) * source_w / target_w - 0.5
    y0 = np.floor(source_y).astype(np.int64)
    x0 = np.floor(source_x).astype(np.int64)
    wy = source_y - y0
    wx = source_x - x0
    y0 = np.clip(y0, 0, source_h - 1)
    x0 = np.clip(x0, 0, source_w - 1)
    y1 = np.clip(y0 + 1, 0, source_h - 1)
    x1 = np.clip(x0 + 1, 0, source_w - 1)

    wy = wy[:, None, None]
    wx = wx[None, :, None]
    top = (
        image[y0[:, None], x0[None, :], :] * (1.0 - wx)
        + image[y0[:, None], x1[None, :], :] * wx
    )
    bottom = (
        image[y1[:, None], x0[None, :], :] * (1.0 - wx)
        + image[y1[:, None], x1[None, :], :] * wx
    )
    return top * (1.0 - wy) + bottom * wy


def resize_image(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an HxWxC float image using Gaussian prefiltering when downsampling."""

    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError("expected an HxW or HxWxC image")
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target dimensions must be positive")

    source_h, source_w = data.shape[:2]
    if source_h == target_h and source_w == target_w:
        result = data.copy()
    else:
        kernel_y = _gaussian_kernel_for_scale(source_h, target_h)
        kernel_x = _gaussian_kernel_for_scale(source_w, target_w)
        filtered = _blur_axis(data, kernel_y, axis=0)
        filtered = _blur_axis(filtered, kernel_x, axis=1)
        result = _bilinear_resize(filtered, target_h, target_w)
    if image.ndim == 2:
        return result[:, :, 0]
    return result

