"""Linear HDR to ACES-filmic sRGB display encoding."""

from __future__ import annotations

import colorsys

import numpy as np


ACES_A = 2.51
ACES_B = 0.03
ACES_C = 2.43
ACES_D = 0.59
ACES_E = 0.14


def aces_filmic(linear_rgb: np.ndarray) -> np.ndarray:
    """Apply the Narkowicz ACES filmic approximation per RGB channel.

    Formula:
        x = max(linear_rgb, 0)
        y = clamp((x * (A*x + B)) / (x * (C*x + D) + E), 0, 1)

    This is a display transform, not a physically calibrated ODT. Its exact
    coefficients are part of this project's versioned color contract.
    """

    x = np.maximum(np.asarray(linear_rgb, dtype=np.float64), 0.0)
    numerator = x * (ACES_A * x + ACES_B)
    denominator = x * (ACES_C * x + ACES_D) + ACES_E
    return np.clip(numerator / denominator, 0.0, 1.0)


def srgb_encode(linear_display: np.ndarray) -> np.ndarray:
    """Encode display-linear [0,1] values with the sRGB transfer function."""

    x = np.clip(np.asarray(linear_display, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )
    return np.clip(encoded, 0.0, 1.0)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values to display-linear [0,1] values."""

    x = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb8(linear_hdr: np.ndarray) -> np.ndarray:
    """Convert linear HDR RGB to an 8-bit ACES/sRGB display image."""

    display = srgb_encode(aces_filmic(linear_hdr))
    return np.clip(np.rint(display * 255.0), 0.0, 255.0).astype(np.uint8)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance for a trailing RGB axis."""

    rgb = np.asarray(linear_rgb, dtype=np.float64)
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )


def rgb_to_hue_degrees(rgb: tuple[float, float, float]) -> float | None:
    """Return the HSV hue of an RGB value, or None for black."""

    r, g, b = (float(v) for v in rgb)
    peak = max(r, g, b)
    if peak <= 0.0:
        return None
    # Hue is scale-invariant, so normalization also makes the result safe for
    # HDR values greater than one.
    hue, _, _ = colorsys.rgb_to_hsv(r / peak, g / peak, b / peak)
    return float(hue * 360.0)

