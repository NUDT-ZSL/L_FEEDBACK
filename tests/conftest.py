from __future__ import annotations

import numpy as np
import pytest

from terrain_softrast.config import (
    Camera,
    CascadeSettings,
    DirectionalLight,
    RenderSettings,
)
from terrain_softrast.samples import generate_heightfield, generate_weight_maps


@pytest.fixture(scope="session")
def deterministic_height() -> np.ndarray:
    return generate_heightfield(17, 1337)


@pytest.fixture(scope="session")
def deterministic_weights(deterministic_height: np.ndarray) -> np.ndarray:
    return generate_weight_maps(deterministic_height, 4, 4242)


@pytest.fixture(scope="session")
def ridge_case():
    size = 25
    height = np.zeros((size, size), dtype=np.float64)
    height[:, 9:12] = 1.0
    weights = generate_weight_maps(height, 4, 4242)
    camera = Camera(eye=(0.0, 8.0, 11.0), target=(0.0, 0.2, 0.0), fov_y_deg=50.0, far=80.0)
    light = DirectionalLight(direction=(-1.0, -0.35, -0.1), color=(1.0, 0.97, 0.9), intensity=1.2)
    settings = RenderSettings(
        width=192,
        height=144,
        tile_size=16,
        cascades=CascadeSettings(count=3, shadow_size=128, split_lambda=0.68, overlap_fraction=0.12),
    )
    return {
        "height": height,
        "weights": weights,
        "camera": camera,
        "light": light,
        "settings": settings,
        "unit_scale": 0.4,
        "height_scale": 2.5,
    }
