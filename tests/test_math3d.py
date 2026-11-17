from __future__ import annotations

import numpy as np

from character_preview.math3d import (
    compose,
    from_column_major,
    look_at,
    perspective,
    rotation_y,
    to_column_major,
    translation,
)


def test_column_major_round_trip_preserves_matrix() -> None:
    matrix = compose(translation(1.5, -2.0, 3.0), rotation_y(17.0))
    restored = from_column_major(to_column_major(matrix))
    np.testing.assert_allclose(restored, matrix, atol=1e-12)


def test_camera_projection_keeps_target_near_center() -> None:
    eye = np.array([3.1, 2.05, 4.15])
    target = np.array([0.0, 1.08, 0.0])
    view_projection = perspective(38.0, 1.0, 0.05, 100.0) @ look_at(
        eye, target, np.array([0.0, 1.0, 0.0])
    )
    clip = view_projection @ np.array([*target, 1.0])
    normalized = clip[:3] / clip[3]
    assert abs(normalized[0]) < 1e-6
    assert abs(normalized[1]) < 1e-6
    assert -1.0 < normalized[2] < 1.0

