"""Linear blend skinning and normal transformation."""

from __future__ import annotations

import numpy as np

from .errors import ErrorLog, E_NON_INVERTIBLE_MATRIX


def skin_frame(
    vertices: np.ndarray,
    weights: np.ndarray,
    skin_mats: list[np.ndarray],
    frame: int,
    error_log: ErrorLog,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply linear blend skinning for one frame.

    Positions use the weight-blended 4x4 matrix. Normals use each bone's
    3x3 inverse-transpose, blended with the same weights and normalized.
    Returns (positions, normals) float64, or None if any bone 3x3 block is
    singular (reported as E_NON_INVERTIBLE_MATRIX; frame is skipped).
    """
    normal_mats: list[np.ndarray] = []
    for bone_idx, m in enumerate(skin_mats):
        upper = m[:3, :3]
        try:
            inv = np.linalg.inv(upper)
        except np.linalg.LinAlgError:
            error_log.report(
                frame,
                E_NON_INVERTIBLE_MATRIX,
                f"bone {bone_idx} 3x3 block is singular; frame skipped",
            )
            return None
        normal_mats.append(inv.T)

    blended = weights[:, 0, None, None] * skin_mats[0] \
        + weights[:, 1, None, None] * skin_mats[1]
    hom = np.concatenate(
        [vertices, np.ones((len(vertices), 1))], axis=1
    )
    positions = np.einsum("nij,nj->ni", blended, hom)[:, :3]

    bind_normals = np.tile(
        np.array([0.0, 0.0, 1.0]), (len(vertices), 1)
    )
    n0 = bind_normals @ normal_mats[0].T
    n1 = bind_normals @ normal_mats[1].T
    blended_n = weights[:, 0:1] * n0 + weights[:, 1:2] * n1
    lengths = np.linalg.norm(blended_n, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    normals = blended_n / lengths
    return positions, normals
