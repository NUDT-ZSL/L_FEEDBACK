"""Per-vertex normal recomputation on the skinned position buffer."""

from __future__ import annotations

import numpy as np

from .errors import ErrorLog

DEGENERATE_AREA_THRESHOLD = 1e-8


def recompute_normals(
    positions: np.ndarray,
    triangles: np.ndarray,
    frame: int,
    error_log: ErrorLog,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted per-vertex normals from triangle face normals.

    Degenerate triangles (area < 1e-8) are reported as E_DEGENERATE_TRIANGLE
    (capped per frame) and skipped; the rest of the frame is still processed.
    Returns (vertex_normals, face_normals).
    """
    tri_pos = positions[triangles]  # (M, 3, 3)
    edge1 = tri_pos[:, 1] - tri_pos[:, 0]
    edge2 = tri_pos[:, 2] - tri_pos[:, 0]
    crosses = np.cross(edge1, edge2)  # 2 * area * face_normal
    areas = 0.5 * np.linalg.norm(crosses, axis=1)

    normals = np.zeros_like(positions)
    face_normals = np.zeros((len(triangles), 3))
    for tri_idx in range(len(triangles)):
        if areas[tri_idx] < DEGENERATE_AREA_THRESHOLD:
            error_log.report_degenerate_triangle(
                frame, tri_idx, float(areas[tri_idx])
            )
            continue
        fn = crosses[tri_idx] / (2.0 * areas[tri_idx])
        face_normals[tri_idx] = fn
        # Unnormalized cross == area-weighted normal.
        for corner in triangles[tri_idx]:
            normals[corner] += crosses[tri_idx]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.array([0.0, 0.0, 1.0])
    safe = np.where(lengths < 1e-12, 0.0, 1.0)
    normals = normals * safe / np.where(lengths < 1e-12, 1.0, lengths)
    normals += (1.0 - safe) * fallback
    return normals, face_normals
