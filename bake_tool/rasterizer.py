from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, normalized_rows


@dataclass
class RasterResult:
    coverage_mask: np.ndarray
    triangle_index: np.ndarray
    barycentric: np.ndarray
    culled_triangles: int
    rasterized_triangles: int
    clipped_triangles: int
    degenerate_uv_triangles: int
    candidate_assignments: int


def _edge(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (p[..., 0] - a[..., 0])


def rasterize_texture_space(
    mesh: Mesh,
    resolution: int,
    backface_culling: bool = True,
) -> RasterResult:
    width = height = int(resolution)
    coverage = np.zeros((height, width), dtype=bool)
    triangle_index = np.full((height, width), -1, dtype=np.int32)
    barycentric = np.zeros((height, width, 3), dtype=np.float64)

    positions = np.asarray(mesh.positions)
    normals = normalized_rows(mesh.normals)
    uvs = np.asarray(mesh.uvs)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    tri_pos = positions[triangles]
    tri_uv = uvs[triangles]
    face_normal = normalized_rows(
        np.cross(
            tri_pos[:, 1] - tri_pos[:, 0],
            tri_pos[:, 2] - tri_pos[:, 0],
        )
    )
    average_normal = normalized_rows(normals[triangles].mean(axis=1))
    backfacing = np.einsum("ij,ij->i", face_normal, average_normal) <= 0.0

    culled = 0
    rasterized = 0
    clipped = 0
    degenerate = 0
    assignments = 0

    for triangle_id in range(triangles.shape[0]):
        if backface_culling and bool(backfacing[triangle_id]):
            culled += 1
            continue

        uv0, uv1, uv2 = tri_uv[triangle_id]
        denominator = (uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (
            uv2[0] - uv0[0]
        ) * (uv1[1] - uv0[1])
        if abs(float(denominator)) <= 1.0e-14:
            degenerate += 1
            continue

        min_u = float(min(uv0[0], uv1[0], uv2[0]))
        max_u = float(max(uv0[0], uv1[0], uv2[0]))
        min_v = float(min(uv0[1], uv1[1], uv2[1]))
        max_v = float(max(uv0[1], uv1[1], uv2[1]))

        x_start = max(0, int(np.floor(min_u * width - 0.5)))
        x_stop = min(width - 1, int(np.floor(max_u * width - 0.5)))
        y_start = max(0, int(np.floor((1.0 - max_v) * height - 0.5)))
        y_stop = min(height - 1, int(np.floor((1.0 - min_v) * height - 0.5)))

        if (
            min_u * width - 0.5 < 0.0
            or max_u * width - 0.5 > width - 1.0
            or (1.0 - max_v) * height - 0.5 < 0.0
            or (1.0 - min_v) * height - 0.5 > height - 1.0
        ):
            clipped += 1

        if x_start > x_stop or y_start > y_stop:
            continue

        ys, xs = np.mgrid[y_start : y_stop + 1, x_start : x_stop + 1]
        u = (xs + 0.5) / width
        v = 1.0 - (ys + 0.5) / height
        points = np.stack((u, v), axis=-1)

        w0 = _edge(uv1, uv2, points) / denominator
        w1 = _edge(uv2, uv0, points) / denominator
        w2 = _edge(uv0, uv1, points) / denominator
        inside = (w0 >= -1.0e-10) & (w1 >= -1.0e-10) & (w2 >= -1.0e-10)
        if not np.any(inside):
            continue

        candidates = inside & (triangle_index[y_start : y_stop + 1, x_start : x_stop + 1] < 0)
        assignments += int(np.count_nonzero(candidates))
        if not np.any(candidates):
            continue

        py = ys[candidates]
        px = xs[candidates]
        triangle_index[py, px] = triangle_id
        barycentric[py, px, 0] = w0[candidates]
        barycentric[py, px, 1] = w1[candidates]
        barycentric[py, px, 2] = w2[candidates]
        coverage[py, px] = True
        rasterized += 1

    return RasterResult(
        coverage_mask=coverage,
        triangle_index=triangle_index,
        barycentric=barycentric,
        culled_triangles=culled,
        rasterized_triangles=rasterized,
        clipped_triangles=clipped,
        degenerate_uv_triangles=degenerate,
        candidate_assignments=assignments,
    )
