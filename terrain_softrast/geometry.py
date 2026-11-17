from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import ValidationError


@dataclass
class GeometryWarnings:
    codes: set[str]
    details: list[dict]


@dataclass
class TerrainGeometry:
    height: np.ndarray
    vertices: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    triangles: np.ndarray
    unit_scale: float
    height_scale: float
    warnings: GeometryWarnings

    @property
    def vertex_count(self) -> int:
        return int(self.height.size)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.vertices.reshape(-1, 3), axis=0), np.max(self.vertices.reshape(-1, 3), axis=0)


def _normalize_weight_array(weights: np.ndarray) -> tuple[np.ndarray, GeometryWarnings]:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 3:
        raise ValidationError("权重图必须为 (图层数, 高度, 宽度)")
    if not np.all(np.isfinite(weights)):
        raise ValidationError("权重图包含 NaN 或无穷值")
    sums = np.sum(weights, axis=0)
    if np.any(sums <= 1e-12):
        raise ValidationError("权重图存在权重和为零的像素")
    codes: set[str] = set()
    details: list[dict] = []
    if np.any(weights < -1e-12):
        codes.add("MATERIAL_WEIGHTS_NEGATIVE")
        details.append(
            {
                "code": "MATERIAL_WEIGHTS_NEGATIVE",
                "message": "权重图包含负权重，已按像素和归一化后继续渲染",
                "negative_count": int(np.count_nonzero(weights < -1e-12)),
            }
        )
    if np.any(np.abs(sums - 1.0) > 1e-6):
        codes.add("MATERIAL_WEIGHTS_NOT_NORMALIZED")
        details.append(
            {
                "code": "MATERIAL_WEIGHTS_NOT_NORMALIZED",
                "message": "权重图未归一化，已按像素和归一化后继续渲染",
                "max_abs_error": float(np.max(np.abs(sums - 1.0))),
                "unnormalized_pixel_count": int(np.count_nonzero(np.abs(sums - 1.0) > 1e-6)),
            }
        )
    return np.asarray(weights / sums[None, :, :], dtype=np.float64), GeometryWarnings(codes, details)


def build_geometry(
    height: np.ndarray,
    weight_maps: np.ndarray,
    unit_scale: float = 1.0,
    height_scale: float = 1.0,
) -> TerrainGeometry:
    height = np.asarray(height, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValidationError("高度场必须是至少 2x2 的二维数组")
    if not np.all(np.isfinite(height)):
        raise ValidationError("高度场包含 NaN 或无穷值")
    if not math.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValidationError("unit_scale 必须为正有限数")
    if not math.isfinite(height_scale) or height_scale <= 0.0:
        raise ValidationError("height_scale 必须为正有限数")
    weights = np.asarray(weight_maps, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1:] != height.shape:
        raise ValidationError(
            f"权重图尺寸不匹配: 期望 (*, {height.shape[0]}, {height.shape[1]}), 实际 {weights.shape}"
        )
    normalized_weights, warnings = _normalize_weight_array(weights)
    rows, columns = height.shape
    x = (np.arange(columns, dtype=np.float64) - (columns - 1) * 0.5) * unit_scale
    z = (np.arange(rows, dtype=np.float64) - (rows - 1) * 0.5) * unit_scale
    xx, zz = np.meshgrid(x, z)
    vertices = np.stack((xx, height * height_scale, zz), axis=-1)

    grad_z, grad_x = np.gradient(height * height_scale, unit_scale, edge_order=1)
    normals = np.stack((-grad_x, np.ones_like(height), -grad_z), axis=-1)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-15)

    base = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    a = base[:-1, :-1]
    b = base[1:, :-1]
    c = base[:-1, 1:]
    d = base[1:, 1:]
    triangles = np.empty(((rows - 1) * (columns - 1) * 2, 3), dtype=np.int64)
    triangles[0::2, 0] = a.ravel()
    triangles[0::2, 1] = b.ravel()
    triangles[0::2, 2] = c.ravel()
    triangles[1::2, 0] = b.ravel()
    triangles[1::2, 1] = d.ravel()
    triangles[1::2, 2] = c.ravel()
    return TerrainGeometry(
        height=height,
        vertices=vertices,
        normals=normals,
        weights=np.moveaxis(normalized_weights, 0, -1),
        triangles=triangles,
        unit_scale=float(unit_scale),
        height_scale=float(height_scale),
        warnings=warnings,
    )


def affected_triangle_indices(
    old_geometry: TerrainGeometry,
    new_geometry: TerrainGeometry,
    dirty_rect: tuple[int, int, int, int],
) -> np.ndarray:
    rows, columns = old_geometry.height.shape
    x0, y0, x1, y1 = dirty_rect
    changed_vertices = (
        (old_geometry.height[y0:y1, x0:x1] != new_geometry.height[y0:y1, x0:x1])
        | np.any(old_geometry.weights[y0:y1, x0:x1] != new_geometry.weights[y0:y1, x0:x1], axis=-1)
    )
    changed_mask = np.zeros_like(old_geometry.height, dtype=bool)
    changed_mask[y0:y1, x0:x1] = changed_vertices
    vertex_delta = np.linalg.norm(new_geometry.vertices - old_geometry.vertices, axis=-1)
    normal_delta = np.linalg.norm(new_geometry.normals - old_geometry.normals, axis=-1)
    changed_mask |= vertex_delta > 1e-14
    changed_mask |= normal_delta > 1e-14
    if not np.any(changed_mask):
        return np.zeros((0,), dtype=np.int64)
    cell_mask = np.zeros((rows - 1, columns - 1), dtype=bool)
    cell_mask |= changed_mask[:-1, :-1]
    cell_mask |= changed_mask[1:, :-1]
    cell_mask |= changed_mask[:-1, 1:]
    cell_mask |= changed_mask[1:, 1:]
    triangle_mask = np.repeat(cell_mask.ravel(), 2)
    return np.flatnonzero(triangle_mask).astype(np.int64)


def changed_world_bounds(
    old_geometry: TerrainGeometry,
    new_geometry: TerrainGeometry,
    triangle_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if triangle_indices.size == 0:
        empty = np.zeros(3, dtype=np.float64)
        return empty, empty
    old_indices = old_geometry.triangles[triangle_indices].reshape(-1)
    new_indices = new_geometry.triangles[triangle_indices].reshape(-1)
    points = np.concatenate(
        (old_geometry.vertices.reshape(-1, 3)[old_indices], new_geometry.vertices.reshape(-1, 3)[new_indices]),
        axis=0,
    )
    return np.min(points, axis=0), np.max(points, axis=0)
