"""UV-derived tangent basis construction and tangent-to-object normal mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .errors import MaterialError
from .scene import Scene


@dataclass
class TangentField:
    tangent: np.ndarray
    bitangent: np.ndarray
    geometry_normal: np.ndarray
    handedness: np.ndarray
    island_id: np.ndarray
    island_meta: dict[int, dict[str, Any]]
    boundary_mask: np.ndarray
    seam_band_mask: np.ndarray
    pole_mask: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def object_normals(
        self, tangent_normal: np.ndarray, normal_strength: float = 1.0
    ) -> np.ndarray:
        if tangent_normal.shape[:2] != self.tangent.shape[:2]:
            raise MaterialError(
                "E007", "tangent-space normal dimensions do not match tangent field"
            )
        vector = np.asarray(tangent_normal, dtype=np.float32).copy()
        vector[:, :, :2] *= np.float32(normal_strength)
        length = np.linalg.norm(vector, axis=2, keepdims=True)
        vector = vector / np.maximum(length, np.float32(1e-8))
        object_vector = (
            self.tangent * vector[:, :, 0:1]
            + self.bitangent * vector[:, :, 1:2]
            + self.geometry_normal[None, None, :] * vector[:, :, 2:3]
        )
        object_length = np.linalg.norm(object_vector, axis=2, keepdims=True)
        return (
            object_vector / np.maximum(object_length, np.float32(1e-8))
        ).astype(np.float32)

    def object_normals_rows(
        self,
        tangent_normal: np.ndarray,
        start: int,
        end: int,
        normal_strength: float = 1.0,
    ) -> np.ndarray:
        vector = np.asarray(tangent_normal[start:end], dtype=np.float32).copy()
        vector[:, :, :2] *= np.float32(normal_strength)
        vector /= np.maximum(
            np.linalg.norm(vector, axis=2, keepdims=True), np.float32(1e-8)
        )
        object_vector = (
            self.tangent[start:end] * vector[:, :, 0:1]
            + self.bitangent[start:end] * vector[:, :, 1:2]
            + self.geometry_normal[None, None, :] * vector[:, :, 2:3]
        )
        object_vector /= np.maximum(
            np.linalg.norm(object_vector, axis=2, keepdims=True),
            np.float32(1e-8),
        )
        return object_vector.astype(np.float32)

    def object_normal_pixels(
        self,
        tangent_normal: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        normal_strength: float = 1.0,
    ) -> np.ndarray:
        vector = np.asarray(tangent_normal[ys, xs], dtype=np.float32).copy()
        vector[:, :2] *= np.float32(normal_strength)
        vector /= np.maximum(
            np.linalg.norm(vector, axis=1, keepdims=True), np.float32(1e-8)
        )
        object_vector = (
            self.tangent[ys, xs] * vector[:, 0:1]
            + self.bitangent[ys, xs] * vector[:, 1:2]
            + self.geometry_normal[None, :] * vector[:, 2:3]
        )
        object_vector /= np.maximum(
            np.linalg.norm(object_vector, axis=1, keepdims=True),
            np.float32(1e-8),
        )
        return object_vector.astype(np.float32)


def _orthogonalized_axes(
    geometry_normal: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    projected_u = axis_u - np.dot(axis_u, geometry_normal) * geometry_normal
    if float(np.linalg.norm(projected_u)) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, geometry_normal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        projected_u = fallback - np.dot(fallback, geometry_normal) * geometry_normal
        pole = True
    else:
        pole = False
    tangent_u = projected_u / np.linalg.norm(projected_u)

    projected_v = axis_v - np.dot(axis_v, geometry_normal) * geometry_normal
    projected_v -= np.dot(projected_v, tangent_u) * tangent_u
    if float(np.linalg.norm(projected_v)) < 1e-8:
        tangent_v = np.cross(geometry_normal, tangent_u)
        pole = True
    else:
        tangent_v = projected_v / np.linalg.norm(projected_v)
    return tangent_u, tangent_v, pole


def _parse_islands(
    raw_islands: Any, width: int, height: int
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    if raw_islands is None:
        island = np.zeros((height, width), dtype=np.int32)
        return island, {
            0: {
                "id": 0,
                "bounds": [0, 0, width, height],
                "flip_u": False,
                "flip_v": False,
            }
        }
    if not isinstance(raw_islands, list) or not raw_islands:
        raise MaterialError("E007", "surface.islands must be a non-empty array")

    island = np.full((height, width), -1, dtype=np.int32)
    metadata: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_islands):
        if not isinstance(raw, dict):
            raise MaterialError("E007", f"surface.islands[{index}] must be an object")
        bounds = raw.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise MaterialError(
                "E007",
                f"surface.islands[{index}].bounds must be [x, y, width, height]",
            )
        try:
            x, y, island_width, island_height = (int(v) for v in bounds)
        except (TypeError, ValueError) as exc:
            raise MaterialError(
                "E007", f"surface.islands[{index}].bounds must be integers"
            ) from exc
        if [x, y, island_width, island_height] != list(bounds):
            raise MaterialError(
                "E007", f"surface.islands[{index}].bounds must be integers"
            )
        if (
            island_width <= 0
            or island_height <= 0
            or x < 0
            or y < 0
            or x + island_width > width
            or y + island_height > height
        ):
            raise MaterialError(
                "E007",
                f"surface.islands[{index}].bounds lie outside the texture",
            )
        raw_id = raw.get("id", index)
        try:
            island_key = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise MaterialError(
                "E007", f"surface.islands[{index}].id must be an integer"
            ) from exc
        if island_key in metadata:
            raise MaterialError("E007", f"duplicate UV island id: {island_key}")
        flip_u = raw.get("flip_u", False)
        flip_v = raw.get("flip_v", False)
        if not isinstance(flip_u, bool) or not isinstance(flip_v, bool):
            raise MaterialError(
                "E007",
                f"surface.islands[{index}].flip_u/flip_v must be boolean",
            )
        region = island[y : y + island_height, x : x + island_width]
        if np.any(region >= 0):
            raise MaterialError(
                "E007", "UV island rectangles must not overlap"
            )
        region[:, :] = island_key
        metadata[island_key] = {
            "id": island_key,
            "bounds": [x, y, island_width, island_height],
            "flip_u": flip_u,
            "flip_v": flip_v,
        }
    if np.any(island < 0):
        raise MaterialError(
            "E007", "UV island rectangles must cover the complete texture"
        )
    return island, metadata


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def build_tangent_field(
    scene: Scene, dimensions: tuple[int, int]
) -> TangentField:
    height, width = dimensions
    geometry_normal = np.asarray(
        scene.surface["geometry_normal"], dtype=np.float32
    )
    geometry_normal /= np.maximum(
        np.linalg.norm(geometry_normal), np.float32(1e-12)
    )

    uv_axis_u = np.asarray(scene.surface["uv_axis_u"], dtype=np.float64)
    uv_axis_v = np.asarray(scene.surface["uv_axis_v"], dtype=np.float64)
    axis_u, axis_v, degenerate_global = _orthogonalized_axes(
        geometry_normal.astype(np.float64), uv_axis_u, uv_axis_v
    )

    island_id, island_meta = _parse_islands(
        scene.surface.get("islands"), width, height
    )
    tangent = np.empty((height, width, 3), dtype=np.float32)
    bitangent = np.empty_like(tangent)
    handedness = np.empty((height, width), dtype=np.int8)

    base_reference_bitangent = np.cross(geometry_normal, axis_u.astype(np.float32))
    base_reference_bitangent /= np.maximum(
        np.linalg.norm(base_reference_bitangent), np.float32(1e-12)
    )

    for island_key, meta in sorted(island_meta.items()):
        x, y, w, h = meta["bounds"]
        sign_u = -1.0 if meta["flip_u"] else 1.0
        sign_v = -1.0 if meta["flip_v"] else 1.0
        tangent_uv = (axis_u * sign_u).astype(np.float32)
        bitangent_uv = (axis_v * sign_v).astype(np.float32)
        t = tangent_uv / max(float(np.linalg.norm(tangent_uv)), 1e-12)
        b_uv = bitangent_uv / max(float(np.linalg.norm(bitangent_uv)), 1e-12)
        w_sign = 1.0 if float(np.dot(np.cross(geometry_normal, t), b_uv)) >= 0.0 else -1.0
        b = np.cross(geometry_normal, t) * w_sign
        b /= max(float(np.linalg.norm(b)), 1e-12)
        tangent[y : y + h, x : x + w, :] = t
        bitangent[y : y + h, x : x + w, :] = b
        handedness[y : y + h, x : x + w] = int(round(w_sign))
        meta["tangent_handedness"] = int(round(w_sign))

    boundary_mask = np.zeros((height, width), dtype=bool)
    if width > 1:
        horizontal = island_id[:, 1:] != island_id[:, :-1]
        boundary_mask[:, :-1] |= horizontal
        boundary_mask[:, 1:] |= horizontal
    if height > 1:
        vertical = island_id[1:, :] != island_id[:-1, :]
        boundary_mask[:-1, :] |= vertical
        boundary_mask[1:, :] |= vertical

    seam_radius = int(scene.validation["seam_width"])
    seam_band = _dilate_mask(boundary_mask, seam_radius)
    raw_tangent_length = float(np.linalg.norm(axis_u))
    pole_mask = np.full((height, width), degenerate_global, dtype=bool)
    if raw_tangent_length < 0.15:
        pole_mask[:, :] = True

    smoothing_radius = int(scene.surface["seam_smoothing_radius"])
    smoothed_count = 0
    if smoothing_radius > 0 and (np.any(seam_band) or np.any(pole_mask)):
        smooth_mask = seam_band | pole_mask
        sign_alignment = np.sign(
            np.einsum("ijk,k->ij", bitangent, base_reference_bitangent)
        )
        sign_alignment[sign_alignment == 0] = 1
        aligned = bitangent * sign_alignment[:, :, None]
        kernel = (2 * smoothing_radius + 1, 2 * smoothing_radius + 1)
        blurred = cv2.blur(
            aligned.astype(np.float32),
            kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        blurred_length = np.linalg.norm(blurred, axis=2)
        valid = (blurred_length > 1e-8) & smooth_mask
        if np.any(valid):
            smoothed_b = blurred[valid] / blurred_length[valid, None]
            normal_broadcast = np.broadcast_to(geometry_normal, smoothed_b.shape)
            normal_dot = np.sum(
                smoothed_b * normal_broadcast, axis=1, keepdims=True
            )
            smoothed_b -= normal_dot * normal_broadcast
            smoothed_b /= np.maximum(
                np.linalg.norm(smoothed_b, axis=1, keepdims=True),
                np.float32(1e-12),
            )
            smoothed_b *= sign_alignment[valid, None]
            candidate_t = np.cross(smoothed_b, normal_broadcast)
            candidate_t /= np.maximum(
                np.linalg.norm(candidate_t, axis=1, keepdims=True),
                np.float32(1e-12),
            )
            original_t = tangent[valid]
            orientation = np.sign(
                np.sum(original_t * candidate_t, axis=1, keepdims=True)
            )
            orientation[orientation == 0] = 1
            smoothed_t = candidate_t * orientation
            smoothed_t /= np.maximum(
                np.linalg.norm(smoothed_t, axis=1, keepdims=True),
                np.float32(1e-12),
            )
            smoothed_b -= (
                np.sum(smoothed_b * smoothed_t, axis=1, keepdims=True)
                * smoothed_t
            )
            smoothed_b /= np.maximum(
                np.linalg.norm(smoothed_b, axis=1, keepdims=True),
                np.float32(1e-12),
            )
            tangent[valid] = smoothed_t.astype(np.float32)
            bitangent[valid] = smoothed_b.astype(np.float32)
            handed = np.sum(
                np.cross(normal_broadcast, smoothed_t) * smoothed_b,
                axis=1,
            )
            handedness[valid] = np.where(handed >= 0.0, 1, -1)
            smoothed_count = int(np.count_nonzero(valid))

    info = {
        "method": "uv_grid_gradients_with_handedness_rebasing",
        "geometry_normal": [round(float(v), 9) for v in geometry_normal],
        "uv_axis_u": [round(float(v), 9) for v in scene.surface["uv_axis_u"]],
        "uv_axis_v": [round(float(v), 9) for v in scene.surface["uv_axis_v"]],
        "mirror_policy": (
            "bitangent sign is derived from dot(cross(N,T), dB/dv); "
            "U/V flips therefore change handedness automatically"
        ),
        "seam_smoothing_radius": smoothing_radius,
        "seam_smoothed_pixels": smoothed_count,
        "seam_width": seam_radius,
        "seam_boundary_pixels": int(np.count_nonzero(boundary_mask)),
        "seam_band_pixels": int(np.count_nonzero(seam_band)),
        "pole_pixels": int(np.count_nonzero(pole_mask)),
        "uv_axes_degenerate": bool(degenerate_global),
        "islands": [island_meta[key] for key in sorted(island_meta)],
    }
    return TangentField(
        tangent=tangent,
        bitangent=bitangent,
        geometry_normal=geometry_normal,
        handedness=handedness,
        island_id=island_id,
        island_meta=island_meta,
        boundary_mask=boundary_mask,
        seam_band_mask=seam_band,
        pole_mask=pole_mask,
        info=info,
    )
